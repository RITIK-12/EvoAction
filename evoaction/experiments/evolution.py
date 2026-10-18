"""Version N -> N+1 controller.

Label-free mode (default): agents and the promotion gate see only label-free signals (evoaction.selfcheck).
Human-in-the-loop mode (--hitl): agents additionally see human reference segments for a fixed ~10% labeled
sample, and the gate uses that sample; the other ~90% stay unlabeled and are measured only for the report.
In both modes ground truth on unlabeled episodes is a "shadow" measurement that never enters agent context,
agent history, or the gate. Rejected candidates are preserved immutably and never change the incumbent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

from evoaction.agents import investigator as investigator_mod
from evoaction.agents import supervisor as supervisor_mod
from evoaction.agents.investigator import TaskGraph, dispatch, tree, unfinished_tasks
from evoaction.agents.supervisor import plan_investigations, propose_experiment, workflow_summary
from evoaction.agents.tools import ToolContext
from evoaction.clients.crusoe import BudgetExceeded, CrusoeClient, ModelCallError
from evoaction.config import Settings, get_settings
from evoaction.data.wgo import calibration_split, episode_set_ids
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.evaluation.segments import paired_comparison
from evoaction.persistence import get_store
from evoaction.pipeline import ensure_n0, headline, run_and_assess
from evoaction.schemas import WorkflowPatch
from evoaction.selfcheck.label_free import SCORE_DEFINITION, compact_assessment
from evoaction.workflow.engine import Version, VersionStore, WorkflowValidationError, apply_patch, run_version, usage_for

PROTOCOL = "v3-macrodata"
GATE = {
    "signal": SCORE_DEFINITION["version"],
    "min_self_score_delta": 0.03,
    "min_nonregressing_fraction": 0.5,
    "require_valid_outputs": True,
    "cost_policy": "no per-episode cost cap; total cost is recorded per recursion (iteration) and per version over the same 50 episodes",
    "note": "label-free gate predeclared before the v3 run; ground truth is never consulted for promotion",
}
HISTORY_KEYS = ("id", "parent_version", "hypothesis", "patch_summary", "decision", "label_free", "labeled_sample")
GATE_HITL = {
    "signal": "human-labeled sample (~10% of the batch, whole instruction groups, fixed before the run)",
    "rule": "pooled snapped Segment F1@0.75 on the labeled sample strictly improves (ties: raw F1@0.75 strictly improves); valid outputs on every episode",
    "cost_policy": GATE["cost_policy"],
    "note": "the ~90% held-out episodes are never labeled for agents or the gate; they are measured only for the report",
}


def set_suffix(set_name: str, hitl: bool = False) -> str:
    return ("" if set_name == "droid50" else f"_{set_name}") + ("_hitl" if hitl else "")


def lineage_path(s: Settings, set_name: str = "droid50", hitl: bool = False):
    return s.artifacts_dir / f"lineage{set_suffix(set_name, hitl)}.json"


def load_lineage(s: Settings, set_name: str = "droid50", hitl: bool = False) -> dict[str, Any]:
    p = lineage_path(s, set_name, hitl)
    if p.exists():
        return json.loads(p.read_text())
    return {"protocol": PROTOCOL, "episode_set": set_name, "incumbent": "N0", "next_experiment": 1, "next_version": 1,
            "experiments": [], "versions": {}}


def save_lineage(s: Settings, lin: dict[str, Any], set_name: str = "droid50", hitl: bool = False) -> None:
    lineage_path(s, set_name, hitl).write_text(json.dumps(lin, indent=1, default=str))


def cost_profile(result: dict[str, Any]) -> dict[str, Any]:
    eps = result["run"]["episodes"].values()
    costs = [e.get("inference_cost_usd", 0.0) for e in eps]
    toks = [e.get("inference_tokens", {}) for e in eps]
    return {"episodes": len(costs), "mean_annotation_cost_usd_per_episode": round(sum(costs) / max(1, len(costs)), 6),
            "mean_completion_tokens_per_episode": round(sum(t.get("completion", 0) for t in toks) / max(1, len(toks))),
            "mean_prompt_tokens_per_episode": round(sum(t.get("prompt", 0) for t in toks) / max(1, len(toks))),
            "mean_vision_calls_per_episode": round(sum(e["calls_used"] for e in eps) / max(1, len(costs)), 2),
            "note": ("measured on this provider: thinking=true multi-image calls returned invalid JSON in 31/42 attempts (token cap "
                     "exhausted while reasoning) and took 40-170s; thinking=false returned valid JSON in 2-11s with ~10x fewer tokens "
                     "but tended toward evenly spaced segment boundaries; each contact-sheet image is encoded to ~570 prompt tokens "
                     "whatever its frame count (1/2/4/11 sheets -> 601/1173/2307/6138 prompt tokens), i.e. ~28 tokens per frame at 20 frames per sheet")}


def decide(cand: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
    """Label-free promotion gate. Uses only self-assessments, run statuses and cost."""
    ca, ia = cand["assessment"], inc["assessment"]
    paired = paired_comparison(ca["per_episode_self_score"], ia["per_episode_self_score"])
    delta = ca["mean_self_score"] - ia["mean_self_score"]
    bad_status = {e: s for e, s in cand["statuses"].items() if s not in ("ok", "partial")}
    empty = [e for e, r in cand["run"]["episodes"].items() if not r["segments"]]
    checks = {
        "self_score_delta": {"value": round(delta, 4), "threshold": GATE["min_self_score_delta"], "pass": delta >= GATE["min_self_score_delta"]},
        "non_regressing_fraction": {"value": paired["non_regressing_fraction"], "threshold": GATE["min_nonregressing_fraction"],
                                    "pass": paired["non_regressing_fraction"] >= GATE["min_nonregressing_fraction"]},
        "valid_outputs": {"value": {"bad_status": bad_status, "empty_output_episodes": empty}, "pass": not bad_status and not empty},
    }
    promote = all(c["pass"] for c in checks.values())
    return {"decision": "PROMOTED" if promote else "REJECTED", "gate": GATE, "checks": checks, "paired_self_score": paired,
            "label_free": {"candidate_self_score": ca["mean_self_score"], "incumbent_self_score": ia["mean_self_score"],
                           "candidate_components": ca["component_means"], "incumbent_components": ia["component_means"]},
            "uncertainty": f"n={len(paired['episodes'])} episodes (same set for both versions); label-free proxy, not ground truth"}


def shadow_gt(cand: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
    """Independent GT measurement for the report only (computed after the decision; never shown to agents)."""
    return {"candidate": headline(cand.get("gt_measurement")), "incumbent": headline(inc.get("gt_measurement"))}


def subset_gt(r: dict[str, Any], ids: list[str], s: Settings | None = None) -> dict[str, Any]:
    """Controller-only GT measurement of a run restricted to a subset of its episodes."""
    eps = r["run"]["episodes"]
    ev = ProtectedEvaluator.for_episode_set(ids, s)
    return headline(ev.score_split({e: eps[e]["segments"] for e in ids}, {e: eps[e]["duration_sec"] for e in ids}))


def labeled_feedback(r: dict[str, Any], ids: list[str], s: Settings | None = None) -> dict[str, Any]:
    eps = r["run"]["episodes"]
    ev = ProtectedEvaluator.for_episode_set(ids, s)
    return ev.labeled_sample_feedback({e: eps[e]["segments"] for e in ids}, {e: eps[e]["duration_sec"] for e in ids})


INNER_ROUNDS = 3  # main-agent refinement rounds per recursion, each tried on the labeled sample


def _med(v: list[float]) -> float | None:
    return round(float(statistics.median(v)), 2) if v else None


def _frac(v: list[float], pred) -> float | None:
    return round(sum(1 for x in v if pred(x)) / len(v), 3) if v else None


def root_cause_analysis(r: dict[str, Any], calib_ids: list[str], s: Settings | None = None) -> dict[str, Any]:
    """Deterministic error analysis: metric-aligned stats on the labeled sample + label-free structure on all episodes."""
    fb = labeled_feedback(r, calib_ids, s)
    eps = list(fb["episodes"].values())
    pd = [b - a for e in eps for a, b in e["predictions"]]
    rd = [x["reference"][1] - x["reference"][0] for e in eps for x in e["references"]]
    cov = [max((b for _, b in e["predictions"]), default=0.0) / max(x["reference"][1] for x in e["references"]) for e in eps if e["references"]]
    best = [x["best_iou"] for e in eps for x in e["references"]]
    stats = {"predicted_vs_reference_segments": [len(pd), len(rd)], "median_predicted_segment_sec": _med(pd), "median_reference_segment_sec": _med(rd),
             "predicted_segments_le_2s_fraction": _frac(pd, lambda x: x <= 2.0), "reference_segments_le_2s_fraction": _frac(rd, lambda x: x <= 2.0),
             "predicted_segments_exactly_1s_fraction": _frac(pd, lambda x: abs(x - 1.0) < 1e-6),
             "median_coverage_last_predicted_end_over_reference_end": _med(cov), "episodes_with_coverage_below_50pct": sum(1 for c in cov if c < 0.5),
             "references_matched_at_iou075": fb["pooled"]["tp_snapped"], "mean_best_iou_per_reference": round(sum(best) / len(best), 3) if best else None,
             "references_with_best_iou_below_0.5": sum(1 for b in best if b < 0.5), "pooled_scores": fb["pooled"]}
    allr = r["run"]["episodes"]
    sheets = [x.get("n_sheets") for v in allr.values() for x in (v.get("trace") or []) if x.get("operator") == "sample_contact_sheets" and x.get("n_sheets")]
    ad = [x["end_sec"] - x["start_sec"] for v in allr.values() for x in v["segments"]]
    acov = [max((x["end_sec"] for x in v["segments"]), default=0.0) / v["duration_sec"] for v in allr.values() if v.get("duration_sec")]
    durs = [v["duration_sec"] for v in allr.values() if v.get("duration_sec")]
    struct = {"episodes": len(allr), "episode_duration_sec": {"min": round(min(durs), 1), "median": _med(durs), "max": round(max(durs), 1)} if durs else None,
              "contact_sheet_images_per_call_at_current_sampling": {"median": _med(sheets), "max": max(sheets)} if sheets else None,
              "predicted_segments_per_episode_mean": round(len(ad) / max(1, len(allr)), 2), "median_predicted_segment_sec": _med(ad),
              "predicted_segments_exactly_1s_fraction": _frac(ad, lambda x: abs(x - 1.0) < 1e-6),
              "median_coverage_last_predicted_end_over_video_duration": _med(acov),
              "episodes_with_predictions_ending_before_half_the_video": sum(1 for c in acov if c < 0.5),
              "empty_or_failed_episodes": sum(1 for v in allr.values() if not v["segments"])}
    return {"labeled_sample": fb, "root_cause_stats_labeled": stats, "structure_all_episodes": struct}


def trial_on_labeled(incumbent: Version, wf, prompts: dict[str, str], trial_id: str, calib_ids: list[str], client: CrusoeClient,
                     run_id: str, store, s: Settings) -> dict[str, Any]:
    v = Version(id=trial_id, path=incumbent.path, workflow=wf, prompts=prompts, config=incumbent.config, manifest={"workflow_sha256": "labeled-trial"})
    res = run_version(v, calib_ids, client, run_id, store=store, settings=s)
    out = root_cause_analysis({"run": res}, calib_ids, s)
    out["failed_episodes"] = sorted(e for e, x in res["episodes"].items() if x["status"] not in ("ok", "partial"))
    return out


def hitl_propose(client: CrusoeClient, store, run_id: str, exp_id: str, incumbent: Version, analysis: dict[str, Any],
                 history: list[dict[str, Any]], cprof: dict[str, Any], calib_ids: list[str], s: Settings) -> dict[str, Any]:
    """Main agent diagnoses root causes and rewrites the harness, refining over INNER_ROUNDS on the fixed labeled sample."""
    trials: list[dict[str, Any]] = []
    best = None
    root_causes = None
    for k in range(INNER_ROUNDS):
        try:
            data = supervisor_mod.engineer_patch(client, incumbent, analysis, history, cprof, trials, f"hitl_engineer_round{k + 1}")
        except (BudgetExceeded, ModelCallError) as e:
            trials.append({"round": k + 1, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            store.event(run_id, "hitl_trial", {"exp_id": exp_id, **trials[-1]})
            if best is not None or k == INNER_ROUNDS - 1:
                break
            continue
        root_causes = root_causes or data.get("root_causes")
        try:
            patch = WorkflowPatch(**(data.get("candidate_patch") or {}))
            wf, prompts = apply_patch(incumbent.workflow, incumbent.prompts, patch)
        except (WorkflowValidationError, ValueError, TypeError) as e:
            trials.append({"round": k + 1, "hypothesis": data.get("hypothesis"), "error": f"invalid patch: {str(e)[:600]}"})
            store.event(run_id, "hitl_trial", {"exp_id": exp_id, **trials[-1]})
            continue
        try:
            ta = trial_on_labeled(incumbent, wf, prompts, f"{exp_id}-trial{k + 1}", calib_ids, client, run_id, store, s)
        except (BudgetExceeded, ModelCallError) as e:
            trials.append({"round": k + 1, "hypothesis": data.get("hypothesis"), "error": f"{type(e).__name__}: {str(e)[:300]}"})
            store.event(run_id, "hitl_trial", {"exp_id": exp_id, **trials[-1]})
            break
        pooled = ta["labeled_sample"]["pooled"]
        tr = {"round": k + 1, "hypothesis": data.get("hypothesis"), "root_causes": data.get("root_causes"), "patch": data.get("candidate_patch"),
              "labeled_scores": pooled, "failed_labeled_episodes": ta["failed_episodes"], "root_cause_stats_labeled": ta["root_cause_stats_labeled"],
              "predictions_by_episode": {e: v["predictions"] for e, v in ta["labeled_sample"]["episodes"].items()}}
        trials.append(tr)
        store.event(run_id, "hitl_trial", {"exp_id": exp_id, "round": k + 1, "hypothesis": data.get("hypothesis"), "labeled_scores": pooled,
                                           "failed_labeled_episodes": ta["failed_episodes"],
                                           "root_cause_stats_labeled": ta["root_cause_stats_labeled"], "patch": data.get("candidate_patch")})
        # a round with failed episodes must not win just because fewer predictions inflate F1
        score = (not ta["failed_episodes"], pooled["snapped_f1_075"], pooled["raw_f1_075"])
        if best is None or score > best[0]:
            best = (score, data, tr)
    if best is None:
        return {"status": "invalid", "attempts": [{"errors": [x.get("error") for x in trials]}], "trials": trials, "root_causes": root_causes}
    data = best[1]
    proposal = {"hypothesis": data.get("hypothesis"), "candidate_patch": data.get("candidate_patch"), "root_causes": data.get("root_causes") or root_causes,
                "expected_outcome": data.get("expected_outcome"), "selected_round": best[2]["round"], "evidence_ids": [],
                "incumbent_labeled_scores": analysis["labeled_sample"]["pooled"],
                "trials": [{k: v for k, v in x.items() if k != "predictions_by_episode"} for x in trials]}
    return {"status": "valid", "proposal": proposal, "trials": trials}


def decide_hitl(cand: dict[str, Any], inc: dict[str, Any], cand_fb: dict[str, Any], inc_fb: dict[str, Any]) -> dict[str, Any]:
    """Human-in-the-loop gate: uses only the labeled sample (plus output validity on every episode)."""
    c, i = cand_fb["pooled"], inc_fb["pooled"]
    d75, draw = c["snapped_f1_075"] - i["snapped_f1_075"], c["raw_f1_075"] - i["raw_f1_075"]
    improves = d75 > 1e-9 or (abs(d75) <= 1e-9 and draw > 1e-9)
    bad_status = {e: st for e, st in cand["statuses"].items() if st not in ("ok", "partial")}
    empty = [e for e, r in cand["run"]["episodes"].items() if not r["segments"]]
    checks = {"labeled_sample_snapped_f1_075": {"value": {"delta": round(d75, 4), "raw_delta": round(draw, 4)},
                                                "threshold": "> 0 (ties: raw F1@0.75 > 0)", "pass": improves},
              "valid_outputs": {"value": {"bad_status": bad_status, "empty_output_episodes": empty}, "pass": not bad_status and not empty}}
    ca, ia = cand["assessment"], inc["assessment"]
    return {"decision": "PROMOTED" if all(x["pass"] for x in checks.values()) else "REJECTED", "gate": GATE_HITL, "checks": checks,
            "labeled_sample": {"candidate": c, "incumbent": i},
            "label_free": {"candidate_self_score": ca["mean_self_score"], "incumbent_self_score": ia["mean_self_score"],
                           "candidate_components": ca["component_means"], "incumbent_components": ia["component_means"]},
            "uncertainty": f"labeled sample of {len(cand_fb['episodes'])} episodes; held-out measured separately"}


def version_record(r: dict[str, Any], heldout_ids: list[str] | None = None, labeled_ids: list[str] | None = None) -> dict[str, Any]:
    eps = r["run"]["episodes"].values()
    extra = {}
    if heldout_ids is not None and r.get("gt_measurement") is not None:
        extra = {"headline_gt_heldout": subset_gt(r, heldout_ids), "headline_gt_labeled": subset_gt(r, labeled_ids or []) if labeled_ids else {},
                 "heldout_episodes": len(heldout_ids), "labeled_episodes": len(labeled_ids or [])}
    return {**extra, "version_id": r["version_id"], "episodes": len(r["episodes"]), "headline_gt": headline(r.get("gt_measurement")),
            "label_free_self_score": r["assessment"]["mean_self_score"], "label_free_components": r["assessment"]["component_means"],
            "annotation_cost_usd_total": round(sum(e.get("inference_cost_usd", 0.0) for e in eps), 6),
            "annotation_calls": sum(e["calls_used"] for e in eps), "statuses": r["statuses"]}


def window_cost(store, run_id: str, t0: float, t1: float) -> dict[str, Any]:
    rows = store.query("SELECT role, COUNT(*) n, SUM(cache_hit) ch, COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(prompt_tokens),0) pt, "
                       "COALESCE(SUM(completion_tokens),0) ct FROM model_calls WHERE run_id = ? AND created_at >= ? AND created_at < ? GROUP BY role",
                       (run_id, t0, t1))
    by = {r["role"]: {"calls": r["n"], "cache_hits": r["ch"] or 0, "cost_usd": round(r["c"], 6), "prompt_tokens": r["pt"],
                      "completion_tokens": r["ct"]} for r in rows}
    return {"by_role": by, "total_cost_usd": round(sum(v["cost_usd"] for v in by.values()), 6)}


def findings_view(graph: TaskGraph) -> list[dict[str, Any]]:
    out = []
    for t in graph.tasks.values():
        out.append({"task_id": t.id, "parent": t.parent_task_id, "depth": t.depth, "question": t.question, "status": t.status,
                    "finding": (t.result or {}).get("finding"), "implication": (t.result or {}).get("implication_for_workflow"),
                    "evidence_ids": (t.result or {}).get("evidence_ids") or t.evidence_ids,
                    "dispatch_decision": (t.result or {}).get("dispatch_decision")})
    return out


def cmd_evolve(args: argparse.Namespace) -> int:
    s = get_settings()
    store = get_store()
    client = CrusoeClient(s, store)
    set_name = getattr(args, "set", None) or "droid50"
    prefix = "" if set_name == "droid50" else f"{set_name}-"  # version ids stay unique across batches
    episode_ids = episode_set_ids(set_name, s)
    hitl = bool(getattr(args, "hitl", False))
    calib_ids, heldout_ids = calibration_split(set_name, s) if hitl else ([], None)
    if hitl:
        prefix += "hitl-"
        supervisor_mod.configure_labeled_sample(None)
        investigator_mod.configure_labeled_sample()
    gate = GATE_HITL if hitl else GATE
    run_id = store.start_run("evolve", {"protocol": PROTOCOL, "mode": "hitl" if hitl else "label_free", "episode_set": set_name,
                                        "episodes": len(episode_ids), "labeled_ids": calib_ids, "gate": gate, "label_free_score": SCORE_DEFINITION})
    client.run_id = run_id
    versions = VersionStore(s, store)
    evaluator = ProtectedEvaluator.for_episode_set(episode_ids, s)  # controller-only, for shadow measurement
    ensure_n0(client, versions)
    lin = load_lineage(s, set_name, hitl)
    lin.setdefault("mode", "hitl" if hitl else "label_free")
    if hitl:
        lin.setdefault("labeled_ids", calib_ids)
    incumbent = versions.load(lin["incumbent"])
    dev_ids = val_ids = episode_ids  # v3: every version of this lineage runs on the same episode set
    graph = TaskGraph(store, run_id, max_total=s.max_agent_tasks)
    history = [{k: e.get(k) for k in HISTORY_KEYS} for e in lin["experiments"]]
    report: dict[str, Any] = {"run_id": run_id, "protocol": PROTOCOL, "mode": "hitl" if hitl else "label_free", "labeled_ids": calib_ids,
                              "episode_set": set_name, "gate": gate, "label_free_score": SCORE_DEFINITION,
                              "episode_ids": episode_ids, "iterations": []}
    ctx: ToolContext | None = None
    status = "ok"
    n_done = sum(1 for e in lin["experiments"] if e.get("decision") in ("PROMOTED", "REJECTED"))
    for it in range(max(0, s.max_candidate_experiments - n_done)):
        it_rec: dict[str, Any] = {"iteration": it + 1, "incumbent": incumbent.id}
        t_iter = time.time()
        try:
            inc_dev = run_and_assess(incumbent, "eval", episode_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
            inc_val = inc_dev
            lin["versions"].setdefault(incumbent.id, version_record(inc_dev, heldout_ids, calib_ids))
            inc_fb = labeled_feedback(inc_dev, calib_ids, s) if hitl else None
            it_rec["incumbent_label_free"] = inc_dev["assessment"]["mean_self_score"]
            it_rec["incumbent_headline_gt"] = headline(inc_dev["gt_measurement"])
            cprof = cost_profile(inc_val)
            if ctx is None:
                ctx = ToolContext(client=client, store=store, incumbent=incumbent, dev_ids=dev_ids, run_id=run_id, incumbent_eval=inc_dev)
            else:
                ctx.incumbent, ctx.incumbent_eval = incumbent, inc_dev
            ctx.evidence[inc_dev["eval_id"]] = {"id": inc_dev["eval_id"], "tool": "self_assessment", "args": {"version": incumbent.id},
                                                "result": compact_assessment(inc_dev["assessment"]), "task_id": None}
            store.event(run_id, "evidence", ctx.evidence[inc_dev["eval_id"]])
            root_evidence = [inc_dev["eval_id"]]
            if hitl:
                fb_id = f"LABELED-{incumbent.id}-{it + 1}"
                ctx.evidence[fb_id] = {"id": fb_id, "tool": "human_labeled_sample", "args": {"version": incumbent.id, "episodes": calib_ids},
                                       "result": inc_fb, "task_id": None}
                store.event(run_id, "evidence", ctx.evidence[fb_id])
                supervisor_mod.configure_labeled_sample(inc_fb, fb_id)
                root_evidence.append(fb_id)
                it_rec["incumbent_labeled_sample"] = inc_fb["pooled"]
            if hitl:
                analysis = root_cause_analysis(inc_dev, calib_ids, s)
                it_rec["root_cause_analysis"] = {k: analysis[k] for k in ("root_cause_stats_labeled", "structure_all_episodes")}
                store.event(run_id, "hitl_root_cause_stats", {"iteration": it + 1, **it_rec["root_cause_analysis"]})
                exp_id = f"{prefix}EXP-{lin['next_experiment']:04d}"
                lin["next_experiment"] += 1
                rec = hitl_propose(client, store, run_id, exp_id, incumbent, analysis, history, cprof, calib_ids, s)
            else:
                plan = plan_investigations(client, ctx, incumbent, inc_val["assessment"], history, cprof)
                it_rec["plan"] = plan
                store.event(run_id, "supervisor_plan", {"iteration": it + 1, "failure_summary": plan.get("failure_summary"),
                                                        "investigations": plan.get("investigations")})
                roots = []
                for inv in (plan.get("investigations") or [])[:2]:
                    t, _ = graph.add(str(inv.get("question", "")), None, inv.get("tools_allowed") or [],
                                     inv.get("tool_requests") or [], evidence_ids=root_evidence)
                    roots.append(t)
                dispatch(graph, roots, ctx, workflow_summary(incumbent))
                exp_id = f"{prefix}EXP-{lin['next_experiment']:04d}"
                lin["next_experiment"] += 1
                rec = propose_experiment(client, store, run_id, exp_id, findings_view(graph), incumbent, history, cprof, set(ctx.evidence))
            it_rec["proposal_record"] = rec
            if rec.get("status") != "valid":
                exp = {"id": exp_id, "parent_version": incumbent.id, "decision": "INVALID_PROPOSAL", "errors": rec["attempts"][-1]["errors"]}
                lin["experiments"].append(exp)
                history.append({k: exp.get(k) for k in HISTORY_KEYS})
                store.insert("experiment_proposals", {"id": exp_id, "run_id": run_id, "parent_version": incumbent.id, "status": "INVALID",
                                                      "proposal_json": rec, "decision_json": None, "created_at": time.time(), "updated_at": time.time()})
                it_rec["decision"] = exp
                report["iterations"].append(it_rec)
                save_lineage(s, lin, set_name, hitl)
                continue
            proposal = rec["proposal"]
            proposal["task_ids"] = [t.id for t in graph.tasks.values()]
            patch = WorkflowPatch(**proposal["candidate_patch"])
            store.insert("experiment_proposals", {"id": exp_id, "run_id": run_id, "parent_version": incumbent.id, "status": "PROPOSED",
                                                  "proposal_json": proposal, "decision_json": None, "created_at": time.time(), "updated_at": time.time()})
            store.event(run_id, "experiment_proposed", {"exp_id": exp_id, "hypothesis": proposal["hypothesis"], "topology_change": patch.is_topology_change()})
            wf, prompts = apply_patch(incumbent.workflow, incumbent.prompts, patch)
            cand = versions.create(exp_id, wf, prompts, incumbent.config, parent_id=incumbent.id, status="candidate",
                                   experiment={**proposal, "protocol": PROTOCOL, "task_tree": findings_view(graph)})
            cand_val = run_and_assess(cand, "eval", episode_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
            if hitl:
                cand_fb = labeled_feedback(cand_val, calib_ids, s)
                decision = decide_hitl(cand_val, inc_val, cand_fb, inc_fb)  # labeled sample only
            else:
                decision = decide(cand_val, inc_val)  # label-free only
            decision["candidate_usage"] = usage_for(store, run_id, cand.id)
            decision["topology_change"] = patch.is_topology_change()
            final_name = f"{exp_id}_{decision['decision'].lower()}"
            versions.finalize(exp_id, decision["decision"], {"experiment_id": exp_id, "protocol": PROTOCOL, "validation": decision},
                              new_id_=final_name)
            new_incumbent = incumbent
            if decision["decision"] == "PROMOTED":
                new_vid = f"{prefix}N{lin['next_version']:03d}"
                lin["next_version"] += 1
                new_incumbent = versions.promote_copy(final_name, new_vid)
                decision["new_incumbent"] = new_vid
            else:
                decision["rollback"] = f"incumbent remains {incumbent.id}; candidate preserved as {final_name}"
                incumbent = versions.load(incumbent.id)
            # Shadow GT measurement: recorded after the decision, for the report only.
            gt = shadow_gt(cand_val, inc_val)
            store.update("experiment_proposals", exp_id, {"status": decision["decision"], "decision_json": decision, "updated_at": time.time()})
            store.event(run_id, "experiment_decision", {"exp_id": exp_id, "decision": decision["decision"], "checks": decision["checks"],
                                                        "incumbent_after": new_incumbent.id})
            store.event(run_id, "shadow_gt_measurement", {"exp_id": exp_id, **gt})
            lin["versions"][final_name] = version_record(cand_val, heldout_ids, calib_ids)
            exp = {"id": exp_id, "parent_version": incumbent.id, "version_dir": final_name, "hypothesis": proposal["hypothesis"],
                   "patch_summary": {k: v for k, v in proposal["candidate_patch"].items() if v}, "decision": decision["decision"],
                   "label_free": decision["label_free"], "labeled_sample": decision.get("labeled_sample"), "topology_change": decision["topology_change"],
                   "root_causes": proposal.get("root_causes"), "selected_round": proposal.get("selected_round"), "trials": proposal.get("trials"),
                   "shadow_gt": gt, "shadow_gt_heldout": ({"candidate": subset_gt(cand_val, heldout_ids, s), "incumbent": subset_gt(inc_val, heldout_ids, s)}
                                                          if hitl else None)}
            lin["experiments"].append(exp)
            history.append({k: exp.get(k) for k in HISTORY_KEYS})
            incumbent = new_incumbent
            lin["incumbent"] = incumbent.id
            it_rec["decision"] = decision
            it_rec["shadow_gt"] = gt
            it_rec["recursion_cost"] = window_cost(store, run_id, t_iter, time.time() + 1)
            exp["recursion_cost"] = it_rec["recursion_cost"]
            save_lineage(s, lin, set_name, hitl)
        except (BudgetExceeded, ModelCallError) as e:
            it_rec["stopped"] = f"{type(e).__name__}: {e}"
            store.event(run_id, "evolve_stopped", {"iteration": it + 1, "reason": str(e)[:400]})
            status = "partial"
            report["iterations"].append(it_rec)
            break
        report["iterations"].append(it_rec)
    report["task_tree"] = tree(graph)
    report["dispatch_decisions"] = graph.decisions
    report["max_depth_reached"] = max((t.depth for t in graph.tasks.values() if t.status in ("done", "failed")), default=0)
    report["task_status"] = unfinished_tasks(store, run_id)
    report["evidence"] = ctx.evidence if ctx else {}
    report["incumbent_after"] = lin["incumbent"]
    report["usage"] = {"nemotron": usage_for(store, run_id, role="nemotron"), "vision": usage_for(store, run_id, role="vision")}
    (s.artifacts_dir / f"evolution{set_suffix(set_name, hitl)}.json").write_text(json.dumps(report, indent=1, default=str))
    save_lineage(s, lin, set_name, hitl)
    store.finish_run(run_id, status)
    for it in report["iterations"]:
        d = it.get("decision") or {}
        print(json.dumps({"iteration": it["iteration"], "incumbent": it["incumbent"], "decision": d.get("decision"),
                          "label_free": d.get("label_free"), "shadow_gt": it.get("shadow_gt"),
                          "topology_change": d.get("topology_change"), "stopped": it.get("stopped")}, default=str))
    print(json.dumps({"tasks": len(graph.tasks), "max_depth_reached": report["max_depth_reached"], "incumbent_after": lin["incumbent"],
                      "usage": report["usage"]}, default=str))
    return 0
