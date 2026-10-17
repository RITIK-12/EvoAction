"""Version N -> N+1 controller (label-free protocol v2).

Agents and the promotion gate see only label-free signals (evoaction.selfcheck).
Ground truth is used solely for an independent "shadow" measurement that is
persisted for the report after each decision and never enters agent context,
agent history, or the gate. Rejected candidates are preserved immutably and
never change the incumbent.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from evoaction.agents.investigator import TaskGraph, dispatch, tree, unfinished_tasks
from evoaction.agents.supervisor import plan_investigations, propose_experiment, workflow_summary
from evoaction.agents.tools import ToolContext
from evoaction.clients.crusoe import BudgetExceeded, CrusoeClient, ModelCallError
from evoaction.config import Settings, get_settings
from evoaction.data.wgo import split_ids
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.evaluation.segments import paired_comparison
from evoaction.persistence import get_store
from evoaction.pipeline import ensure_baselines, run_and_assess
from evoaction.schemas import WorkflowPatch
from evoaction.selfcheck.label_free import SCORE_DEFINITION, compact_assessment
from evoaction.workflow.engine import VersionStore, apply_patch, usage_for

PROTOCOL = "v2-label-free"
GATE = {
    "signal": SCORE_DEFINITION["version"],
    "min_self_score_delta": 0.03,
    "min_nonregressing_fraction": 0.5,
    "max_annotation_cost_ratio": 1.5,
    "require_valid_outputs": True,
    "note": "label-free gate predeclared before the v2 run; ground truth is never consulted for promotion",
}
HISTORY_KEYS = ("id", "parent_version", "hypothesis", "patch_summary", "decision", "label_free")


def lineage_path(s: Settings):
    return s.artifacts_dir / "lineage.json"


def load_lineage(s: Settings) -> dict[str, Any]:
    p = lineage_path(s)
    if p.exists():
        return json.loads(p.read_text())
    return {"protocol": PROTOCOL, "incumbent": "N000", "next_experiment": 1, "next_version": 1, "experiments": []}


def save_lineage(s: Settings, lin: dict[str, Any]) -> None:
    lineage_path(s).write_text(json.dumps(lin, indent=1, default=str))


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
                     "but tended toward evenly spaced segment boundaries")}


def decide(cand: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
    """Label-free promotion gate. Uses only self-assessments, run statuses and cost."""
    ca, ia = cand["assessment"], inc["assessment"]
    paired = paired_comparison(ca["per_episode_self_score"], ia["per_episode_self_score"])
    delta = ca["mean_self_score"] - ia["mean_self_score"]
    c_cost = cost_profile(cand)["mean_annotation_cost_usd_per_episode"]
    i_cost = cost_profile(inc)["mean_annotation_cost_usd_per_episode"]
    ratio = (c_cost / i_cost) if i_cost > 0 else float("inf")
    bad_status = {e: s for e, s in cand["statuses"].items() if s not in ("ok", "partial")}
    empty = [e for e, r in cand["run"]["episodes"].items() if not r["segments"]]
    checks = {
        "self_score_delta": {"value": round(delta, 4), "threshold": GATE["min_self_score_delta"], "pass": delta >= GATE["min_self_score_delta"]},
        "non_regressing_fraction": {"value": paired["non_regressing_fraction"], "threshold": GATE["min_nonregressing_fraction"],
                                    "pass": paired["non_regressing_fraction"] >= GATE["min_nonregressing_fraction"]},
        "valid_outputs": {"value": {"bad_status": bad_status, "empty_output_episodes": empty}, "pass": not bad_status and not empty},
        "annotation_cost_ratio": {"value": round(ratio, 3) if ratio != float("inf") else None, "threshold": GATE["max_annotation_cost_ratio"],
                                  "candidate_usd_per_episode": c_cost, "incumbent_usd_per_episode": i_cost,
                                  "pass": ratio <= GATE["max_annotation_cost_ratio"]},
    }
    promote = all(c["pass"] for c in checks.values())
    return {"decision": "PROMOTED" if promote else "REJECTED", "gate": GATE, "checks": checks, "paired_self_score": paired,
            "label_free": {"candidate_self_score": ca["mean_self_score"], "incumbent_self_score": ia["mean_self_score"],
                           "candidate_components": ca["component_means"], "incumbent_components": ia["component_means"]},
            "uncertainty": f"n={len(paired['episodes'])} validation episodes; label-free proxy, not ground truth"}


def shadow_gt(cand: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
    """Independent GT measurement for the report only (computed after the decision; never shown to agents)."""
    def pick(r):
        g = r.get("gt_measurement") or {}
        raw = g.get("raw", {})
        return {k: raw.get(k) for k in ("micro_f1", "micro_f1_iou05", "short_event_recall", "mean_boundary_mae_sec")} | {
            "outer_snapped_micro_f1": (g.get("outer_snapped") or {}).get("micro_f1"), "per_episode_f1": raw.get("per_episode_f1")}
    return {"candidate": pick(cand), "incumbent": pick(inc)}


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
    pilot = not getattr(args, "full", False)
    run_id = store.start_run("evolve", {"pilot": pilot, "protocol": PROTOCOL, "gate": GATE, "label_free_score": SCORE_DEFINITION})
    client.run_id = run_id
    versions = VersionStore(s, store)
    evaluator = ProtectedEvaluator(s)  # controller-only, for shadow measurement
    ensure_baselines(client, versions)
    lin = load_lineage(s)
    incumbent = versions.load(lin["incumbent"])
    dev_ids = split_ids("dev", pilot)
    val_ids = split_ids("validation", pilot)
    graph = TaskGraph(store, run_id, max_total=s.max_agent_tasks)
    history = [{k: e.get(k) for k in HISTORY_KEYS} for e in lin["experiments"]]
    report: dict[str, Any] = {"run_id": run_id, "protocol": PROTOCOL, "pilot": pilot, "gate": GATE, "label_free_score": SCORE_DEFINITION,
                              "dev_ids": dev_ids, "val_ids": val_ids, "iterations": []}
    ctx: ToolContext | None = None
    status = "ok"
    n_done = sum(1 for e in lin["experiments"] if e.get("decision") in ("PROMOTED", "REJECTED"))
    for it in range(max(0, s.max_candidate_experiments - n_done)):
        it_rec: dict[str, Any] = {"iteration": it + 1, "incumbent": incumbent.id}
        try:
            inc_dev = run_and_assess(incumbent, "dev", dev_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
            inc_val = run_and_assess(incumbent, "validation", val_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
            it_rec["incumbent_label_free"] = {"dev": inc_dev["assessment"]["mean_self_score"], "val": inc_val["assessment"]["mean_self_score"]}
            it_rec["incumbent_shadow_gt"] = {"dev": (inc_dev["gt_measurement"] or {}).get("raw", {}).get("micro_f1"),
                                             "val": (inc_val["gt_measurement"] or {}).get("raw", {}).get("micro_f1")}
            cprof = cost_profile(inc_val)
            if ctx is None:
                ctx = ToolContext(client=client, store=store, incumbent=incumbent, dev_ids=dev_ids, run_id=run_id, incumbent_eval=inc_dev)
            else:
                ctx.incumbent, ctx.incumbent_eval = incumbent, inc_dev
            ctx.evidence[inc_dev["eval_id"]] = {"id": inc_dev["eval_id"], "tool": "self_assessment", "args": {"version": incumbent.id},
                                                "result": compact_assessment(inc_dev["assessment"]), "task_id": None}
            store.event(run_id, "evidence", ctx.evidence[inc_dev["eval_id"]])
            plan = plan_investigations(client, ctx, incumbent, inc_val["assessment"], history, cprof)
            it_rec["plan"] = plan
            store.event(run_id, "supervisor_plan", {"iteration": it + 1, "failure_summary": plan.get("failure_summary"),
                                                    "investigations": plan.get("investigations")})
            roots = []
            for inv in (plan.get("investigations") or [])[:2]:
                t, _ = graph.add(str(inv.get("question", "")), None, inv.get("tools_allowed") or [],
                                 inv.get("tool_requests") or [], evidence_ids=[inc_dev["eval_id"]])
                roots.append(t)
            dispatch(graph, roots, ctx, workflow_summary(incumbent))
            exp_id = f"EXP-{lin['next_experiment']:04d}"
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
                save_lineage(s, lin)
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
            cand_val = run_and_assess(cand, "validation", val_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
            decision = decide(cand_val, inc_val)  # label-free only
            decision["candidate_usage"] = usage_for(store, run_id, cand.id)
            decision["topology_change"] = patch.is_topology_change()
            final_name = f"{exp_id}_{decision['decision'].lower()}"
            versions.finalize(exp_id, decision["decision"], {"experiment_id": exp_id, "protocol": PROTOCOL, "validation": decision},
                              new_id_=final_name)
            new_incumbent = incumbent
            if decision["decision"] == "PROMOTED":
                new_vid = f"N{lin['next_version']:03d}"
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
            exp = {"id": exp_id, "parent_version": incumbent.id, "version_dir": final_name, "hypothesis": proposal["hypothesis"],
                   "patch_summary": {k: v for k, v in proposal["candidate_patch"].items() if v}, "decision": decision["decision"],
                   "label_free": decision["label_free"], "topology_change": decision["topology_change"], "shadow_gt": gt}
            lin["experiments"].append(exp)
            history.append({k: exp.get(k) for k in HISTORY_KEYS})
            incumbent = new_incumbent
            lin["incumbent"] = incumbent.id
            it_rec["decision"] = decision
            it_rec["shadow_gt"] = gt
            save_lineage(s, lin)
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
    (s.artifacts_dir / "evolution.json").write_text(json.dumps(report, indent=1, default=str))
    save_lineage(s, lin)
    store.finish_run(run_id, status)
    for it in report["iterations"]:
        d = it.get("decision") or {}
        print(json.dumps({"iteration": it["iteration"], "incumbent": it["incumbent"], "decision": d.get("decision"),
                          "label_free": d.get("label_free"), "shadow_gt": it.get("shadow_gt"),
                          "topology_change": d.get("topology_change"), "stopped": it.get("stopped")}, default=str))
    print(json.dumps({"tasks": len(graph.tasks), "max_depth_reached": report["max_depth_reached"], "incumbent_after": lin["incumbent"],
                      "usage": report["usage"]}, default=str))
    return 0
