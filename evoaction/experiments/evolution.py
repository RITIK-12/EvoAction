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
from evoaction.data.wgo import episode_set_ids
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.evaluation.segments import paired_comparison
from evoaction.persistence import get_store
from evoaction.pipeline import ensure_n0, headline, run_and_assess
from evoaction.schemas import WorkflowPatch
from evoaction.selfcheck.label_free import SCORE_DEFINITION, compact_assessment
from evoaction.workflow.engine import VersionStore, apply_patch, usage_for

PROTOCOL = "v3-macrodata"
GATE = {
    "signal": SCORE_DEFINITION["version"],
    "min_self_score_delta": 0.03,
    "min_nonregressing_fraction": 0.5,
    "require_valid_outputs": True,
    "cost_policy": "no per-episode cost cap; total cost is recorded per recursion (iteration) and per version over the same 50 episodes",
    "note": "label-free gate predeclared before the v3 run; ground truth is never consulted for promotion",
}
HISTORY_KEYS = ("id", "parent_version", "hypothesis", "patch_summary", "decision", "label_free")


def set_suffix(set_name: str) -> str:
    return "" if set_name == "droid50" else f"_{set_name}"


def lineage_path(s: Settings, set_name: str = "droid50"):
    return s.artifacts_dir / f"lineage{set_suffix(set_name)}.json"


def load_lineage(s: Settings, set_name: str = "droid50") -> dict[str, Any]:
    p = lineage_path(s, set_name)
    if p.exists():
        return json.loads(p.read_text())
    return {"protocol": PROTOCOL, "episode_set": set_name, "incumbent": "N0", "next_experiment": 1, "next_version": 1,
            "experiments": [], "versions": {}}


def save_lineage(s: Settings, lin: dict[str, Any], set_name: str = "droid50") -> None:
    lineage_path(s, set_name).write_text(json.dumps(lin, indent=1, default=str))


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
            "uncertainty": f"n={len(paired['episodes'])} validation episodes; label-free proxy, not ground truth"}


def shadow_gt(cand: dict[str, Any], inc: dict[str, Any]) -> dict[str, Any]:
    """Independent GT measurement for the report only (computed after the decision; never shown to agents)."""
    return {"candidate": headline(cand.get("gt_measurement")), "incumbent": headline(inc.get("gt_measurement"))}


def version_record(r: dict[str, Any]) -> dict[str, Any]:
    eps = r["run"]["episodes"].values()
    return {"version_id": r["version_id"], "episodes": len(r["episodes"]), "headline_gt": headline(r.get("gt_measurement")),
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
    run_id = store.start_run("evolve", {"protocol": PROTOCOL, "episode_set": set_name, "episodes": len(episode_ids), "gate": GATE, "label_free_score": SCORE_DEFINITION})
    client.run_id = run_id
    versions = VersionStore(s, store)
    evaluator = ProtectedEvaluator.for_episode_set(episode_ids, s)  # controller-only, for shadow measurement
    ensure_n0(client, versions)
    lin = load_lineage(s, set_name)
    incumbent = versions.load(lin["incumbent"])
    dev_ids = val_ids = episode_ids  # v3: every version of this lineage runs on the same episode set
    graph = TaskGraph(store, run_id, max_total=s.max_agent_tasks)
    history = [{k: e.get(k) for k in HISTORY_KEYS} for e in lin["experiments"]]
    report: dict[str, Any] = {"run_id": run_id, "protocol": PROTOCOL, "episode_set": set_name, "gate": GATE, "label_free_score": SCORE_DEFINITION,
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
            lin["versions"].setdefault(incumbent.id, version_record(inc_dev))
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
                save_lineage(s, lin, set_name)
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
            lin["versions"][final_name] = version_record(cand_val)
            exp = {"id": exp_id, "parent_version": incumbent.id, "version_dir": final_name, "hypothesis": proposal["hypothesis"],
                   "patch_summary": {k: v for k, v in proposal["candidate_patch"].items() if v}, "decision": decision["decision"],
                   "label_free": decision["label_free"], "topology_change": decision["topology_change"], "shadow_gt": gt}
            lin["experiments"].append(exp)
            history.append({k: exp.get(k) for k in HISTORY_KEYS})
            incumbent = new_incumbent
            lin["incumbent"] = incumbent.id
            it_rec["decision"] = decision
            it_rec["shadow_gt"] = gt
            it_rec["recursion_cost"] = window_cost(store, run_id, t_iter, time.time() + 1)
            exp["recursion_cost"] = it_rec["recursion_cost"]
            save_lineage(s, lin, set_name)
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
    (s.artifacts_dir / f"evolution{set_suffix(set_name)}.json").write_text(json.dumps(report, indent=1, default=str))
    save_lineage(s, lin, set_name)
    store.finish_run(run_id, status)
    for it in report["iterations"]:
        d = it.get("decision") or {}
        print(json.dumps({"iteration": it["iteration"], "incumbent": it["incumbent"], "decision": d.get("decision"),
                          "label_free": d.get("label_free"), "shadow_gt": it.get("shadow_gt"),
                          "topology_change": d.get("topology_change"), "stopped": it.get("stopped")}, default=str))
    print(json.dumps({"tasks": len(graph.tasks), "max_depth_reached": report["max_depth_reached"], "incumbent_after": lin["incumbent"],
                      "usage": report["usage"]}, default=str))
    return 0
