"""Version N -> N+1 controller: investigate, propose, validate, promote or reject, roll back.

The controller (not any agent) applies the predeclared promotion gate. Rejected
candidates are preserved immutably and never change the incumbent.
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
from evoaction.persistence import Store, get_store
from evoaction.pipeline import ensure_baselines, run_and_score
from evoaction.workflow.engine import Version, VersionStore, apply_patch, usage_for
from evoaction.schemas import WorkflowPatch

GATE = {
    "min_val_micro_f1_delta": 0.02,
    "min_nonregressing_fraction": 0.5,
    "max_annotation_cost_ratio": 1.5,
    "require_valid_outputs": True,
    "note": "operational heuristics on a tiny validation set, not statistical significance",
}


def lineage_path(s: Settings):
    return s.artifacts_dir / "lineage.json"


def load_lineage(s: Settings) -> dict[str, Any]:
    p = lineage_path(s)
    if p.exists():
        return json.loads(p.read_text())
    return {"incumbent": "N000", "next_experiment": 1, "next_version": 1, "experiments": []}


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
    cv, iv = cand["agent_view"], inc["agent_view"]
    paired = paired_comparison(cv["per_episode_f1"], iv["per_episode_f1"])
    delta = cv["micro_f1"] - iv["micro_f1"]
    c_cost = cost_profile(cand)["mean_annotation_cost_usd_per_episode"]
    i_cost = cost_profile(inc)["mean_annotation_cost_usd_per_episode"]
    ratio = (c_cost / i_cost) if i_cost > 0 else float("inf")
    bad_status = {e: s for e, s in cand["statuses"].items() if s not in ("ok", "partial")}
    valid = not bad_status and cv["invalid_items"] <= iv["invalid_items"] and cv["empty_output_episodes"] <= iv["empty_output_episodes"]
    checks = {
        "val_micro_f1_delta": {"value": round(delta, 4), "threshold": GATE["min_val_micro_f1_delta"], "pass": delta >= GATE["min_val_micro_f1_delta"]},
        "non_regressing_fraction": {"value": paired["non_regressing_fraction"], "threshold": GATE["min_nonregressing_fraction"],
                                    "pass": paired["non_regressing_fraction"] >= GATE["min_nonregressing_fraction"]},
        "valid_outputs": {"value": {"candidate_invalid": cv["invalid_items"], "candidate_empty": cv["empty_output_episodes"],
                                    "bad_status": bad_status}, "pass": valid},
        "annotation_cost_ratio": {"value": round(ratio, 3) if ratio != float("inf") else None, "threshold": GATE["max_annotation_cost_ratio"],
                                  "candidate_usd_per_episode": c_cost, "incumbent_usd_per_episode": i_cost,
                                  "pass": ratio <= GATE["max_annotation_cost_ratio"]},
    }
    promote = all(c["pass"] for c in checks.values())
    marginal = promote and delta < 2 * GATE["min_val_micro_f1_delta"]
    return {"decision": "PROMOTED" if promote else "REJECTED", "gate": GATE, "checks": checks, "paired": paired,
            "candidate_val": {k: cv.get(k) for k in ("micro_f1", "micro_f1_iou05", "short_event_recall", "mean_boundary_mae_sec", "outer_snapped_micro_f1")},
            "incumbent_val": {k: iv.get(k) for k in ("micro_f1", "micro_f1_iou05", "short_event_recall", "mean_boundary_mae_sec", "outer_snapped_micro_f1")},
            "uncertainty": f"n={len(paired['episodes'])} validation episodes; differences of a few segments swing F1 by >0.05"
                           + ("; marginal improvement" if marginal else "")}


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
    run_id = store.start_run("evolve", {"pilot": pilot})
    client.run_id = run_id
    versions = VersionStore(s, store)
    evaluator = ProtectedEvaluator(s)
    ensure_baselines(client, versions)
    lin = load_lineage(s)
    incumbent = versions.load(lin["incumbent"])
    dev_ids = split_ids("dev", pilot)
    val_ids = split_ids("validation", pilot)
    graph = TaskGraph(store, run_id, max_total=s.max_agent_tasks)
    history = [{k: e.get(k) for k in ("id", "parent_version", "hypothesis", "patch_summary", "decision", "candidate_val", "incumbent_val")}
               for e in lin["experiments"]]
    report: dict[str, Any] = {"run_id": run_id, "pilot": pilot, "gate": GATE, "dev_ids": dev_ids, "val_ids": val_ids, "iterations": []}
    ctx: ToolContext | None = None
    status = "ok"
    n_done = sum(1 for e in lin["experiments"] if e.get("decision") in ("PROMOTED", "REJECTED"))
    for it in range(max(0, s.max_candidate_experiments - n_done)):
        it_rec: dict[str, Any] = {"iteration": it + 1, "incumbent": incumbent.id}
        try:
            inc_dev = run_and_score(incumbent, "dev", dev_ids, client, evaluator, run_id, store, s)
            inc_val = run_and_score(incumbent, "validation", val_ids, client, evaluator, run_id, store, s)
            cprof = cost_profile(inc_val)
            if ctx is None:
                ctx = ToolContext(client=client, store=store, evaluator=evaluator, incumbent=incumbent, dev_ids=dev_ids,
                                  run_id=run_id, incumbent_dev=inc_dev)
            else:
                ctx.incumbent, ctx.incumbent_dev = incumbent, inc_dev
            ctx.record("dev_failure_profile", {"version": incumbent.id, "eval_id": inc_dev["eval_id"]}, inc_dev["agent_view"], None)
            ctx.evidence[inc_dev["eval_id"]] = {"id": inc_dev["eval_id"], "tool": "dev_failure_profile", "args": {"version": incumbent.id},
                                                "result": inc_dev["agent_view"], "task_id": None}
            plan = plan_investigations(client, ctx, incumbent, inc_val["agent_view"], history, cprof)
            it_rec["plan"] = plan
            store.event(run_id, "supervisor_plan", {"iteration": it + 1, "failure_summary": plan.get("failure_summary"),
                                                    "investigations": plan.get("investigations")})
            roots = []
            for inv in (plan.get("investigations") or [])[:2]:
                t, verdict = graph.add(str(inv.get("question", "")), None, inv.get("tools_allowed") or [],
                                       inv.get("tool_requests") or [], evidence_ids=[inc_dev["eval_id"]])
                roots.append(t)
            dispatch(graph, roots, ctx, workflow_summary(incumbent))
            exp_id = f"EXP-{lin['next_experiment']:04d}"
            lin["next_experiment"] += 1
            rec = propose_experiment(client, store, run_id, exp_id, findings_view(graph), incumbent, history, cprof,
                                     set(ctx.evidence))
            it_rec["proposal_record"] = rec
            if rec.get("status") != "valid":
                exp = {"id": exp_id, "parent_version": incumbent.id, "decision": "INVALID_PROPOSAL", "errors": rec["attempts"][-1]["errors"]}
                lin["experiments"].append(exp)
                store.insert("experiment_proposals", {"id": exp_id, "run_id": run_id, "parent_version": incumbent.id, "status": "INVALID",
                                                      "proposal_json": rec, "decision_json": None, "created_at": time.time(), "updated_at": time.time()})
                it_rec["decision"] = exp
                report["iterations"].append(it_rec)
                save_lineage(s, lin)
                continue
            proposal = rec["proposal"]
            proposal["task_ids"] = [t.id for t in graph.tasks.values()]
            store.insert("experiment_proposals", {"id": exp_id, "run_id": run_id, "parent_version": incumbent.id, "status": "PROPOSED",
                                                  "proposal_json": proposal, "decision_json": None, "created_at": time.time(), "updated_at": time.time()})
            store.event(run_id, "experiment_proposed", {"exp_id": exp_id, "hypothesis": proposal["hypothesis"],
                                                        "topology_change": WorkflowPatch(**proposal["candidate_patch"]).is_topology_change()})
            wf, prompts = apply_patch(incumbent.workflow, incumbent.prompts, WorkflowPatch(**proposal["candidate_patch"]))
            cand = versions.create(exp_id, wf, prompts, incumbent.config, parent_id=incumbent.id, status="candidate",
                                   experiment={**proposal, "task_tree": findings_view(graph)})
            cand_val = run_and_score(cand, "validation", val_ids, client, evaluator, run_id, store, s)
            decision = decide(cand_val, inc_val)
            decision["candidate_usage"] = usage_for(store, run_id, cand.id)
            decision["topology_change"] = WorkflowPatch(**proposal["candidate_patch"]).is_topology_change()
            final_name = f"{exp_id}_{decision['decision'].lower()}"
            versions.finalize(exp_id, decision["decision"], {"experiment_id": exp_id, "validation": decision}, new_id_=final_name)
            new_incumbent = incumbent
            if decision["decision"] == "PROMOTED":
                new_vid = f"N{lin['next_version']:03d}"
                lin["next_version"] += 1
                new_incumbent = versions.promote_copy(final_name, new_vid)
                decision["new_incumbent"] = new_vid
            else:
                decision["rollback"] = f"incumbent remains {incumbent.id}; candidate preserved as {final_name}"
                incumbent = versions.load(incumbent.id)  # reload last approved incumbent from immutable artifacts
            store.update("experiment_proposals", exp_id, {"status": decision["decision"], "decision_json": decision, "updated_at": time.time()})
            store.event(run_id, "experiment_decision", {"exp_id": exp_id, "decision": decision["decision"], "checks": decision["checks"],
                                                        "incumbent_after": new_incumbent.id})
            patch_summary = {k: v for k, v in proposal["candidate_patch"].items() if v}
            exp = {"id": exp_id, "parent_version": incumbent.id, "version_dir": final_name, "hypothesis": proposal["hypothesis"],
                   "patch_summary": patch_summary, "decision": decision["decision"], "candidate_val": decision["candidate_val"],
                   "incumbent_val": decision["incumbent_val"], "topology_change": decision["topology_change"]}
            lin["experiments"].append(exp)
            history.append({k: exp.get(k) for k in ("id", "parent_version", "hypothesis", "patch_summary", "decision", "candidate_val", "incumbent_val")})
            incumbent = new_incumbent
            lin["incumbent"] = incumbent.id
            it_rec["decision"] = decision
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
                          "candidate_val": d.get("candidate_val"), "incumbent_val": d.get("incumbent_val"),
                          "topology_change": d.get("topology_change"), "stopped": it.get("stopped")}, default=str))
    print(json.dumps({"tasks": len(graph.tasks), "max_depth_reached": report["max_depth_reached"], "incumbent_after": lin["incumbent"],
                      "usage": report["usage"]}, default=str))
    return 0
