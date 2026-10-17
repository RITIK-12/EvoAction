"""Gate 6: freeze A/B/C, then run the sealed held-out evaluation exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

from evoaction.clients.crusoe import CrusoeClient
from evoaction.config import get_settings
from evoaction.data.wgo import load_splits, split_ids
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.evaluation.segments import PROTOCOL_VERSION, paired_comparison
from evoaction.experiments.evolution import load_lineage
from evoaction.persistence import get_store
from evoaction.pipeline import ensure_baselines, run_and_assess
from evoaction.workflow.engine import VersionStore, usage_for


def _usage_by_kind(store, kinds: tuple[str, ...]) -> dict[str, Any]:
    marks = ",".join("?" for _ in kinds)
    rows = store.query(
        f"SELECT m.role, m.model, COUNT(*) calls, SUM(m.cache_hit) cache_hits, COALESCE(SUM(m.prompt_tokens),0) pt, "
        f"COALESCE(SUM(m.completion_tokens),0) ct, COALESCE(SUM(m.cost_usd),0) cost, COALESCE(SUM(m.latency_s),0) lat "
        f"FROM model_calls m JOIN runs r ON m.run_id = r.id WHERE r.kind IN ({marks}) GROUP BY m.role, m.model", kinds)
    return {f"{r['role']}:{r['model']}": {"calls": r["calls"], "cache_hits": r["cache_hits"] or 0, "prompt_tokens": r["pt"],
                                          "completion_tokens": r["ct"], "cost_usd": round(r["cost"], 6), "model_latency_s": round(r["lat"], 1)}
            for r in rows}


def cmd_final(args: argparse.Namespace) -> int:
    s = get_settings()
    out_path = s.artifacts_dir / "final_test.json"
    if out_path.exists():
        print(f"{out_path} exists: the sealed test evaluation runs once. Delete it only to discard that result deliberately.")
        return 0
    store = get_store()
    client = CrusoeClient(s, store)
    pilot = not getattr(args, "full", False)
    run_id = store.start_run("final_test", {"pilot": pilot})
    client.run_id = run_id
    client.budget.phase = "test"
    versions = VersionStore(s, store)
    ensure_baselines(client, versions)
    lin = load_lineage(s)
    systems = {"A": versions.load("N000"), "B": versions.load("strong_fixed"), "C": versions.load(lin["incumbent"])}
    c_label = "evolved incumbent" if lin["incumbent"] != "N000" else "unchanged incumbent (no candidate promoted)"
    manifest: dict[str, Any] = {"protocol": PROTOCOL_VERSION, "frozen_at": time.time(), "pilot": pilot,
                                "splits_sha256": load_splits(s)["splits_sha256"], "C_label": c_label}
    for k, v in systems.items():
        files = {name: hashlib.sha256((v.path / name).read_bytes()).hexdigest() for name in sorted(v.manifest["files"])}
        manifest[k] = {"version_id": v.id, "workflow_sha256": v.workflow_sha256, "files_sha256": files,
                       "models": v.config.get("models"), "nodes": [n.operator for n in v.workflow.nodes]}
    full_test, pilot_test = split_ids("test", False), split_ids("test", True)
    test_set = getattr(args, "test_set", "fresh")
    test_ids = {"fresh": [e for e in full_test if e not in pilot_test], "pilot": pilot_test, "full": full_test}[test_set]
    continuity_ids = pilot_test if test_set == "fresh" else []
    manifest["test_set"] = {"name": test_set, "episode_ids": test_ids, "continuity_episode_ids": continuity_ids,
                            "note": "fresh = full-split test episodes never evaluated before (v1 scored the pilot test episodes)"}
    if getattr(args, "dry_run", False):
        from evoaction.data.wgo import load_episode

        need = {k: (v.workflow.max_calls + 1) * len(test_ids + continuity_ids) for k, v in systems.items() if not (k == "C" and v.id == systems["A"].id)}
        check = {"systems": {k: (manifest[k]["version_id"], manifest[k]["workflow_sha256"][:12], manifest[k]["nodes"]) for k in "ABC"},
                 "C_label": c_label, "test_ids": test_ids, "test_media_ingested": all(load_episode(e)["video_path"] for e in test_ids),
                 "max_vision_calls_needed": sum(need.values()), "vision_limit_in_test_phase": client.budget.limit_for("vision"),
                 "test_sealed_before_unseal": not ProtectedEvaluator(s)._test_unsealed}
        store.finish_run(run_id, "dry_run")
        print(json.dumps(check, indent=1))
        return 0 if check["max_vision_calls_needed"] <= check["vision_limit_in_test_phase"] else 1
    (s.artifacts_dir / "freeze.json").write_text(json.dumps(manifest, indent=1))
    store.event(run_id, "systems_frozen", {k: manifest[k]["version_id"] for k in "ABC"})

    evaluator = ProtectedEvaluator(s)
    evaluator.unseal_test(manifest)
    results: dict[str, Any] = {}
    preds: dict[str, dict[str, list]] = {}
    durations: dict[str, float] = {}
    for k, v in systems.items():
        if k == "C" and v.id == systems["A"].id:
            results[k] = results["A"]
        else:
            results[k] = run_and_assess(v, "test", test_ids, client, run_id, evaluator=evaluator, store=store, settings=s)
        preds[k] = {e: r["segments"] for e, r in results[k]["run"]["episodes"].items()}
        durations.update({e: r["duration_sec"] for e, r in results[k]["run"]["episodes"].items()})
    rep = evaluator.final_test_report(preds, durations)
    continuity: dict[str, Any] = {}
    if continuity_ids:
        for k, v in systems.items():
            if k == "C" and v.id == systems["A"].id:
                continuity[k] = continuity["A"]
                continue
            r = run_and_assess(v, "test", continuity_ids, client, run_id, evaluator=evaluator, store=store, settings=s, verify=False)
            raw = r["gt_measurement"]["raw"]
            continuity[k] = {"version_id": v.id, "micro_f1": raw["micro_f1"], "micro_f1_iou05": raw["micro_f1_iou05"],
                             "per_episode_f1": raw["per_episode_f1"]}
    summary: dict[str, Any] = {}
    for k in "ABC":
        raw = rep["systems"][k]["raw"]
        eps = results[k]["run"]["episodes"].values()
        summary[k] = {
            "version_id": systems[k].id, "micro_f1": raw["micro_f1"], "micro_f1_iou05": raw["micro_f1_iou05"],
            "outer_snapped_micro_f1": rep["systems"][k]["outer_snapped"]["micro_f1"], "short_event_recall": raw["short_event_recall"],
            "invalid_items": raw["invalid_items"], "empty_output_episodes": raw["empty_output_episodes"],
            "mean_boundary_mae_sec": raw["mean_boundary_mae_sec"], "per_episode_f1": raw["per_episode_f1"],
            "vision_calls": sum(e["calls_used"] for e in eps),
            "annotation_cost_usd": round(sum(e.get("inference_cost_usd", 0.0) for e in eps), 6),
            "annotation_tokens": {"prompt": sum(e.get("inference_tokens", {}).get("prompt", 0) for e in eps),
                                  "completion": sum(e.get("inference_tokens", {}).get("completion", 0) for e in eps)},
            "wall_s": results[k]["wall_s"], "statuses": results[k]["statuses"],
            "label_free_self_score": results[k]["assessment"]["mean_self_score"],
            "label_free_per_episode": results[k]["assessment"]["per_episode_self_score"],
            "label_free_components": results[k]["assessment"]["component_means"],
        }
    out = {"run_id": run_id, "protocol": PROTOCOL_VERSION, "evolution_protocol": "v2-label-free", "test_set": test_set,
           "continuity_episode_ids": continuity_ids, "continuity": continuity, "pilot": pilot, "test_episode_ids": test_ids, "C_label": c_label,
           "freeze_manifest": manifest, "summary": summary,
           "paired": {"C_vs_A": paired_comparison(summary["C"]["per_episode_f1"], summary["A"]["per_episode_f1"]),
                      "C_vs_B": paired_comparison(summary["C"]["per_episode_f1"], summary["B"]["per_episode_f1"]),
                      "B_vs_A": paired_comparison(summary["B"]["per_episode_f1"], summary["A"]["per_episode_f1"])},
           "per_episode": {k: rep["systems"][k]["per_episode"] for k in "ABC"},
           "timelines": rep["timelines"],
           "optimization_overhead": {"evolve_runs": _usage_by_kind(store, ("evolve",)),
                                     "baseline_runs": _usage_by_kind(store, ("baselines",)),
                                     "final_test_run": usage_for(store, run_id)},
           "limitations": [
               f"{len(test_ids)} held-out episodes: a feasibility result, not statistically established generalization.",
               "Temporal Segment F1 evaluates boundaries only; it does not assess the semantic correctness of subtask labels.",
               "The outer-boundary-snapped score is an evaluator-only variant; WGO-Bench's official scorer was not available to verify.",
               "Costs are provider-reported tokens x listed per-token prices from /v1/models; billing was not reconciled.",
           ]}
    out_path.write_text(json.dumps(out, indent=1, default=str))
    store.finish_run(run_id, "ok")
    for k in "ABC":
        sm = summary[k]
        print(json.dumps({"system": k, "version": sm["version_id"], "micro_f1@0.75": round(sm["micro_f1"], 3),
                          "micro_f1@0.5": round(sm["micro_f1_iou05"], 3), "short_recall": sm["short_event_recall"],
                          "calls": sm["vision_calls"], "cost_usd": sm["annotation_cost_usd"], "statuses": sm["statuses"]}, default=str))
    return 0
