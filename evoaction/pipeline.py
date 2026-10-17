"""Run + evaluate helpers shared by the CLI, controller and final evaluation."""

from __future__ import annotations

import json
import time
from typing import Any

from evoaction.clients.crusoe import CrusoeClient
from evoaction.config import Settings, get_settings
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.persistence import Store, get_store, new_id
from evoaction.workflow.baselines import workflow_n0
from evoaction.workflow.engine import Version, VersionStore, run_version, usage_for


def model_config(client: CrusoeClient) -> dict[str, Any]:
    return {"models": {"vision": client.vision_model, "nemotron": client.nemotron_model}, "api_base": client.settings.api_base}


def ensure_n0(client: CrusoeClient, versions: VersionStore) -> Version:
    wf, prompts = workflow_n0()
    return versions.create("N0", wf, prompts, model_config(client), parent_id=None, status="incumbent")


def headline(gt: dict[str, Any] | None) -> dict[str, Any]:
    """Macrodata headline: pooled Segment F1@0.75 with first start / last end snapped to the reference."""
    if not gt:
        return {}
    raw, snap = gt["raw"], gt["outer_snapped"]
    return {"snapped_f1_075": snap["micro_f1"], "raw_f1_075": raw["micro_f1"], "snapped_f1_05": snap["micro_f1_iou05"],
            "raw_f1_05": raw["micro_f1_iou05"], "tp_snapped": snap["tp"], "tp_raw": raw["tp"], "tp05_snapped": snap.get("tp_05"),
            "tp05_raw": raw.get("tp_05"), "n_pred": raw["n_pred"], "n_pred_snapped": snap["n_pred"], "n_ref": raw["n_ref"],
            "invalid_items": raw["invalid_items"], "empty_output_episodes": raw["empty_output_episodes"],
            "mean_boundary_mae_sec": raw["mean_boundary_mae_sec"], "episodes": raw["episodes"],
            "per_episode_snapped_f1": snap["per_episode_f1"]}


def run_and_score(version: Version, split: str, episode_ids: list[str], client: CrusoeClient, evaluator: ProtectedEvaluator,
                  run_id: str, store: Store | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Execute a version, persist privileged per-episode scores, return the split-appropriate agent view."""
    store = store or get_store()
    settings = settings or get_settings()
    t0 = time.time()
    res = run_version(version, episode_ids, client, run_id, store=store, settings=settings)
    preds = {e: r["segments"] for e, r in res["episodes"].items()}
    durs = {e: r["duration_sec"] for e, r in res["episodes"].items()}
    full = evaluator.score_split(preds, durs)
    for e, m in full["per_episode"].items():
        store.insert("eval_by_episode", {"id": new_id("EVAL"), "run_id": run_id, "version_id": version.id, "split": split,
                                         "episode_id": e, "metrics_json": {**m, "status": res["episodes"][e]["status"],
                                                                           "calls_used": res["episodes"][e]["calls_used"]},
                                         "created_at": time.time()})
    if split == "dev":
        agent_view = evaluator.dev_diagnostics(preds, durs)
    elif split == "validation":
        agent_view = evaluator.validation_summary(preds, durs)
    else:
        agent_view = None  # test results never go to agents
    eval_id = new_id("EVAL")
    store.event(run_id, "evaluation", {"eval_id": eval_id, "version_id": version.id, "split": split,
                                       "micro_f1": full["raw"]["micro_f1"], "micro_f1_iou05": full["raw"]["micro_f1_iou05"],
                                       "episodes": episode_ids})
    return {"eval_id": eval_id, "version_id": version.id, "split": split, "episodes": episode_ids, "run": res, "full": full,
            "agent_view": agent_view, "usage": usage_for(store, run_id, version.id), "wall_s": round(time.time() - t0, 1),
            "statuses": {e: r["status"] for e, r in res["episodes"].items()},
            "calls_used": sum(r["calls_used"] for r in res["episodes"].values())}


def run_and_assess(version: Version, split: str, episode_ids: list[str], client: CrusoeClient, run_id: str, *,
                   evaluator: ProtectedEvaluator | None = None, verify: bool = True, store: Store | None = None,
                   settings: Settings | None = None) -> dict[str, Any]:
    """Execute a version and compute its label-free self-assessment (the only signal agents/gate see).

    If an evaluator is given, a GT measurement is also computed and persisted for the report, returned under
    `gt_measurement`. Callers must never pass `gt_measurement` into agent context or the promotion gate.
    """
    from evoaction.selfcheck.label_free import assess_run, sampling_hz_of

    store = store or get_store()
    settings = settings or get_settings()
    t0 = time.time()
    res = run_version(version, episode_ids, client, run_id, store=store, settings=settings)
    assessment = assess_run(client, res, sampling_hz_of(version), version.id, verify=verify)
    eval_id = new_id("SELF")
    store.event(run_id, "self_assessment", {"eval_id": eval_id, "version_id": version.id, "split": split, "episodes": episode_ids,
                                            "mean_self_score": assessment["mean_self_score"],
                                            "per_episode_self_score": assessment["per_episode_self_score"],
                                            "component_means": assessment["component_means"]})
    gt = None
    if evaluator is not None:
        preds = {e: r["segments"] for e, r in res["episodes"].items()}
        durs = {e: r["duration_sec"] for e, r in res["episodes"].items()}
        gt = evaluator.score_split(preds, durs)
        for e, m in gt["per_episode"].items():
            store.insert("eval_by_episode", {"id": new_id("EVAL"), "run_id": run_id, "version_id": version.id, "split": split,
                                             "episode_id": e, "metrics_json": {**m, "status": res["episodes"][e]["status"],
                                                                               "self_score": assessment["per_episode_self_score"].get(e)},
                                             "created_at": time.time()})
        store.event(run_id, "gt_measurement_hidden_from_agents", {"version_id": version.id, "split": split,
                                                                  "micro_f1": gt["raw"]["micro_f1"], "micro_f1_iou05": gt["raw"]["micro_f1_iou05"]})
    return {"eval_id": eval_id, "version_id": version.id, "split": split, "episodes": episode_ids, "run": res,
            "assessment": assessment, "gt_measurement": gt, "usage": usage_for(store, run_id, version.id),
            "wall_s": round(time.time() - t0, 1), "statuses": {e: r["status"] for e, r in res["episodes"].items()},
            "calls_used": sum(r["calls_used"] for r in res["episodes"].values())}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    raw = result["full"]["raw"]
    return {"version": result["version_id"], "split": result["split"], "episodes": len(result["episodes"]),
            "micro_f1@0.75": round(raw["micro_f1"], 3), "micro_f1@0.5": round(raw["micro_f1_iou05"], 3),
            "outer_snapped_f1@0.75": round(result["full"]["outer_snapped"]["micro_f1"], 3),
            "short_recall": None if raw["short_event_recall"] is None else round(raw["short_event_recall"], 3),
            "boundary_mae_s": None if raw["mean_boundary_mae_sec"] is None else round(raw["mean_boundary_mae_sec"], 2),
            "n_pred/n_ref": f"{raw['n_pred']}/{raw['n_ref']}", "invalid": raw["invalid_items"], "empty": raw["empty_output_episodes"],
            "calls": result["calls_used"], "usage": result["usage"], "statuses": result["statuses"]}


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)
