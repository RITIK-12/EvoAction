"""Allowlisted, label-free investigation tools.

Agents never receive anything derived from reference labels. Tools expose only the
incumbent's own predictions, video frames, robot telemetry, sampling metadata and
the predeclared label-free self-assessment. Every result is persisted as an
evidence record with an ID that proposals must cite. Tools are restricted to the
protocol's episode set.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from evoaction.clients.crusoe import ROLE_VISION, BudgetExceeded, CrusoeClient, ModelCallError, image_part
from evoaction.data.wgo import load_episode
from evoaction.media.contact_sheets import build_sheets, uniform_timestamps
from evoaction.persistence import Store, new_id
from evoaction.schemas import HintParams, WorkflowPatch
from evoaction.selfcheck.label_free import compact_assessment
from evoaction.workflow.engine import Version, WorkflowValidationError, apply_patch
from evoaction.workflow.operators import detect_hints

TOOL_DOCS = {
    "self_assessment": "self_assessment() -> the incumbent's label-free self-assessment on all episodes in the set (scores, plus details for the lowest-scoring ones): telemetry agreement, visual verification of each boundary, segment-duration variability, temporal resolution, predicted boundaries and gripper events. No reference labels exist for you.",
    "episode_predictions": "episode_predictions(episode_id) -> the incumbent's own predicted segments for an episode in the set plus its sampling metadata.",
    "telemetry_hints": "telemetry_hints(episode_id, signals=['gripper'|'eef_pause']) -> deterministic robot-state change points for an episode in the set, each with distance to the nearest predicted boundary.",
    "sampling_coverage": "sampling_coverage(episode_id, sampling_hz) -> frame count/spacing for a uniform sampling rate and how many predicted segments are shorter than two sampled frames.",
    "inspect_frames": "inspect_frames(episode_id, center_sec, window_sec<=4, local_hz<=5, question) -> a Crusoe vision model's structured description of a close-up frame strip of an episode in the set. Costs one vision call.",
}


@dataclass
class ToolContext:
    client: CrusoeClient
    store: Store
    incumbent: Version
    dev_ids: list[str]
    run_id: str
    incumbent_eval: dict[str, Any]  # run_and_assess result for the incumbent on dev (label-free only)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    trial_count: int = 0
    max_trials: int = 2

    def record(self, tool: str, args: dict[str, Any], result: Any, task_id: str | None) -> str:
        ev_id = new_id("EV")
        rec = {"id": ev_id, "tool": tool, "args": args, "result": result, "task_id": task_id, "created_at": time.time()}
        self.evidence[ev_id] = rec
        self.store.event(self.run_id, "evidence", rec)
        return ev_id

    def require_dev(self, episode_id: str) -> dict[str, Any]:
        if episode_id not in self.dev_ids:
            raise PermissionError(f"episode {episode_id!r} is not an accessible dev episode; dev episodes: {self.dev_ids}")
        return load_episode(episode_id)


def _pred(ctx: ToolContext, episode_id: str) -> list[dict[str, Any]]:
    return ctx.incumbent_eval["run"]["episodes"][episode_id]["segments"]


def t_self_assessment(ctx: ToolContext, **_: Any) -> Any:
    return compact_assessment(ctx.incumbent_eval["assessment"])


def t_episode_predictions(ctx: ToolContext, episode_id: str, **_: Any) -> Any:
    ep = ctx.require_dev(episode_id)
    sample = next((n for n in ctx.incumbent.workflow.nodes if n.operator == "sample_contact_sheets"), None)
    return {"episode_id": episode_id, "instruction": ep["instruction"], "duration_sec": ep["duration_sec"],
            "has_telemetry": ep["has_telemetry"], "sampling": sample.params if sample else None,
            "predicted_segments": _pred(ctx, episode_id)}


def t_telemetry_hints(ctx: ToolContext, episode_id: str, signals: list[str] | None = None, **_: Any) -> Any:
    ep = ctx.require_dev(episode_id)
    if not ep.get("telemetry_path"):
        return {"episode_id": episode_id, "hints": [], "note": "no telemetry for this episode"}
    p = HintParams(signals=signals or ["gripper"])
    hints = detect_hints(ep["telemetry_path"], p)
    preds = _pred(ctx, episode_id)
    bounds = [s["end_sec"] for s in preds[:-1]]
    for h in hints:
        h["nearest_predicted_boundary_sec"] = min(bounds, key=lambda b: abs(b - h["t"])) if bounds else None
        h["distance_to_predicted_boundary_sec"] = round(min(abs(b - h["t"]) for b in bounds), 2) if bounds else None
    return {"episode_id": episode_id, "signals": p.signals, "n_hints": len(hints), "n_predicted_boundaries": len(bounds), "hints": hints}


def t_sampling_coverage(ctx: ToolContext, episode_id: str, sampling_hz: float = 1.0, **_: Any) -> Any:
    ep = ctx.require_dev(episode_id)
    hz = float(min(max(sampling_hz, 0.25), 5.0))
    ts = uniform_timestamps(ep["duration_sec"], hz)
    preds = _pred(ctx, episode_id)
    return {"episode_id": episode_id, "sampling_hz": hz, "n_frames": len(ts), "frame_spacing_sec": round(1 / hz, 3),
            "duration_sec": ep["duration_sec"],
            "predicted_segments_shorter_than_2_frames": sum(1 for s in preds if s["end_sec"] - s["start_sec"] < 2 / hz),
            "n_predicted_segments": len(preds)}


def t_inspect_frames(ctx: ToolContext, episode_id: str, center_sec: float, window_sec: float = 2.0, local_hz: float = 4.0,
                     question: str = "", task_id: str | None = None, **_: Any) -> Any:
    ep = ctx.require_dev(episode_id)
    window_sec = float(min(max(window_sec, 0.5), 4.0))
    local_hz = float(min(max(local_hz, 1.0), 5.0))
    lo = max(0.0, float(center_sec) - window_sec / 2)
    hi = min(ep["duration_sec"], float(center_sec) + window_sec / 2)
    ts = [round(lo + i / local_hz, 2) for i in range(int((hi - lo) * local_hz) + 1)]
    sheet = build_sheets(ep["video_path"], ts, tile_px=224, frames_per_sheet=len(ts), cols=min(6, len(ts)))[0]
    prompt = (f"Close-up frames from a robot manipulation video (task: {ep['instruction']!r}); timestamps in yellow. "
              f"Question from an investigator: {question[:400]}\n"
              "Return only JSON: {\"observed_state_changes\": [{\"completed_at_sec\": float, \"description\": str}], "
              "\"answer\": str, \"visual_confidence\": float}")
    res = ctx.client.chat(ROLE_VISION, [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(sheet.jpeg)]}],
                          purpose="investigate_inspect_frames", json_mode=True, max_tokens=2048, task_id=task_id,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}}, episode_id=episode_id)
    return {"episode_id": episode_id, "window": [lo, hi], "frame_timestamps": ts, "vision_model": res.model, "observation": res.data}


def t_trial_patch_on_dev(ctx: ToolContext, patch: dict[str, Any], **_: Any) -> Any:
    from evoaction.pipeline import run_and_assess

    if ctx.trial_count >= ctx.max_trials:
        return {"error": f"trial budget exhausted ({ctx.max_trials} per run)"}
    try:
        wp = WorkflowPatch(**patch)
        wf, prompts = apply_patch(ctx.incumbent.workflow, ctx.incumbent.prompts, wp)
    except (WorkflowValidationError, ValueError, TypeError) as e:
        return {"error": f"invalid patch: {e}"}
    ctx.trial_count += 1
    trial = Version(id=f"{ctx.incumbent.id}-devtrial{ctx.trial_count}", path=ctx.incumbent.path, workflow=wf, prompts=prompts,
                    config=ctx.incumbent.config, manifest={"workflow_sha256": "dev-trial"})
    r = run_and_assess(trial, "dev", ctx.dev_ids, ctx.client, ctx.run_id, evaluator=None, store=ctx.store)
    inc = ctx.incumbent_eval["assessment"]
    new = r["assessment"]
    return {"patch_topology_change": wp.is_topology_change(), "workflow_nodes": [n.operator for n in wf.nodes],
            "incumbent_dev_self_score": inc["mean_self_score"], "trial_dev_self_score": new["mean_self_score"],
            "incumbent_component_means": inc["component_means"], "trial_component_means": new["component_means"],
            "trial_per_episode_self_score": new["per_episode_self_score"], "vision_calls": r["calls_used"]}


TOOLS: dict[str, Callable[..., Any]] = {
    "self_assessment": t_self_assessment,
    "episode_predictions": t_episode_predictions,
    "telemetry_hints": t_telemetry_hints,
    "sampling_coverage": t_sampling_coverage,
    "inspect_frames": t_inspect_frames,
}


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any], task_id: str | None, allowed: list[str]) -> tuple[str, Any]:
    if name not in TOOLS:
        result: Any = {"error": f"unknown tool {name!r}; allowed: {sorted(TOOLS)}"}
    elif name not in allowed:
        result = {"error": f"tool {name!r} not allowed for this task; allowed: {allowed}"}
    else:
        try:
            kwargs = dict(args or {})
            if name == "inspect_frames":
                kwargs["task_id"] = task_id
            result = TOOLS[name](ctx, **kwargs)
        except (PermissionError, KeyError, TypeError, ValueError) as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        except (BudgetExceeded, ModelCallError) as e:
            result = {"error": f"{type(e).__name__}: {e}"}
    ev_id = ctx.record(name, args or {}, result, task_id)
    return ev_id, result


def compact(obj: Any, limit: int = 6000) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def condense_assessment(a: dict[str, Any], detailed: int = 15) -> dict[str, Any]:
    """Prompt-sized label-free view: aggregates, every episode's score, and details for the lowest-scoring episodes."""
    if not isinstance(a, dict) or "episodes" not in a:
        return a
    ranked = sorted(a["episodes"].items(), key=lambda kv: kv[1]["self_score"])
    details = {}
    for e, d in ranked[:detailed]:
        vv = d["components"].get("visual_verification") or {}
        ta = d["components"].get("telemetry_agreement") or {}
        details[e] = {"self_score": d["self_score"], "component_values": d["component_values"], "n_segments": d["n_segments"],
                      "predicted_boundaries": d["predicted_boundaries"][:12], "gripper_events": d["gripper_events"][:12],
                      "telemetry_precision_recall": [ta.get("precision"), ta.get("recall")] if ta else None,
                      "visually_confirmed": f"{vv.get('confirmed')}/{vv.get('checked')}" if vv else None,
                      "duration_cv": d["components"]["duration_variability"].get("cv")}
    seg_counts = [d["n_segments"] for d in a["episodes"].values()]
    return {"definition": a.get("definition"), "mean_self_score": a.get("mean_self_score"), "component_means": a.get("component_means"),
            "episodes_total": len(a["episodes"]), "predicted_segments_per_episode": {"min": min(seg_counts, default=0),
            "max": max(seg_counts, default=0), "mean": round(sum(seg_counts) / max(1, len(seg_counts)), 2)},
            "per_episode_self_score": {e: d["self_score"] for e, d in ranked},
            f"lowest_{detailed}_episodes_detail": details}


# Backwards-compatible name used by investigator/supervisor for profile-like records.
condense_profile = condense_assessment


def evidence_item(ev_id: str, tool: Any, args: Any, result: Any, per_record: int = 3500) -> dict[str, Any]:
    if tool == "self_assessment":
        result = condense_assessment(result)
        per_record = 9000
    elif tool == "human_labeled_sample":
        per_record = 12000
    text = json.dumps(result, default=str)
    if len(text) > per_record:
        text = text[:per_record] + "...<truncated>"
    return {"id": ev_id, "tool": tool, "args": args, "result_json": text}
