"""Label-free self-assessment of annotations. Never reads reference labels.

Inputs are only what a deployed system has: its own predicted segments, the video,
robot telemetry, the task instruction and the workflow's own sampling settings.
The score definition below is predeclared (fixed before any v2 run) and is the
only quality signal agents and the promotion gate ever see.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from evoaction.clients.crusoe import ROLE_VISION, BudgetExceeded, CrusoeClient, ModelCallError, image_part
from evoaction.data.wgo import load_episode
from evoaction.media.contact_sheets import build_sheets
from evoaction.schemas import HintParams
from evoaction.workflow.operators import detect_hints

SCORE_DEFINITION = {
    "version": "label-free-v1",
    "components": {
        "telemetry_agreement": "F1 between predicted interior boundaries and gripper open/close transitions from robot telemetry (tolerance 0.75 s)",
        "visual_verification": "fraction of predicted interior boundaries where the Crusoe vision model, shown a close-up 2 s frame strip, confirms a manipulation state change completes",
        "duration_variability": "min(1, CV(segment durations) / 0.5); evenly spaced segments (an artifact of tiled sampling) score low",
        "temporal_resolution": "1 - fraction of segments shorter than two sampled frames of the workflow's own sampling rate",
    },
    "weights": {"telemetry_agreement": 0.35, "visual_verification": 0.35, "duration_variability": 0.15, "temporal_resolution": 0.15},
    "aggregation": "weighted mean over the components available for an episode (weights renormalized), then mean over episodes",
    "gt_access": "none",
}
TOLERANCE_SEC = 0.75
VERIFY_WINDOW_SEC = 2.0
VERIFY_HZ = 3.0
VERIFY_MAX_WINDOWS = 8


def interior_boundaries(segs: list[dict[str, Any]]) -> list[float]:
    segs = sorted(segs, key=lambda s: s["start_sec"])
    return [round((a["end_sec"] + b["start_sec"]) / 2, 3) for a, b in zip(segs, segs[1:])]


def telemetry_agreement(boundaries: list[float], hints: list[dict[str, Any]], tol: float = TOLERANCE_SEC) -> dict[str, Any] | None:
    grip = [h["t"] for h in hints if str(h.get("kind", "")).startswith("gripper")]
    if not grip:
        return None
    if not boundaries:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "gripper_events": len(grip), "boundaries": 0}
    p = sum(1 for b in boundaries if any(abs(b - g) <= tol for g in grip)) / len(boundaries)
    r = sum(1 for g in grip if any(abs(b - g) <= tol for b in boundaries)) / len(grip)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"f1": round(f, 4), "precision": round(p, 4), "recall": round(r, 4), "gripper_events": len(grip), "boundaries": len(boundaries)}


def duration_variability(segs: list[dict[str, Any]]) -> dict[str, Any]:
    d = np.array([s["end_sec"] - s["start_sec"] for s in segs], dtype=float)
    if len(d) < 2 or d.mean() <= 0:
        return {"score": 0.0, "cv": 0.0, "n_segments": int(len(d))}
    cv = float(d.std() / d.mean())
    return {"score": round(min(1.0, cv / 0.5), 4), "cv": round(cv, 4), "n_segments": int(len(d))}


def temporal_resolution(segs: list[dict[str, Any]], sampling_hz: float | None) -> dict[str, Any]:
    if not segs or not sampling_hz:
        return {"score": 0.0 if not segs else 1.0, "sub_resolution_segments": 0}
    min_len = 2.0 / sampling_hz
    short = sum(1 for s in segs if s["end_sec"] - s["start_sec"] < min_len - 1e-6)
    return {"score": round(1 - short / len(segs), 4), "sub_resolution_segments": short, "min_resolvable_sec": round(min_len, 3)}


def visual_verification(client: CrusoeClient | None, ep: dict[str, Any], segs: list[dict[str, Any]], version_id: str) -> dict[str, Any] | None:
    bounds = interior_boundaries(segs)
    if not bounds or client is None:
        return None
    bounds = bounds[:VERIFY_MAX_WINDOWS]
    half = VERIFY_WINDOW_SEC / 2
    content: list[dict[str, Any]] = [{"type": "text", "text": (
        f"You are checking a robot manipulation video annotation (task: {ep['instruction']!r}). Each image below is a close-up strip of "
        f"consecutive frames (timestamps in yellow) around one proposed segment boundary. For each window decide whether a manipulation "
        f"state change is COMPLETED inside the window (e.g. object grasped and lifted, object released at its destination, drawer closed). "
        f"Small motions or approach movements do not count. Return only JSON: "
        f"{{\"windows\": [{{\"window_id\": \"W1\", \"state_change_completed\": true|false, \"completed_at_sec\": <float or null>}}]}}")}]
    for k, b in enumerate(bounds):
        lo, hi = max(0.0, b - half), min(ep["duration_sec"], b + half)
        ts = [round(lo + i / VERIFY_HZ, 2) for i in range(int(math.floor((hi - lo) * VERIFY_HZ)) + 1)]
        sheet = build_sheets(ep["video_path"], ts, tile_px=224, frames_per_sheet=len(ts), cols=min(len(ts), 7))[0]
        content.append({"type": "text", "text": f"Window W{k + 1} (proposed boundary at {b:.2f}s):"})
        content.append(image_part(sheet.jpeg))
    try:
        res = client.chat(ROLE_VISION, [{"role": "user", "content": content}], purpose="label_free_verify", json_mode=True,
                          max_tokens=2048, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                          version_id=version_id, episode_id=ep["id"])
    except (BudgetExceeded, ModelCallError) as e:
        return {"error": str(e)[:200]}
    items = res.data.get("windows", []) if isinstance(res.data, dict) else []
    verdicts = {}
    for it in items:
        if isinstance(it, dict) and str(it.get("window_id", "")).startswith("W"):
            verdicts[str(it["window_id"])] = it
    confirmed, offsets = 0, []
    for k, b in enumerate(bounds):
        v = verdicts.get(f"W{k + 1}")
        if v and v.get("state_change_completed") is True:
            confirmed += 1
            try:
                offsets.append(abs(float(v.get("completed_at_sec")) - b))
            except (TypeError, ValueError):
                pass
    return {"rate": round(confirmed / len(bounds), 4), "confirmed": confirmed, "checked": len(bounds),
            "mean_offset_to_confirmed_change_sec": round(float(np.mean(offsets)), 3) if offsets else None,
            "call_id": res.call_id, "cache_hit": res.cache_hit, "cost_usd": res.cost_usd}


def assess_episode(client: CrusoeClient | None, episode_id: str, segs: list[dict[str, Any]], sampling_hz: float | None,
                   version_id: str, verify: bool = True) -> dict[str, Any]:
    ep = load_episode(episode_id)
    bounds = interior_boundaries(segs)
    hints = detect_hints(ep["telemetry_path"], HintParams(signals=["gripper"])) if ep.get("telemetry_path") else []
    comp: dict[str, Any] = {
        "telemetry_agreement": telemetry_agreement(bounds, hints),
        "visual_verification": visual_verification(client, ep, segs, version_id) if verify else None,
        "duration_variability": duration_variability(segs),
        "temporal_resolution": temporal_resolution(segs, sampling_hz),
    }
    values = {
        "telemetry_agreement": comp["telemetry_agreement"]["f1"] if comp["telemetry_agreement"] else None,
        "visual_verification": comp["visual_verification"].get("rate") if comp["visual_verification"] else None,
        "duration_variability": comp["duration_variability"]["score"],
        "temporal_resolution": comp["temporal_resolution"]["score"],
    }
    w = SCORE_DEFINITION["weights"]
    avail = {k: v for k, v in values.items() if v is not None}
    score = sum(w[k] * v for k, v in avail.items()) / sum(w[k] for k in avail) if avail else 0.0
    if not segs:
        score = 0.0
    return {"episode_id": episode_id, "self_score": round(score, 4), "component_values": values, "components": comp,
            "n_segments": len(segs), "predicted_boundaries": bounds, "gripper_events": [h["t"] for h in hints]}


def assess_run(client: CrusoeClient | None, run_result: dict[str, Any], sampling_hz: float | None, version_id: str,
               verify: bool = True) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    items = list(run_result["episodes"].items())
    from evoaction.config import get_settings

    with ThreadPoolExecutor(max_workers=get_settings().workers) as ex:
        results = list(ex.map(lambda kv: assess_episode(client, kv[0], kv[1]["segments"], sampling_hz, version_id, verify), items))
    eps = {a["episode_id"]: a for a in results}
    scores = [a["self_score"] for a in eps.values()]
    comp_means = {}
    for k in SCORE_DEFINITION["weights"]:
        vals = [a["component_values"][k] for a in eps.values() if a["component_values"][k] is not None]
        comp_means[k] = round(float(np.mean(vals)), 4) if vals else None
    return {"definition": SCORE_DEFINITION["version"], "mean_self_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "per_episode_self_score": {e: a["self_score"] for e, a in eps.items()}, "component_means": comp_means, "episodes": eps}


def sampling_hz_of(version) -> float | None:
    for n in version.workflow.nodes:
        if n.operator == "sample_contact_sheets":
            return float(n.params.get("sampling_hz", 1.0))
    return None


def compact_assessment(a: dict[str, Any]) -> dict[str, Any]:
    """Agent-facing view: everything is label-free already; drop call ids."""
    out = json.loads(json.dumps(a, default=str))
    for ep in out.get("episodes", {}).values():
        vv = ep["components"].get("visual_verification")
        if isinstance(vv, dict):
            for k in ("call_id", "cache_hit", "cost_usd"):
                vv.pop(k, None)
    return out
