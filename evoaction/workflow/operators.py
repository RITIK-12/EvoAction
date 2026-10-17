"""Allowlisted executable operators. No operator can read gold or the evaluator.

Each operator: fn(ctx, state, params) -> (state_updates, detail).
State keys: episode, sheets, segments, hints, evidence, export.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable

import numpy as np

from evoaction.clients.crusoe import ROLE_VISION, BudgetExceeded, CrusoeClient, ModelCallError, image_part
from evoaction.media.contact_sheets import build_sheets, uniform_timestamps
from evoaction.schemas import (
    AnnotateParams, ExportParams, HintParams, InspectParams, MergeParams, RefineParams, SampleParams, Segment,
)

SHORT_SEC = 2.0


class OperatorBudgetError(RuntimeError):
    pass


@dataclass
class EpisodeContext:
    episode: dict[str, Any]
    prompts: dict[str, str]
    client: CrusoeClient | None
    version_id: str
    run_id: str | None
    out_dir: Path
    max_calls: int
    calls_used: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def take_call(self) -> None:
        if self.calls_used + 1 > self.max_calls:
            raise OperatorBudgetError(f"workflow max_calls={self.max_calls} per episode reached")
        self.calls_used += 1


# -- requirements (used by the validator) ------------------------------------

REQUIRES: dict[str, set[str]] = {
    "sample_contact_sheets": {"episode"},
    "initial_annotate": {"sheets"},
    "state_hint_detection": {"episode"},
    "inspect_transition_windows": {"segments"},
    "refine_intervals": {"segments", "evidence"},
    "merge_filter_segments": {"segments"},
    "validate_export": {"segments"},
}
PROVIDES: dict[str, set[str]] = {
    "sample_contact_sheets": {"sheets"},
    "initial_annotate": {"segments"},
    "state_hint_detection": {"hints"},
    "inspect_transition_windows": {"evidence"},
    "refine_intervals": {"segments"},
    "merge_filter_segments": {"segments"},
    "validate_export": {"export"},
}
MODEL_OPERATORS = {"initial_annotate", "refine_intervals"}


def extra_requires(operator: str, params: dict[str, Any]) -> set[str]:
    if operator == "inspect_transition_windows":
        p = InspectParams(**params)
        if p.select in ("near_hints", "hint_disagreement") or p.include_unmatched_hints:
            return {"hints"}
    return set()


def _segments_from_model(data: Any, duration: float) -> list[dict[str, Any]]:
    items = data.get("segments", data.get("events", [])) if isinstance(data, dict) else data
    out = []
    for it in items or []:
        try:
            seg = Segment(start_sec=float(it["start_sec"]), end_sec=float(it["end_sec"]),
                          subtask=str(it.get("subtask", "")),
                          confidence=float(it["confidence"]) if it.get("confidence") is not None else None)
        except (KeyError, TypeError, ValueError):
            continue
        out.append(seg.model_dump())
    out.sort(key=lambda s: (s["start_sec"], s["end_sec"]))
    return out


def _render(template: str, **kw: Any) -> str:
    return Template(template).safe_substitute(**{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in kw.items()})


# -- operators ----------------------------------------------------------------

def op_sample_contact_sheets(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = SampleParams(**params)
    ep = state["episode"]
    ts = uniform_timestamps(ep["duration_sec"], p.sampling_hz)
    sheets = build_sheets(ep["video_path"], ts, tile_px=p.tile_px, frames_per_sheet=p.frames_per_sheet, cols=p.cols,
                          badge=p.badge, jpeg_quality=p.jpeg_quality)
    return {"sheets": {"sheets": sheets, "sampling_hz": p.sampling_hz, "n_frames": len(ts)}}, {
        "n_frames": len(ts), "n_sheets": len(sheets), "sheet_sha256": [s.sha256 for s in sheets]}


def op_initial_annotate(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = AnnotateParams(**params)
    ep = state["episode"]
    sh = state["sheets"]
    prompt = _render(ctx.prompts[p.prompt], instruction=ep["instruction"], duration_sec=ep["duration_sec"],
                     sampling_hz=sh["sampling_hz"], frame_count=sh["n_frames"], sheet_count=len(sh["sheets"]))
    content = [{"type": "text", "text": prompt}] + [image_part(s.jpeg) for s in sh["sheets"]]
    ctx.take_call()
    res = ctx.client.chat(ROLE_VISION, [{"role": "user", "content": content}], purpose="initial_annotate", json_mode=True,
                          max_tokens=p.max_tokens, temperature=p.temperature,
                          extra_body={"chat_template_kwargs": {"enable_thinking": p.thinking}},
                          cache_scope=f"initial_annotate|{json.dumps(params, sort_keys=True)}",
                          version_id=ctx.version_id, episode_id=ep["id"])
    segs = _segments_from_model(res.data, ep["duration_sec"])
    return {"segments": segs}, {"n_segments": len(segs), "call_id": res.call_id, "cache_hit": res.cache_hit,
                                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                                "cost_usd": res.cost_usd}


def _load_channels(telemetry_path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t = json.loads(Path(telemetry_path).read_text())
    ts = np.asarray(t["timestamp_sec"], dtype=float)
    chans: dict[str, np.ndarray] = {}
    for ch in t["channels"]:
        vals = np.asarray(ch["values"], dtype=float)
        if vals.ndim == 1:
            vals = vals[:, None]
        for k, name in enumerate(ch["value_names"]):
            if k < vals.shape[1]:
                chans[f"{ch['name']}:{name}"] = vals[:, k]
    return ts, chans


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def detect_hints(telemetry_path: str, p: HintParams) -> list[dict[str, Any]]:
    ts, chans = _load_channels(telemetry_path)
    if len(ts) < 3:
        return []
    fps = 1.0 / max(np.median(np.diff(ts)), 1e-3)
    hints: list[dict[str, Any]] = []
    if "gripper" in p.signals:
        for name, x in chans.items():
            low = name.lower()
            if "gripper" not in low or not any(k in low for k in ("gripper_state", "open_scale", "open_state")):
                continue
            rng = float(np.nanmax(x) - np.nanmin(x))
            if not np.isfinite(rng) or rng < 0.2 * max(abs(float(np.nanmax(x))), 1e-6) or rng < 1e-3:
                continue
            xs = _smooth((x - np.nanmin(x)) / rng, max(1, int(fps * 0.2)))
            state = xs[0] > 0.5
            last_t = -1e9
            for i in range(1, len(xs)):
                # hysteresis: transition completes when the signal settles past 0.3/0.7
                if state and xs[i] < 0.3:
                    state = False
                    kind = "gripper_closed" if "open" in low else "gripper_released"
                elif not state and xs[i] > 0.7:
                    state = True
                    kind = "gripper_opened" if "open" in low else "gripper_engaged"
                else:
                    continue
                if ts[i] - last_t >= 0.3:
                    hints.append({"t": round(float(ts[i]), 2), "kind": kind, "channel": name, "strength": 1.0})
                    last_t = ts[i]
    if "eef_pause" in p.signals:
        speed = None
        vel = [v for k, v in chans.items() if "velocity" in k.lower()]
        if vel:
            speed = np.max(np.abs(np.stack(vel, 1)), 1)
        else:
            pos = [v for k, v in chans.items() if k.endswith((":x", ":y", ":z")) or "pos_" in k.lower()]
            if pos:
                P = np.stack(pos, 1)
                speed = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) * fps]
        if speed is not None and len(speed) == len(ts):
            s = _smooth(speed, max(1, int(fps * 0.3)))
            thr = float(np.quantile(s, p.pause_speed_quantile))
            below = s <= thr
            i = 0
            while i < len(below):
                if below[i]:
                    j = i
                    while j + 1 < len(below) and below[j + 1]:
                        j += 1
                    if ts[j] - ts[i] >= p.min_pause_sec and 0.5 < ts[i] and ts[j] < ts[-1] - 0.5:
                        hints.append({"t": round(float((ts[i] + ts[j]) / 2), 2), "kind": "eef_pause", "channel": "speed",
                                      "strength": 0.5})
                    i = j + 1
                else:
                    i += 1
    hints.sort(key=lambda h: h["t"])
    merged: list[dict[str, Any]] = []
    for h in hints:
        if merged and h["t"] - merged[-1]["t"] < p.min_gap_sec:
            if h["strength"] > merged[-1]["strength"]:
                merged[-1] = h
            continue
        merged.append(h)
    return merged


def op_state_hint_detection(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = HintParams(**params)
    ep = state["episode"]
    if not ep.get("telemetry_path"):
        return {"hints": []}, {"skipped": "no telemetry channels"}
    hints = detect_hints(ep["telemetry_path"], p)
    return {"hints": hints}, {"n_hints": len(hints), "hints": hints}


def _boundaries(segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i in range(len(segs) - 1):
        b = (segs[i]["end_sec"] + segs[i + 1]["start_sec"]) / 2
        confs = [c for c in (segs[i].get("confidence"), segs[i + 1].get("confidence")) if c is not None]
        out.append({"index": i, "t": b, "min_conf": min(confs) if confs else None,
                    "short_adjacent": min(segs[i]["end_sec"] - segs[i]["start_sec"],
                                          segs[i + 1]["end_sec"] - segs[i + 1]["start_sec"]) < SHORT_SEC,
                    "before": segs[i].get("subtask", ""), "after": segs[i + 1].get("subtask", "")})
    return out


def op_inspect_transition_windows(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = InspectParams(**params)
    ep = state["episode"]
    segs = state["segments"]
    hints = state.get("hints") or []
    half = p.window_sec / 2
    bounds = _boundaries(segs)
    near_hint = lambda t: any(abs(h["t"] - t) <= half for h in hints)  # noqa: E731
    if p.select == "all_boundaries":
        chosen = bounds
    elif p.select == "low_confidence":
        chosen = [b for b in bounds if b["min_conf"] is None or b["min_conf"] < p.confidence_threshold]
    elif p.select == "near_hints":
        chosen = [b for b in bounds if near_hint(b["t"])]
    elif p.select == "hint_disagreement":
        chosen = [b for b in bounds if not near_hint(b["t"])]
    else:  # short_segments
        chosen = [b for b in bounds if b["short_adjacent"]]
    windows = [{"kind": "boundary", "center": b["t"], "boundary_index": b["index"], "before": b["before"], "after": b["after"]}
               for b in chosen]
    if p.include_unmatched_hints:
        for h in hints:
            if not any(abs(h["t"] - b["t"]) <= half for b in bounds):
                windows.append({"kind": "hint", "center": h["t"], "boundary_index": None, "hint": h["kind"]})
    windows.sort(key=lambda w: w["center"])
    windows = windows[: p.max_windows]
    evidence = []
    for k, w in enumerate(windows):
        lo = max(0.0, w["center"] - half)
        hi = min(ep["duration_sec"], w["center"] + half)
        ts = [round(lo + i / p.local_hz, 2) for i in range(int(math.floor((hi - lo) * p.local_hz)) + 1)]
        if hi <= lo or not ts:  # window centred outside the video: nothing to inspect
            continue
        sheet = build_sheets(ep["video_path"], ts, tile_px=p.tile_px, frames_per_sheet=len(ts), cols=min(len(ts), 6))[0]
        evidence.append({**w, "window_id": f"W{k + 1}", "start_sec": lo, "end_sec": hi, "timestamps": ts, "sheet": sheet})
    return {"evidence": evidence}, {"n_boundaries": len(bounds), "n_windows": len(evidence), "select": p.select,
                                    "windows": [{k: v for k, v in e.items() if k != "sheet"} for e in evidence]}


def op_refine_intervals(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = RefineParams(**params)
    ep = state["episode"]
    segs = [dict(s) for s in state["segments"]]
    evidence = state.get("evidence") or []
    if not evidence:
        return {"segments": segs}, {"skipped": "no transition windows selected"}
    win_desc = []
    for e in evidence:
        if e["kind"] == "boundary":
            win_desc.append(f"{e['window_id']}: frames {e['start_sec']:.2f}-{e['end_sec']:.2f}s; current boundary at {e['center']:.2f}s "
                            f"between '{e['before']}' and '{e['after']}'.")
        else:
            win_desc.append(f"{e['window_id']}: frames {e['start_sec']:.2f}-{e['end_sec']:.2f}s; no current boundary here; "
                            f"robot telemetry suggests a possible state change ({e.get('hint')}) near {e['center']:.2f}s.")
    seg_desc = json.dumps([{"i": i, "start_sec": round(s["start_sec"], 2), "end_sec": round(s["end_sec"], 2), "subtask": s["subtask"]}
                           for i, s in enumerate(segs)])
    prompt = _render(ctx.prompts[p.prompt], instruction=ep["instruction"], duration_sec=ep["duration_sec"],
                     segments_json=seg_desc, windows="\n".join(win_desc))
    content = [{"type": "text", "text": prompt}]
    for e in evidence:
        content.append({"type": "text", "text": f"Window {e['window_id']}:"})
        content.append(image_part(e["sheet"].jpeg))
    ctx.take_call()
    res = ctx.client.chat(ROLE_VISION, [{"role": "user", "content": content}], purpose="refine_intervals", json_mode=True,
                          max_tokens=p.max_tokens, temperature=p.temperature,
                          extra_body={"chat_template_kwargs": {"enable_thinking": p.thinking}},
                          cache_scope=f"refine_intervals|{json.dumps(params, sort_keys=True)}",
                          version_id=ctx.version_id, episode_id=ep["id"])
    decisions = res.data.get("windows", []) if isinstance(res.data, dict) else (res.data or [])
    by_id = {e["window_id"]: e for e in evidence}
    applied = []
    # Apply moves/deletes on boundaries first (by index), then inserts.
    moves: dict[int, float] = {}
    deletes: set[int] = set()
    inserts: list[tuple[float, str]] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        e = by_id.get(str(d.get("window_id")))
        if e is None:
            continue
        action = str(d.get("action", "keep")).lower()
        t = d.get("boundary_sec")
        try:
            t = float(t) if t is not None else None
        except (TypeError, ValueError):
            t = None
        if e["kind"] == "boundary":
            i = e["boundary_index"]
            if action == "move" and t is not None and e["start_sec"] - 0.01 <= t <= e["end_sec"] + 0.01:
                moves[i] = t
                applied.append({"window": e["window_id"], "action": "move", "to": round(t, 2)})
            elif action == "delete" and p.allow_delete:
                deletes.add(i)
                applied.append({"window": e["window_id"], "action": "delete"})
            else:
                applied.append({"window": e["window_id"], "action": "keep"})
        elif e["kind"] == "hint" and action == "insert" and p.allow_insert and t is not None:
            inserts.append((t, str(d.get("new_subtask", ""))))
            applied.append({"window": e["window_id"], "action": "insert", "at": round(t, 2)})
    for i, t in moves.items():
        lo = segs[i]["start_sec"] + 0.2
        hi = segs[i + 1]["end_sec"] - 0.2
        if lo < t < hi:
            segs[i]["end_sec"] = t
            segs[i + 1]["start_sec"] = t
    for i in sorted(deletes, reverse=True):
        if i + 1 < len(segs):
            segs[i]["end_sec"] = segs[i + 1]["end_sec"]
            segs[i]["subtask"] = f"{segs[i]['subtask']}; {segs[i + 1]['subtask']}".strip("; ")
            del segs[i + 1]
    for t, label in inserts:
        for j, s in enumerate(segs):
            if s["start_sec"] + 0.3 < t < s["end_sec"] - 0.3:
                new = dict(s, start_sec=t, subtask=label or s["subtask"])
                s["end_sec"] = t
                segs.insert(j + 1, new)
                break
    return {"segments": segs}, {"call_id": res.call_id, "cache_hit": res.cache_hit, "decisions": applied,
                                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                                "cost_usd": res.cost_usd}


def op_merge_filter_segments(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    p = MergeParams(**params)
    dur = state["episode"]["duration_sec"]
    segs = sorted((dict(s) for s in state["segments"]), key=lambda s: (s["start_sec"], s["end_sec"]))
    segs = [dict(s, start_sec=max(0.0, s["start_sec"]), end_sec=min(dur, s["end_sec"])) for s in segs]
    segs = [s for s in segs if s["end_sec"] > s["start_sec"]]
    out: list[dict[str, Any]] = []
    for s in segs:
        if out:
            prev = out[-1]
            if s["start_sec"] < prev["end_sec"]:  # overlap -> split at midpoint
                mid = (s["start_sec"] + prev["end_sec"]) / 2
                if s["end_sec"] <= prev["end_sec"]:
                    continue  # fully contained duplicate
                prev["end_sec"] = mid
                s["start_sec"] = mid
            elif s["start_sec"] - prev["end_sec"] <= p.max_gap_sec:
                prev["end_sec"] = s["start_sec"]
            if p.merge_same_label and s["subtask"].strip().lower() == prev["subtask"].strip().lower():
                prev["end_sec"] = s["end_sec"]
                continue
        out.append(s)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, s in enumerate(out):
            if s["end_sec"] - s["start_sec"] < p.min_duration_sec:
                if i > 0:
                    out[i - 1]["end_sec"] = s["end_sec"]
                else:
                    out[i + 1]["start_sec"] = s["start_sec"]
                del out[i]
                changed = True
                break
    if p.tile_full_episode and out:
        out[0]["start_sec"] = 0.0
        out[-1]["end_sec"] = dur
        for a, b in zip(out, out[1:]):
            if b["start_sec"] > a["end_sec"]:
                a["end_sec"] = b["start_sec"]
    return {"segments": out}, {"n_in": len(state["segments"]), "n_out": len(out)}


def op_validate_export(ctx: EpisodeContext, state: dict, params: dict) -> tuple[dict, dict]:
    ExportParams(**params)
    dur = state["episode"]["duration_sec"]
    segs = []
    problems = []
    for s in state["segments"]:
        try:
            seg = Segment(**{k: s[k] for k in ("start_sec", "end_sec", "subtask") if k in s})
        except Exception as e:  # schema failure is recorded, item dropped
            problems.append(str(e)[:120])
            continue
        if seg.end_sec <= seg.start_sec or seg.start_sec < 0 or seg.start_sec > dur + 0.5:
            problems.append(f"invalid interval {seg.start_sec}-{seg.end_sec}")
            continue
        segs.append({"start_sec": round(seg.start_sec, 3), "end_sec": round(min(seg.end_sec, dur), 3), "subtask": seg.subtask})
    segs.sort(key=lambda s: s["start_sec"])
    monotonic = all(a["end_sec"] <= b["start_sec"] + 1e-6 for a, b in zip(segs, segs[1:]))
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.out_dir / f"{state['episode']['id']}.json"
    path.write_text(json.dumps({"episode_id": state["episode"]["id"], "version_id": ctx.version_id, "segments": segs}, indent=1))
    return {"export": {"path": str(path), "segments": segs}, "segments": segs}, {
        "n_segments": len(segs), "problems": problems, "monotonic": monotonic}


OPERATORS: dict[str, Callable[[EpisodeContext, dict, dict], tuple[dict, dict]]] = {
    "sample_contact_sheets": op_sample_contact_sheets,
    "initial_annotate": op_initial_annotate,
    "state_hint_detection": op_state_hint_detection,
    "inspect_transition_windows": op_inspect_transition_windows,
    "refine_intervals": op_refine_intervals,
    "merge_filter_segments": op_merge_filter_segments,
    "validate_export": op_validate_export,
}

__all__ = ["OPERATORS", "REQUIRES", "PROVIDES", "MODEL_OPERATORS", "EpisodeContext", "OperatorBudgetError",
           "extra_requires", "detect_hints", "BudgetExceeded", "ModelCallError"]
