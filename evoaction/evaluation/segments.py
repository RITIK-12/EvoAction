"""Pure temporal-segmentation metrics (no I/O, no gold access).

Primary metric: micro Segment F1 at temporal IoU >= 0.75 using a
maximum-cardinality one-to-one bipartite matching over the thresholded
IoU adjacency (not greedy, not max-IoU-then-threshold).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

PRIMARY_IOU = 0.75
SECONDARY_IOU = 0.5
SHORT_EVENT_SEC = 2.0
PROTOCOL_VERSION = "evoaction-temporal-v1"


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def normalize_predictions(preds: list[dict[str, Any]], duration: float | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (valid, invalid). Valid intervals are finite, start < end, inside [0, duration], sorted by start."""
    valid, invalid = [], []
    for p in preds or []:
        try:
            s, e = float(p["start_sec"]), float(p["end_sec"])
        except (KeyError, TypeError, ValueError):
            invalid.append({"item": p, "reason": "missing/non-numeric start_sec/end_sec"})
            continue
        if not (math.isfinite(s) and math.isfinite(e)):
            invalid.append({"item": p, "reason": "non-finite"})
        elif s < 0 or e <= s:
            invalid.append({"item": p, "reason": "start<0 or end<=start"})
        elif duration is not None and s >= duration + 0.5:
            invalid.append({"item": p, "reason": "starts after video end"})
        else:
            if duration is not None:
                e = min(e, duration)
            valid.append({"start_sec": s, "end_sec": e, "subtask": str(p.get("subtask", ""))})
    valid.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    return valid, invalid


def match(preds: list[dict[str, Any]], refs: list[dict[str, Any]], iou_thr: float) -> list[tuple[int, int, float]]:
    """Maximum-cardinality one-to-one matching on the IoU >= thr adjacency. Returns (pred_idx, ref_idx, iou)."""
    if not preds or not refs:
        return []
    iou = np.array([[temporal_iou((p["start_sec"], p["end_sec"]), (r["start_sec"], r["end_sec"])) for r in refs] for p in preds])
    adj = csr_matrix((iou >= iou_thr - 1e-12).astype(np.int8))
    pred_to_ref = maximum_bipartite_matching(adj, perm_type="column")
    return [(i, int(j), float(iou[i, j])) for i, j in enumerate(pred_to_ref) if j >= 0]


@dataclass
class EpisodeScore:
    episode_id: str
    n_pred: int
    n_ref: int
    n_invalid: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    tp_05: int
    f1_05: float
    short_ref: int
    short_tp: int
    boundary_mae_sec: float | None
    matches: list[tuple[int, int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prf(tp: int, n_pred: int, n_ref: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    if n_pred == 0 and n_ref == 0:
        return 1.0, 1.0, 1.0
    return p, r, f


def internal_boundaries(segs: list[dict[str, Any]]) -> list[float]:
    """Interior change points: every segment start except the earliest, merged within 0.05 s."""
    pts = sorted({round(s["start_sec"], 3) for s in segs} | {round(s["end_sec"], 3) for s in segs})
    if len(pts) <= 2:
        return []
    inner = pts[1:-1]
    merged: list[float] = []
    for p in inner:
        if not merged or p - merged[-1] > 0.05:
            merged.append(p)
    return merged


def boundary_mae(preds: list[dict[str, Any]], refs: list[dict[str, Any]]) -> float | None:
    """Mean |distance| from each reference interior boundary to the nearest predicted boundary (report-only)."""
    rb = internal_boundaries(refs)
    pb = internal_boundaries(preds) + ([preds[0]["start_sec"], preds[-1]["end_sec"]] if preds else [])
    if not rb:
        return None
    if not pb:
        return None
    return float(np.mean([min(abs(r - p) for p in pb) for r in rb]))


def outer_snap(preds: list[dict[str, Any]], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluator-only variant: snap first predicted start / last predicted end to reference outer bounds.

    Returns a copy; raw predictions are never mutated or written back.
    """
    if not preds or not refs:
        return [dict(p) for p in preds]
    out = [dict(p) for p in preds]
    out[0]["start_sec"] = min(r["start_sec"] for r in refs)
    out[-1]["end_sec"] = max(r["end_sec"] for r in refs)
    out = [p for p in out if p["end_sec"] > p["start_sec"]]
    return out


def score_episode(episode_id: str, raw_preds: list[dict[str, Any]], refs: list[dict[str, Any]],
                  duration: float | None = None, snap: bool = False) -> EpisodeScore:
    preds, invalid = normalize_predictions(raw_preds, duration)
    if snap:
        preds = outer_snap(preds, refs)
    m = match(preds, refs, PRIMARY_IOU)
    m05 = match(preds, refs, SECONDARY_IOU)
    tp = len(m)
    p, r, f = _prf(tp, len(preds), len(refs))
    _, _, f05 = _prf(len(m05), len(preds), len(refs))
    short_idx = {j for j, s in enumerate(refs) if s["end_sec"] - s["start_sec"] < SHORT_EVENT_SEC}
    short_tp = sum(1 for _, j, _ in m if j in short_idx)
    return EpisodeScore(
        episode_id=episode_id, n_pred=len(preds), n_ref=len(refs), n_invalid=len(invalid),
        tp=tp, fp=len(preds) - tp, fn=len(refs) - tp, precision=p, recall=r, f1=f,
        tp_05=len(m05), f1_05=f05, short_ref=len(short_idx), short_tp=short_tp,
        boundary_mae_sec=boundary_mae(preds, refs), matches=m,
    )


def aggregate(scores: list[EpisodeScore]) -> dict[str, Any]:
    tp = sum(s.tp for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    n_ref = sum(s.n_ref for s in scores)
    p, r, f = _prf(tp, n_pred, n_ref)
    _, _, f05 = _prf(sum(s.tp_05 for s in scores), n_pred, n_ref)
    short_ref = sum(s.short_ref for s in scores)
    maes = [s.boundary_mae_sec for s in scores if s.boundary_mae_sec is not None]
    return {
        "episodes": len(scores),
        "micro_precision": p, "micro_recall": r, "micro_f1": f,
        "micro_f1_iou05": f05,
        "tp": tp, "fp": n_pred - tp, "fn": n_ref - tp, "n_pred": n_pred, "n_ref": n_ref,
        "short_event_recall": (sum(s.short_tp for s in scores) / short_ref) if short_ref else None,
        "short_ref": short_ref,
        "invalid_items": sum(s.n_invalid for s in scores),
        "empty_output_episodes": sum(1 for s in scores if s.n_pred == 0),
        "mean_boundary_mae_sec": float(np.mean(maes)) if maes else None,
        "per_episode_f1": {s.episode_id: s.f1 for s in scores},
    }


def paired_comparison(candidate: dict[str, float], incumbent: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(candidate) & set(incumbent))
    deltas = {e: candidate[e] - incumbent[e] for e in common}
    return {
        "episodes": common,
        "per_episode_delta": deltas,
        "non_regressing_fraction": (sum(1 for d in deltas.values() if d >= -1e-9) / len(deltas)) if deltas else 0.0,
        "improved": sum(1 for d in deltas.values() if d > 1e-9),
        "regressed": sum(1 for d in deltas.values() if d < -1e-9),
        "mean_delta": float(np.mean(list(deltas.values()))) if deltas else 0.0,
    }
