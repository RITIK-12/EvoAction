"""The only module that reads reference labels.

Access policy (enforced here, not by convention in callers):
  * dev        -> score summaries + per-prediction error classes with *binned* offsets;
                  never raw reference timestamps or labels.
  * validation -> aggregate metrics and paired per-episode F1 only.
  * test       -> sealed. Raises until `unseal_test(freeze_manifest)` is called with a
                  frozen A/B/C manifest; then returns full scores + timelines for the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evoaction.config import Settings, get_settings
from evoaction.evaluation import segments as M


class SealedSplitError(PermissionError):
    pass


def _bin_offset(d: float) -> str:
    a = abs(d)
    side = "late" if d > 0 else "early"
    if a <= 0.5:
        return "within 0.5s"
    if a <= 1.0:
        return f"0.5-1s {side}"
    if a <= 2.0:
        return f"1-2s {side}"
    return f">2s {side}"


class ProtectedEvaluator:
    def __init__(self, settings: Settings | None = None, splits: dict[str, Any] | None = None):
        self.settings = settings or get_settings()
        self._gold_dir: Path = self.settings.protected_dir / "gold"
        if splits is None:
            from evoaction.data.wgo import load_splits

            splits = load_splits(self.settings)
        self._split_of: dict[str, str] = {}
        for name, sp in splits["splits"].items():
            for eid in sp["episode_ids"]:
                self._split_of[eid] = name
        self._test_unsealed = False
        self.freeze_manifest: dict[str, Any] | None = None

    # -- access control --------------------------------------------------
    def split_of(self, episode_id: str) -> str:
        return self._split_of.get(episode_id, "unknown")

    def _require(self, episode_ids: list[str], allowed: set[str]) -> None:
        for e in episode_ids:
            sp = self.split_of(e)
            if sp == "test" and not self._test_unsealed:
                raise SealedSplitError("test split is sealed until A/B/C are frozen")
            if sp not in allowed:
                raise PermissionError(f"episode {e} is in split {sp!r}; this view allows {sorted(allowed)}")

    def unseal_test(self, freeze_manifest: dict[str, Any]) -> None:
        for k in ("A", "B", "C"):
            if not freeze_manifest.get(k, {}).get("workflow_sha256"):
                raise SealedSplitError(f"cannot unseal: system {k} is not frozen with a workflow hash")
        self.freeze_manifest = freeze_manifest
        self._test_unsealed = True

    def _refs(self, episode_id: str) -> list[dict[str, Any]]:
        return json.loads((self._gold_dir / f"{episode_id}.json").read_text())["segments"]

    # -- privileged scoring ----------------------------------------------
    def _score(self, episode_id: str, preds: list[dict[str, Any]], duration: float | None, snap: bool = False) -> M.EpisodeScore:
        return M.score_episode(episode_id, preds, self._refs(episode_id), duration, snap=snap)

    def score_split(self, preds_by_episode: dict[str, list[dict[str, Any]]], durations: dict[str, float]) -> dict[str, Any]:
        """Raw + outer-snapped aggregate for any non-sealed split (internal use by controller/report)."""
        eids = sorted(preds_by_episode)
        self._require(eids, {"dev", "validation", "test"})
        raw = [self._score(e, preds_by_episode[e], durations.get(e)) for e in eids]
        snapped = [self._score(e, preds_by_episode[e], durations.get(e), snap=True) for e in eids]
        return {"protocol": M.PROTOCOL_VERSION, "raw": M.aggregate(raw), "outer_snapped": M.aggregate(snapped),
                "per_episode": {s.episode_id: {k: v for k, v in s.to_dict().items() if k != "matches"} for s in raw}}

    # -- agent-facing views ----------------------------------------------
    def dev_diagnostics(self, preds_by_episode: dict[str, list[dict[str, Any]]], durations: dict[str, float]) -> dict[str, Any]:
        """Controlled development feedback. No reference timestamps or labels leave this method."""
        eids = sorted(preds_by_episode)
        self._require(eids, {"dev"})
        per_ep = {}
        scores = []
        for e in eids:
            refs = self._refs(e)
            preds, invalid = M.normalize_predictions(preds_by_episode[e], durations.get(e))
            sc = M.score_episode(e, preds_by_episode[e], refs, durations.get(e))
            scores.append(sc)
            ref_b = M.internal_boundaries(refs)
            pred_items = []
            for i, p in enumerate(preds):
                ious = [M.temporal_iou((p["start_sec"], p["end_sec"]), (r["start_sec"], r["end_sec"])) for r in refs]
                best = max(range(len(refs)), key=lambda j: ious[j]) if refs else None
                covered = sum(1 for r in refs if (min(p["end_sec"], r["end_sec"]) - max(p["start_sec"], r["start_sec"])) > 0.5 * (r["end_sec"] - r["start_sec"]))
                item = {"pred_index": i, "pred_start_sec": round(p["start_sec"], 2), "pred_end_sec": round(p["end_sec"], 2),
                        "best_iou": round(ious[best], 2) if best is not None else 0.0,
                        "matched_at_0.75": any(pi == i for pi, _, _ in sc.matches),
                        "reference_events_mostly_covered": covered}
                if best is not None:
                    r = refs[best]
                    item["start_error"] = _bin_offset(p["start_sec"] - r["start_sec"])
                    item["end_error"] = _bin_offset(p["end_sec"] - r["end_sec"])
                if covered >= 2:
                    item["error_class"] = "merged: spans multiple reference events"
                elif item["matched_at_0.75"]:
                    item["error_class"] = "correct"
                elif covered == 0 and item["best_iou"] < 0.3:
                    item["error_class"] = "spurious or badly placed"
                else:
                    item["error_class"] = "boundary misaligned"
                pred_items.append(item)
            pred_b = M.internal_boundaries(preds)
            near = lambda b, pts, tol: any(abs(b - q) <= tol for q in pts)  # noqa: E731
            per_ep[e] = {
                "n_pred": sc.n_pred, "n_ref": sc.n_ref, "n_invalid": len(invalid),
                "segmentation": "under-segmented" if sc.n_pred < sc.n_ref else ("over-segmented" if sc.n_pred > sc.n_ref else "count matches"),
                "f1_iou075": round(sc.f1, 3), "f1_iou05": round(sc.f1_05, 3),
                "short_reference_events": sc.short_ref, "short_events_recalled": sc.short_tp,
                "reference_interior_boundaries": len(ref_b),
                "reference_boundaries_with_pred_within_0.5s": sum(1 for b in ref_b if near(b, pred_b, 0.5)),
                "reference_boundaries_with_pred_within_1.0s": sum(1 for b in ref_b if near(b, pred_b, 1.0)),
                "boundary_mae_sec": round(sc.boundary_mae_sec, 2) if sc.boundary_mae_sec is not None else None,
                "predictions": pred_items,
            }
        agg = M.aggregate(scores)
        # Reference durations, bucketed only (no times).
        buckets = {"<2s": 0, "2-5s": 0, "5-10s": 0, ">=10s": 0}
        for e in eids:
            for r in self._refs(e):
                d = r["end_sec"] - r["start_sec"]
                buckets["<2s" if d < 2 else "2-5s" if d < 5 else "5-10s" if d < 10 else ">=10s"] += 1
        return {"split": "dev", "protocol": M.PROTOCOL_VERSION,
                "aggregate": {k: v for k, v in agg.items() if k != "per_episode_f1"},
                "reference_duration_buckets": buckets, "episodes": per_ep}

    def validation_summary(self, preds_by_episode: dict[str, list[dict[str, Any]]], durations: dict[str, float]) -> dict[str, Any]:
        """Aggregate + paired per-episode F1 only."""
        eids = sorted(preds_by_episode)
        self._require(eids, {"validation"})
        res = self.score_split(preds_by_episode, durations)
        raw = res["raw"]
        return {"split": "validation", "protocol": M.PROTOCOL_VERSION,
                "micro_f1": raw["micro_f1"], "micro_f1_iou05": raw["micro_f1_iou05"],
                "short_event_recall": raw["short_event_recall"], "invalid_items": raw["invalid_items"],
                "empty_output_episodes": raw["empty_output_episodes"], "mean_boundary_mae_sec": raw["mean_boundary_mae_sec"],
                "per_episode_f1": raw["per_episode_f1"], "outer_snapped_micro_f1": res["outer_snapped"]["micro_f1"]}

    def final_test_report(self, preds_by_system: dict[str, dict[str, list[dict[str, Any]]]], durations: dict[str, float]) -> dict[str, Any]:
        """Sealed held-out evaluation with timelines; only after unseal_test()."""
        if not self._test_unsealed:
            raise SealedSplitError("test split is sealed until A/B/C are frozen")
        out: dict[str, Any] = {"protocol": M.PROTOCOL_VERSION, "freeze_manifest": self.freeze_manifest, "systems": {}, "timelines": {}}
        for system, preds in preds_by_system.items():
            self._require(sorted(preds), {"test"})
            out["systems"][system] = self.score_split(preds, durations)
        eids = sorted(next(iter(preds_by_system.values())))
        for e in eids:
            out["timelines"][e] = {"reference": self._refs(e), "duration_sec": durations.get(e),
                                   **{sys: M.normalize_predictions(p[e], durations.get(e))[0] for sys, p in preds_by_system.items()}}
        return out
