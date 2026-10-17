import json

import pytest

from evoaction.evaluation import segments as M
from evoaction.evaluation.protected import ProtectedEvaluator, SealedSplitError


def seg(s, e, label="x"):
    return {"start_sec": s, "end_sec": e, "subtask": label}


def test_temporal_iou():
    assert M.temporal_iou((0, 10), (0, 10)) == 1.0
    assert M.temporal_iou((0, 10), (10, 20)) == 0.0
    assert M.temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)


def test_maximum_cardinality_beats_greedy():
    # Greedy-by-IoU pairs P1-R1 (0.961) and strands P2; max matching finds P1-R2 + P2-R1.
    refs = [seg(0, 10), seg(0.5, 10.5)]
    preds = [seg(0.2, 10.2), seg(0, 8)]
    assert M.temporal_iou((0, 8), (0.5, 10.5)) < 0.75
    sc = M.score_episode("e", preds, refs)
    assert sc.tp == 2 and sc.fp == 0 and sc.fn == 0 and sc.f1 == 1.0


def test_threshold_applied_before_matching():
    refs = [seg(0, 10)]
    preds = [seg(0, 7)]  # IoU 0.7
    sc = M.score_episode("e", preds, refs)
    assert sc.tp == 0 and sc.fp == 1 and sc.fn == 1 and sc.f1 == 0.0
    assert sc.tp_05 == 1


def test_empty_predictions():
    sc = M.score_episode("e", [], [seg(0, 5), seg(5, 9)])
    assert (sc.tp, sc.fp, sc.fn, sc.f1) == (0, 0, 2, 0.0)


def test_duplicates_are_false_positives():
    refs = [seg(0, 5), seg(5, 10)]
    preds = [seg(0, 5), seg(0, 5.1), seg(5, 10)]
    sc = M.score_episode("e", preds, refs)
    assert (sc.tp, sc.fp, sc.fn) == (2, 1, 0)


def test_malformed_intervals_excluded_and_counted():
    preds = [seg(3, 2), {"start_sec": "a", "end_sec": 3}, seg(-1, 2), {"end_sec": 4}, seg(0, float("nan")), seg(0, 5)]
    valid, invalid = M.normalize_predictions(preds, duration=10)
    assert len(valid) == 1 and len(invalid) == 5
    sc = M.score_episode("e", preds, [seg(0, 5)], duration=10)
    assert sc.n_invalid == 5 and sc.tp == 1


def test_predictions_sorted_and_clipped():
    valid, _ = M.normalize_predictions([seg(5, 12), seg(0, 5)], duration=10)
    assert [v["start_sec"] for v in valid] == [0, 5] and valid[1]["end_sec"] == 10


def test_short_event_recall():
    refs = [seg(0, 1.5), seg(1.5, 6), seg(6, 7.8)]
    preds = [seg(0, 1.5), seg(1.5, 6), seg(6, 9)]
    sc = M.score_episode("e", preds, refs)
    assert sc.short_ref == 2 and sc.short_tp == 1


def test_micro_aggregate_and_paired():
    a = M.score_episode("a", [seg(0, 5)], [seg(0, 5), seg(5, 10)])
    b = M.score_episode("b", [seg(0, 5), seg(5, 10)], [seg(0, 5), seg(5, 10)])
    agg = M.aggregate([a, b])
    assert agg["tp"] == 3 and agg["n_ref"] == 4 and agg["n_pred"] == 3
    assert agg["micro_f1"] == pytest.approx(2 * 1.0 * 0.75 / 1.75)
    pc = M.paired_comparison({"a": 0.5, "b": 1.0}, {"a": 0.6, "b": 0.9})
    assert pc["improved"] == 1 and pc["regressed"] == 1 and pc["non_regressing_fraction"] == 0.5


def test_outer_snap_never_mutates_raw():
    preds = [seg(0.8, 5), seg(5, 9.0)]
    original = json.dumps(preds)
    snapped = M.outer_snap(preds, [seg(0, 5), seg(5, 10)])
    assert json.dumps(preds) == original
    assert snapped[0]["start_sec"] == 0 and snapped[-1]["end_sec"] == 10
    raw = M.score_episode("e", preds, [seg(0, 5), seg(5, 10)])
    sn = M.score_episode("e", preds, [seg(0, 5), seg(5, 10)], snap=True)
    assert sn.tp >= raw.tp


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    from evoaction.config import Settings

    s = Settings(artifacts_dir=tmp_path / "artifacts", cache_dir=tmp_path / "cache")
    gold = s.protected_dir / "gold"
    gold.mkdir(parents=True)
    refs = {"d1": [seg(0, 3.217, "pick"), seg(3.217, 7.443, "place")],
            "v1": [seg(0, 4.0), seg(4.0, 8.0)],
            "t1": [seg(0, 2.0), seg(2.0, 6.0)]}
    for k, v in refs.items():
        (gold / f"{k}.json").write_text(json.dumps({"episode_id": k, "segments": v}))
    splits = {"splits": {"dev": {"episode_ids": ["d1"]}, "validation": {"episode_ids": ["v1"]}, "test": {"episode_ids": ["t1"]}}}
    return ProtectedEvaluator(s, splits)


def test_test_split_sealed(evaluator):
    with pytest.raises(SealedSplitError):
        evaluator.score_split({"t1": [seg(0, 6)]}, {"t1": 6})
    with pytest.raises(SealedSplitError):
        evaluator.final_test_report({"A": {"t1": []}}, {"t1": 6})
    with pytest.raises(SealedSplitError):
        evaluator.unseal_test({"A": {"workflow_sha256": "x"}, "B": {}, "C": {}})
    evaluator.unseal_test({k: {"workflow_sha256": k} for k in "ABC"})
    rep = evaluator.final_test_report({"A": {"t1": [seg(0, 2), seg(2, 6)]}}, {"t1": 6})
    assert rep["systems"]["A"]["raw"]["micro_f1"] == 1.0


def test_views_enforce_split(evaluator):
    with pytest.raises(PermissionError):
        evaluator.dev_diagnostics({"v1": []}, {"v1": 8})
    with pytest.raises(PermissionError):
        evaluator.validation_summary({"d1": []}, {"d1": 8})


def test_dev_diagnostics_do_not_leak_reference_values(evaluator):
    diag = evaluator.dev_diagnostics({"d1": [seg(0, 5, "do it")]}, {"d1": 7.443})
    text = json.dumps(diag)
    for leaked in ("3.217", "7.443", "pick", "place"):
        assert leaked not in text
    ep = diag["episodes"]["d1"]
    assert ep["segmentation"] == "under-segmented" and ep["predictions"][0]["error_class"]


def test_validation_summary_is_aggregate_only(evaluator):
    v = evaluator.validation_summary({"v1": [seg(0, 4), seg(4, 8)]}, {"v1": 8})
    assert v["micro_f1"] == 1.0 and set(v["per_episode_f1"]) == {"v1"}
    assert "predictions" not in json.dumps(v) and "reference" not in json.dumps(v)


def test_real_episode_scores_reproducibly():
    from evoaction.config import get_settings

    s = get_settings()
    if not (s.artifacts_dir / "splits.json").exists() or not (s.protected_dir / "gold" / "robointer_droid_000010.json").exists():
        pytest.skip("real WGO data not ingested")
    ev = ProtectedEvaluator(s)
    preds = {"robointer_droid_000010": [seg(0, 6.0, "pick"), seg(6.0, 8.5, "place"), seg(8.5, 15.2, "close")]}
    a = ev.dev_diagnostics(preds, {"robointer_droid_000010": 15.2})
    b = ev.dev_diagnostics(preds, {"robointer_droid_000010": 15.2})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["episodes"]["robointer_droid_000010"]["n_ref"] == 3
