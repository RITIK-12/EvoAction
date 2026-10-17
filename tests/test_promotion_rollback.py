import json

from evoaction.config import Settings
from evoaction.experiments.evolution import HISTORY_KEYS, decide
from evoaction.persistence import Store
from evoaction.schemas import NodeInsert, WorkflowNode, WorkflowPatch
from evoaction.workflow.baselines import workflow_a
from evoaction.workflow.engine import VersionStore, apply_patch


def result(per_ep, mean, cost, statuses=None, empty=()):
    eps = {e: {"inference_cost_usd": cost, "inference_tokens": {"prompt": 100, "completion": 100}, "calls_used": 1,
               "segments": [] if e in empty else [{"start_sec": 0, "end_sec": 1, "subtask": "x"}]} for e in per_ep}
    return {"assessment": {"mean_self_score": mean, "per_episode_self_score": per_ep, "component_means": {}},
            "run": {"episodes": eps}, "statuses": statuses or {e: "ok" for e in per_ep},
            "gt_measurement": {"raw": {"micro_f1": 0.999, "per_episode_f1": {e: 0.999 for e in per_ep}}}}


def test_label_free_gate_promotes_and_ignores_gt():
    inc = result({"a": 0.30, "b": 0.40}, 0.35, 0.01)
    cand = result({"a": 0.40, "b": 0.42}, 0.41, 0.012)
    cand["gt_measurement"]["raw"]["micro_f1"] = 0.0  # GT says worse; the label-free gate must not look at it
    d = decide(cand, inc)
    assert d["decision"] == "PROMOTED"
    assert "0.999" not in json.dumps(d) and "micro_f1" not in json.dumps(d)


def test_label_free_gate_rejections():
    inc = result({"a": 0.30, "b": 0.40}, 0.35, 0.01)
    assert decide(result({"a": 0.31, "b": 0.41}, 0.36, 0.01), inc)["decision"] == "REJECTED"  # delta < 0.03
    d = decide(result({"a": 0.6, "b": 0.6}, 0.6, 0.02), inc)
    assert d["decision"] == "REJECTED" and not d["checks"]["annotation_cost_ratio"]["pass"]
    d = decide(result({"a": 0.6, "b": 0.6}, 0.6, 0.01, statuses={"a": "model_error", "b": "ok"}), inc)
    assert d["decision"] == "REJECTED" and not d["checks"]["valid_outputs"]["pass"]
    d = decide(result({"a": 0.6, "b": 0.6}, 0.6, 0.01, empty=("a",)), inc)
    assert not d["checks"]["valid_outputs"]["pass"]
    d = decide(result({"a": 0.9, "b": 0.2}, 0.55, 0.01), inc)
    assert d["checks"]["non_regressing_fraction"]["value"] == 0.5


def test_history_keys_exclude_shadow_gt():
    assert "shadow_gt" not in HISTORY_KEYS and "gt_measurement" not in HISTORY_KEYS


def test_rejection_preserves_candidate_and_incumbent(tmp_path):
    s = Settings(artifacts_dir=tmp_path / "artifacts", cache_dir=tmp_path / "cache")
    store = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")
    vs = VersionStore(s, store)
    wf, prompts = workflow_a()
    inc = vs.create("N000", wf, prompts, {"models": {}}, None, "incumbent")
    patch = WorkflowPatch(insert_nodes=[NodeInsert(node=WorkflowNode(id="merge", operator="merge_filter_segments", depends_on=["annotate"]),
                                                   rewire_dependents_of="annotate")])
    cwf, cprompts = apply_patch(inc.workflow, inc.prompts, patch)
    vs.create("EXP-0001", cwf, cprompts, {"models": {}}, "N000", "candidate", experiment={"hypothesis": "h"})
    rej = vs.finalize("EXP-0001", "REJECTED", {"decision": "REJECTED"}, new_id_="EXP-0001_rejected")
    assert rej.manifest["frozen"] and rej.manifest["status"] == "REJECTED"
    again = vs.load("N000")
    assert again.workflow_sha256 == inc.workflow_sha256


def test_promotion_creates_new_incumbent_copy(tmp_path):
    s = Settings(artifacts_dir=tmp_path / "artifacts", cache_dir=tmp_path / "cache")
    store = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")
    vs = VersionStore(s, store)
    wf, prompts = workflow_a()
    vs.create("N000", wf, prompts, {"models": {}}, None, "incumbent")
    cwf, cprompts = apply_patch(wf, prompts, WorkflowPatch(set_params={"sample": {"sampling_hz": 2.0}}))
    vs.create("EXP-0002", cwf, cprompts, {"models": {}}, "N000", "candidate")
    vs.finalize("EXP-0002", "PROMOTED", {"decision": "PROMOTED"}, new_id_="EXP-0002_promoted")
    n1 = vs.promote_copy("EXP-0002_promoted", "N001")
    assert n1.manifest["source_experiment"] == "EXP-0002_promoted"
