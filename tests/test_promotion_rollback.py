import json

from evoaction.config import Settings
from evoaction.experiments.evolution import decide
from evoaction.persistence import Store
from evoaction.schemas import NodeInsert, WorkflowNode, WorkflowPatch
from evoaction.workflow.baselines import workflow_a
from evoaction.workflow.engine import VersionStore, apply_patch


def result(per_ep, micro, cost, statuses=None, invalid=0, empty=0, calls=1):
    eps = {e: {"inference_cost_usd": cost, "inference_tokens": {"prompt": 100, "completion": 100}, "calls_used": calls} for e in per_ep}
    return {"agent_view": {"micro_f1": micro, "per_episode_f1": per_ep, "invalid_items": invalid, "empty_output_episodes": empty,
                           "micro_f1_iou05": micro, "short_event_recall": None, "mean_boundary_mae_sec": 1.0, "outer_snapped_micro_f1": micro},
            "run": {"episodes": eps}, "statuses": statuses or {e: "ok" for e in per_ep}}


def test_promotes_when_all_checks_pass():
    inc = result({"a": 0.2, "b": 0.3}, 0.25, 0.01)
    cand = result({"a": 0.3, "b": 0.3}, 0.30, 0.012)
    d = decide(cand, inc)
    assert d["decision"] == "PROMOTED"


def test_rejects_small_delta_regressions_cost_and_invalid():
    inc = result({"a": 0.2, "b": 0.3}, 0.25, 0.01)
    assert decide(result({"a": 0.21, "b": 0.3}, 0.255, 0.01), inc)["decision"] == "REJECTED"  # delta < 0.02
    assert decide(result({"a": 0.6, "b": 0.1}, 0.35, 0.01, ), result({"a": 0.2, "b": 0.3, }, 0.25, 0.01))["checks"]["non_regressing_fraction"]["pass"]
    d = decide(result({"a": 0.5, "b": 0.5}, 0.5, 0.02), inc)  # 2x cost
    assert d["decision"] == "REJECTED" and not d["checks"]["annotation_cost_ratio"]["pass"]
    d = decide(result({"a": 0.5, "b": 0.5}, 0.5, 0.01, statuses={"a": "model_error", "b": "ok"}), inc)
    assert d["decision"] == "REJECTED" and not d["checks"]["valid_outputs"]["pass"]
    d = decide(result({"a": 0.9, "b": 0.0}, 0.45, 0.01), result({"a": 0.2, "b": 0.3}, 0.25, 0.01))
    assert d["checks"]["non_regressing_fraction"]["value"] == 0.5


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
    assert (s.artifacts_dir / "versions" / "EXP-0001_rejected" / "experiment.json").exists()
    again = vs.load("N000")
    assert again.workflow_sha256 == inc.workflow_sha256 and [n.operator for n in again.workflow.nodes] == [n.operator for n in wf.nodes]


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
    assert json.loads((n1.path / "workflow.json").read_text())["nodes"][0]["params"]["sampling_hz"] == 2.0
