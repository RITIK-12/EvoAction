import json

import pytest
from pydantic import ValidationError

from evoaction.config import Settings
from evoaction.persistence import Store
from evoaction.schemas import HarnessWorkflow, NodeInsert, WorkflowNode, WorkflowPatch
from evoaction.workflow.baselines import workflow_n0
from evoaction.workflow.engine import VersionStore, WorkflowValidationError, apply_patch, validate_workflow, version_hash
from evoaction.workflow.operators import EpisodeContext, detect_hints, op_merge_filter_segments
from evoaction.schemas import HintParams


def wf_dict(nodes, max_calls=2):
    return {"nodes": nodes, "max_calls": max_calls, "max_wall_seconds": 120}


def node(id, op, deps=(), **params):
    return {"id": id, "operator": op, "depends_on": list(deps), "params": params}


P = {"prompt.md": "x", "refine.md": "y"}


def test_baselines_validate():
    for f in (workflow_n0,):
        wf, prompts = f()
        validate_workflow(wf, prompts)


def test_unknown_operator_rejected():
    with pytest.raises(WorkflowValidationError):
        validate_workflow(wf_dict([node("s", "sample_contact_sheets"), node("g", "read_gold", ["s"]), node("e", "validate_export", ["g"])]), P)


def test_cycle_rejected():
    nodes = [node("s", "sample_contact_sheets", ["e"]), node("a", "initial_annotate", ["s"]), node("e", "validate_export", ["a"])]
    with pytest.raises(WorkflowValidationError, match="cycle"):
        validate_workflow(wf_dict(nodes), P)


def test_node_limit():
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"])]
    prev = "a"
    for i in range(6):
        nodes.append(node(f"m{i}", "merge_filter_segments", [prev]))
        prev = f"m{i}"
    nodes.append(node("e", "validate_export", [prev]))
    with pytest.raises(WorkflowValidationError, match="too many nodes"):
        validate_workflow(wf_dict(nodes), P)


def test_type_requirements():
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"]),
             node("r", "refine_intervals", ["a"]), node("e", "validate_export", ["r"])]
    with pytest.raises(WorkflowValidationError, match="evidence"):
        validate_workflow(wf_dict(nodes), P)


def test_hint_selection_requires_hint_node():
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"]),
             node("i", "inspect_transition_windows", ["a"], select="near_hints"),
             node("r", "refine_intervals", ["i"]), node("e", "validate_export", ["r"])]
    with pytest.raises(WorkflowValidationError, match="hints"):
        validate_workflow(wf_dict(nodes), P)
    nodes.insert(2, node("h", "state_hint_detection"))
    nodes[3]["depends_on"] = ["a", "h"]
    validate_workflow(wf_dict(nodes), P)


def test_model_pass_limit():
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"]),
             node("i", "inspect_transition_windows", ["a"]), node("r", "refine_intervals", ["i"]),
             node("i2", "inspect_transition_windows", ["r"]), node("r2", "refine_intervals", ["i2"]),
             node("e", "validate_export", ["r2"])]
    with pytest.raises(WorkflowValidationError, match="multimodal passes"):
        validate_workflow(wf_dict(nodes, max_calls=3), P)


def test_unknown_param_and_bounds_rejected():
    with pytest.raises(WorkflowValidationError, match="params"):
        validate_workflow(wf_dict([node("s", "sample_contact_sheets", sampling_hz=50), node("a", "initial_annotate", ["s"]),
                                   node("e", "validate_export", ["a"])]), P)
    with pytest.raises(WorkflowValidationError, match="params"):
        validate_workflow(wf_dict([node("s", "sample_contact_sheets", shell="rm -rf /"), node("a", "initial_annotate", ["s"]),
                                   node("e", "validate_export", ["a"])]), P)


def test_export_must_be_terminal_and_reachable():
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"]), node("e", "validate_export", ["a"]),
             node("m", "merge_filter_segments", ["e"])]
    with pytest.raises(WorkflowValidationError):
        validate_workflow(wf_dict(nodes), P)
    nodes = [node("s", "sample_contact_sheets"), node("a", "initial_annotate", ["s"]), node("h", "state_hint_detection"),
             node("e", "validate_export", ["a"])]
    with pytest.raises(WorkflowValidationError, match="not feeding"):
        validate_workflow(wf_dict(nodes), P)


def test_topology_patch_inserts_inspect_and_refine():
    wf, prompts = workflow_n0()
    patch = WorkflowPatch(insert_nodes=[
        NodeInsert(node=WorkflowNode(id="hints", operator="state_hint_detection", when="has_telemetry")),
        NodeInsert(node=WorkflowNode(id="inspect", operator="inspect_transition_windows", depends_on=["annotate", "hints"],
                                     params={"select": "near_hints", "include_unmatched_hints": True}),
                   rewire_dependents_of="annotate"),
        NodeInsert(node=WorkflowNode(id="refine", operator="refine_intervals", depends_on=["annotate", "inspect"]),
                   rewire_dependents_of="inspect"),
    ])
    assert patch.is_topology_change()
    new_wf, new_prompts = apply_patch(wf, prompts, patch)
    ops = [n.operator for n in new_wf.nodes]
    assert ops[-1] == "validate_export" and "refine_intervals" in ops and "refine.md" in new_prompts
    export = [n for n in new_wf.nodes if n.operator == "validate_export"][0]
    assert export.depends_on == ["refine"]
    assert new_wf.max_calls >= 2
    # original untouched
    assert [n.operator for n in wf.nodes] == ["sample_contact_sheets", "initial_annotate", "validate_export"]


def test_prompt_only_patch_and_bad_prompt_name():
    wf, prompts = workflow_n0()
    new_wf, new_prompts = apply_patch(wf, prompts, WorkflowPatch(prompt_updates={"prompt.md": "new $instruction"}))
    assert new_prompts["prompt.md"] == "new $instruction" and not WorkflowPatch(prompt_updates={"prompt.md": "z"}).is_topology_change()
    with pytest.raises(ValidationError):
        WorkflowPatch(prompt_updates={"../evaluation/segments.py": "pwned"})


def test_patch_cannot_remove_into_invalid_graph():
    wf, prompts = workflow_n0()
    with pytest.raises(WorkflowValidationError):
        apply_patch(wf, prompts, WorkflowPatch(remove_nodes=["annotate"]))


def test_version_store_immutability(tmp_path):
    s = Settings(artifacts_dir=tmp_path / "artifacts", cache_dir=tmp_path / "cache")
    store = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")
    vs = VersionStore(s, store)
    wf, prompts = workflow_n0()
    v = vs.create("N000", wf, prompts, {"models": {"vision": "m"}}, None, "incumbent")
    assert v.workflow_sha256 == version_hash(wf, prompts, {"models": {"vision": "m"}})
    assert vs.create("N000", wf, prompts, {"models": {"vision": "m"}}, None, "incumbent").workflow_sha256 == v.workflow_sha256
    vs.finalize("N000", "baseline", {"ok": True})
    with pytest.raises(PermissionError):
        vs.finalize("N000", "x", {})
    (v.path / "prompt.md").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        vs.load("N000")


def test_merge_filter_tiles_and_fixes_overlaps(tmp_path):
    ctx = EpisodeContext(episode={"id": "e", "duration_sec": 10.0}, prompts={}, client=None, version_id="v", run_id=None,
                         out_dir=tmp_path, max_calls=1)
    state = {"episode": ctx.episode, "segments": [
        {"start_sec": 0.5, "end_sec": 4.0, "subtask": "a"}, {"start_sec": 3.5, "end_sec": 6.0, "subtask": "b"},
        {"start_sec": 6.1, "end_sec": 6.3, "subtask": "tiny"}, {"start_sec": 6.8, "end_sec": 9.0, "subtask": "c"}]}
    out, _ = op_merge_filter_segments(ctx, state, {"min_duration_sec": 0.4, "max_gap_sec": 1.0, "tile_full_episode": True})
    segs = out["segments"]
    assert segs[0]["start_sec"] == 0.0 and segs[-1]["end_sec"] == 10.0
    assert all(a["end_sec"] == b["start_sec"] for a, b in zip(segs, segs[1:]))
    assert all(s["end_sec"] - s["start_sec"] >= 0.4 for s in segs)


def test_gripper_hints_from_telemetry(tmp_path):
    ts = [i / 10 for i in range(100)]
    grip = [1.0 if t < 3 or t > 7 else 0.0 for t in ts]
    tel = {"timestamp_sec": ts, "channels": [{"name": "state", "value_names": ["x", "gripper_state"],
                                              "values": [[0.0, g] for g in grip]}]}
    p = tmp_path / "telemetry.json"
    p.write_text(json.dumps(tel))
    hints = detect_hints(str(p), HintParams(signals=["gripper"]))
    times = [h["t"] for h in hints]
    assert len(times) == 2 and abs(times[0] - 3.0) < 0.4 and abs(times[1] - 7.0) < 0.4


def test_flat_insert_node_is_normalized():
    wf, prompts = workflow_n0()
    patch = WorkflowPatch(insert_nodes=[{"id": "merge", "operator": "merge_filter_segments", "depends_on": ["annotate"],
                                         "params": {"min_duration_sec": 2.0}, "rewire_dependents_of": "annotate"}])
    new_wf, _ = apply_patch(wf, prompts, patch)
    assert [n.operator for n in new_wf.nodes][-2:] == ["merge_filter_segments", "validate_export"]


def test_operator_name_references_resolve_to_node_ids():
    wf, prompts = workflow_n0()
    patch = WorkflowPatch(insert_nodes=[{"id": "merge", "operator": "merge_filter_segments", "depends_on": ["initial_annotate"]}],
                          set_depends_on={"validate_export": ["merge_filter_segments"]}, set_params={"initial_annotate": {"max_tokens": 4096}})
    new_wf, _ = apply_patch(wf, prompts, patch)
    by_id = {n.id: n for n in new_wf.nodes}
    assert by_id["export"].depends_on == ["merge"] and by_id["merge"].depends_on == ["annotate"]
    assert by_id["annotate"].params["max_tokens"] == 4096


class _StubClient:
    """Fails the first non-thinking call with a truncated reply, then answers; records the temperatures it saw."""
    vision_model = "stub"

    def __init__(self, raw_text):
        self.raw_text, self.temps = raw_text, []

    def chat(self, role, messages, **kw):
        from evoaction.clients.crusoe import ChatResult, ModelCallError
        self.temps.append(kw["temperature"])
        if len(self.temps) == 1:
            e = ModelCallError("initial_annotate: failed after retries: unterminated JSON in model output")
            e.raw_text = self.raw_text
            raise e
        return ChatResult(text="{}", data={"segments": [{"start_sec": 0.0, "end_sec": 3.0, "subtask": "x"}]})


def test_runaway_reply_is_salvaged_or_retried(tmp_path):
    from evoaction.schemas import AnnotateParams
    from evoaction.workflow.operators import _vision_json
    p = AnnotateParams(thinking=False)
    ctx = EpisodeContext(episode={"id": "e", "duration_sec": 10.0}, prompts={}, client=None, version_id="v", run_id=None, out_dir=tmp_path, max_calls=1)
    ctx.client = _StubClient('{"segments":[{"start_sec":0,"end_sec":2,"subtask":"a"}' + ',{"start_sec":2,"end_sec":3,"subtask":"b"}' * 50 + ',{"st')
    res, how = _vision_json(ctx, [], "initial_annotate", p, {}, "e")
    assert how == "salvaged_truncated" and [s["end_sec"] for s in res.data["segments"]] == [2, 3]
    ctx.client = _StubClient('{"segments":[{"st')  # nothing complete to salvage -> one retry at higher temperature
    res, how = _vision_json(ctx, [], "initial_annotate", p, {}, "e")
    assert how == "retried_t0.6" and ctx.client.temps == [p.temperature, 0.6] and res.data["segments"]
