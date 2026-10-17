"""Label-free signals and no-ground-truth guarantees for the N+1 loop."""

import json
import re
from pathlib import Path

import pytest

from evoaction.clients.crusoe import ChatResult
from evoaction.selfcheck import label_free as LF

ROOT = Path(__file__).resolve().parent.parent
SENTINEL_F1 = 0.731337
SENTINEL_LABEL = "SENTINEL_GOLD_LABEL_xyz"
SENTINEL_TIME = 7.4321


def seg(s, e, label="x"):
    return {"start_sec": s, "end_sec": e, "subtask": label}


def test_agent_and_selfcheck_code_never_touch_ground_truth():
    banned = re.compile(r"protected_dir|ProtectedEvaluator|evaluation\.protected|gold|gt_measurement|shadow_gt")
    for path in list((ROOT / "evoaction" / "agents").glob("*.py")) + list((ROOT / "evoaction" / "selfcheck").glob("*.py")):
        text = path.read_text()
        hits = [m.group(0) for m in banned.finditer(text)]
        assert not hits, f"{path.name} references ground truth: {hits}"


def test_telemetry_agreement():
    hints = [{"t": 3.0, "kind": "gripper_closed"}, {"t": 7.0, "kind": "gripper_opened"}, {"t": 5.0, "kind": "eef_pause"}]
    r = LF.telemetry_agreement([3.2, 5.0, 7.5], hints)
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-3) and r["recall"] == 1.0
    assert LF.telemetry_agreement([1.0], [{"t": 2, "kind": "eef_pause"}]) is None


def test_duration_variability_flags_uniform_grids():
    uniform = [seg(i, i + 1) for i in range(10)]
    varied = [seg(0, 1), seg(1, 5), seg(5, 6.5), seg(6.5, 12)]
    assert LF.duration_variability(uniform)["score"] == 0.0
    assert LF.duration_variability(varied)["score"] > 0.9


def test_temporal_resolution():
    r = LF.temporal_resolution([seg(0, 1), seg(1, 4), seg(4, 4.5)], sampling_hz=1.0)
    assert r["sub_resolution_segments"] == 2 and r["score"] == pytest.approx(1 / 3, abs=1e-3)


def test_score_renormalizes_over_available_components(monkeypatch):
    monkeypatch.setattr(LF, "load_episode", lambda e: {"id": e, "duration_sec": 12.0, "telemetry_path": None, "instruction": "x", "video_path": ""})
    a = LF.assess_episode(None, "e1", [seg(0, 1), seg(1, 5), seg(5, 6.5), seg(6.5, 12)], 1.0, "v", verify=False)
    w = LF.SCORE_DEFINITION["weights"]
    vals = a["component_values"]
    assert vals["telemetry_agreement"] is None and vals["visual_verification"] is None
    expected = (w["duration_variability"] * vals["duration_variability"] + w["temporal_resolution"] * vals["temporal_resolution"]) / (
        w["duration_variability"] + w["temporal_resolution"])
    assert a["self_score"] == pytest.approx(expected, abs=1e-4)
    assert LF.assess_episode(None, "e1", [], 1.0, "v", verify=False)["self_score"] == 0.0


class CapturingClient:
    nemotron_model = "fake-nemotron"
    vision_model = "fake-vision"

    def __init__(self):
        self.prompts = []

    def chat(self, role, messages, **kw):
        self.prompts.append(json.dumps(messages, default=str))
        purpose = kw.get("purpose", "")
        if purpose == "supervisor_plan":
            data = {"failure_summary": "evenly spaced segments", "investigations": [
                {"question": "Are boundaries aligned with gripper events?", "tools_allowed": ["self_assessment"],
                 "tool_requests": [{"tool": "self_assessment", "args": {}}]}]}
        elif purpose == "investigate":
            ids = re.findall(r"(?:EV|SELF)-[0-9a-z]+", messages[-1]["content"])
            data = {"status": "answered", "finding": "misaligned", "evidence_ids": ids[:1], "new_questions": []}
        elif purpose == "critic_review":
            data = {"verdict": "approve", "issues": []}
        else:
            data = {"hypothesis": "h", "failure_slice": "all", "evidence_ids": ["SELF-abc"], "objective": "o",
                    "candidate_patch": {"set_params": {"sample": {"sampling_hz": 2.0}}}, "expected_outcome": "e",
                    "verification_plan": {}, "stop_conditions": []}
        return ChatResult(text="{}", data=data, model="fake", call_id=f"CALL-{len(self.prompts)}")


def test_no_ground_truth_reaches_any_agent_prompt(tmp_path, monkeypatch):
    from evoaction.agents import investigator, supervisor
    from evoaction.agents.tools import ToolContext
    from evoaction.experiments.evolution import HISTORY_KEYS
    from evoaction.persistence import Store
    from evoaction.workflow.baselines import workflow_a
    from evoaction.workflow.engine import Version

    monkeypatch.setattr("evoaction.data.wgo.load_episode",
                        lambda e: {"id": e, "has_telemetry": True, "duration_sec": 12.0, "source": "droid", "instruction": "x"})
    wf, prompts = workflow_a()
    inc = Version("N000", tmp_path, wf, prompts, {"models": {}}, {"workflow_sha256": "h"})
    gt = {"raw": {"micro_f1": SENTINEL_F1, "per_episode_f1": {"d1": SENTINEL_F1}}, "reference": [seg(0, SENTINEL_TIME, SENTINEL_LABEL)]}
    assessment = {"mean_self_score": 0.4, "per_episode_self_score": {"d1": 0.4}, "component_means": {},
                  "episodes": {"d1": {"self_score": 0.4, "component_values": {}, "n_segments": 2, "predicted_boundaries": [5.0],
                                      "gripper_events": [3.0], "components": {"duration_variability": {"cv": 0.1},
                                                                              "temporal_resolution": {"sub_resolution_segments": 0}}}}}
    inc_eval = {"eval_id": "SELF-abc", "assessment": assessment, "gt_measurement": gt,
                "run": {"episodes": {"d1": {"segments": [seg(0, 5), seg(5, 12)], "duration_sec": 12.0}}}}
    client = CapturingClient()
    store = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")
    ctx = ToolContext(client=client, store=store, incumbent=inc, dev_ids=["d1"], run_id="RUN-t", incumbent_eval=inc_eval)
    ctx.evidence["SELF-abc"] = {"id": "SELF-abc", "tool": "self_assessment", "args": {}, "result": assessment, "task_id": None}
    lineage_exps = [{"id": "EXP-0001", "parent_version": "N000", "hypothesis": "h0", "patch_summary": {}, "decision": "REJECTED",
                     "label_free": {"candidate_self_score": 0.3}, "shadow_gt": {"candidate": {"micro_f1": SENTINEL_F1}}}]
    history = [{k: e.get(k) for k in HISTORY_KEYS} for e in lineage_exps]
    plan = supervisor.plan_investigations(client, ctx, inc, {"mean_self_score": 0.41, "component_means": {}}, history, {"x": 1})
    graph = investigator.TaskGraph(store, "RUN-t", max_total=12)
    roots = [graph.add(i["question"], None, i["tools_allowed"], i["tool_requests"], evidence_ids=["SELF-abc"])[0] for i in plan["investigations"]]
    investigator.dispatch(graph, roots, ctx, "wf")
    supervisor.propose_experiment(client, store, "RUN-t", "EXP-0002", [{"finding": "f"}], inc, history, {"x": 1}, set(ctx.evidence))
    assert len(client.prompts) >= 4
    blob = "\n".join(client.prompts)
    for sentinel in (str(SENTINEL_F1), SENTINEL_LABEL, str(SENTINEL_TIME), "micro_f1", "shadow_gt"):
        assert sentinel not in blob, f"ground-truth sentinel {sentinel!r} leaked into an agent prompt"
