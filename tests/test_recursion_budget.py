"""Recursion mechanics with a scripted fake model (unit tests only; never a real-run branch)."""

import re

import pytest

from evoaction.agents.investigator import TaskGraph, dispatch, similar, unfinished_tasks
from evoaction.agents.tools import ToolContext
from evoaction.clients.crusoe import ChatResult, ModelCallError
from evoaction.persistence import Store


class FakeClient:
    nemotron_model = "fake-nemotron"
    vision_model = "fake-vision"

    def __init__(self, script):
        self.script = script  # fn(prompt) -> dict | Exception
        self.calls = 0

    def chat(self, role, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        out = self.script(prompt)
        if isinstance(out, Exception):
            raise out
        return ChatResult(text="{}", data=out, model=self.nemotron_model, call_id=f"CALL-{self.calls}")


class FakeVersion:
    id = "N000"
    prompts = {}

    class workflow:
        nodes = []


def make_ctx(tmp_path, client):
    store = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")
    ctx = ToolContext(client=client, store=store, incumbent=FakeVersion(), dev_ids=["d1"], run_id="RUN-test",
                      incumbent_eval={"assessment": {"mean_self_score": 0.4, "episodes": {}}, "run": {"episodes": {}}, "eval_id": "SELF-x"})
    return ctx, store


def ev_ids(prompt):
    return re.findall(r"EV-[0-9a-f]{10}", prompt)


def test_similarity():
    assert similar("Why are short events missed?", "why are short events missed")
    assert not similar("Why are short events missed?", "Does gripper telemetry align with boundaries?")


def test_child_and_grandchild_spawn_via_gates(tmp_path):
    def script(prompt):
        ids = ev_ids(prompt)
        if "depth 1" in prompt:
            return {"status": "answered", "finding": "coarse sampling", "evidence_ids": ids[:1],
                    "new_questions": [{"question": "Do sampled frames straddle the transitions?", "evidence_ids": ids[:1],
                                       "tool_requests": [{"tool": "self_assessment", "args": {}}]}]}
        if "depth 2" in prompt:
            return {"status": "answered", "finding": "yes they straddle", "evidence_ids": ids[:1],
                    "new_questions": [{"question": "Would local 4 Hz windows resolve the straddled transitions?", "evidence_ids": ids[:1],
                                       "tool_requests": [{"tool": "self_assessment", "args": {}}]}]}
        return {"status": "answered", "finding": "probably", "evidence_ids": ids[:1], "new_questions": []}

    client = FakeClient(script)
    ctx, store = make_ctx(tmp_path, client)
    g = TaskGraph(store, "RUN-test", max_total=12)
    root, verdict = g.add("Why is short-event recall low?", None, ["self_assessment"], [{"tool": "self_assessment", "args": {}}])
    assert verdict == "accepted"
    dispatch(g, [root], ctx, "wf")
    depths = sorted(t.depth for t in g.tasks.values() if t.status == "done")
    assert depths == [1, 2, 3]
    assert all(t.parent_task_id for t in g.tasks.values() if t.depth > 1)


def test_dedup_cycle_depth_fanout_and_evidence_gates(tmp_path):
    ctx, store = make_ctx(tmp_path, FakeClient(lambda p: {}))
    g = TaskGraph(store, "RUN-test", max_total=12, max_depth=3, max_children=2)
    root, _ = g.add("Why is recall low?", None, [], [])
    root.status = "done"
    _, v = g.add("why is recall low", root, [], [], evidence_ids=["EV-1"])
    assert v.startswith("rejected: cycle")
    c1, v1 = g.add("Are boundaries late?", root, [], [], evidence_ids=["EV-1"])
    _, vdup = g.add("are boundaries late", root, [], [], evidence_ids=["EV-1"])
    assert v1 == "accepted" and vdup.startswith("deduplicated")
    _, vnoev = g.add("Is the prompt ambiguous?", root, [], [], evidence_ids=[])
    assert "cites no evidence" in vnoev
    c2, v2 = g.add("Is sampling too coarse?", root, [], [], evidence_ids=["EV-1"])
    _, v3 = g.add("Is the model hallucinating labels?", root, [], [], evidence_ids=["EV-1"])
    assert v2 == "accepted" and "children" in v3
    gc, _ = g.add("Do frames straddle transitions?", c1, [], [], evidence_ids=["EV-1"])
    ggc, vd = g.add("Would 4 Hz windows fix it?", gc, [], [], evidence_ids=["EV-1"])
    assert gc.depth == 3 and "depth" in vd


def test_total_task_budget(tmp_path):
    ctx, store = make_ctx(tmp_path, FakeClient(lambda p: {}))
    g = TaskGraph(store, "RUN-test", max_total=2)
    a, _ = g.add("q one about sampling", None, [], [])
    b, _ = g.add("q two about telemetry", None, [], [])
    c, v = g.add("q three about prompts", None, [], [])
    assert "budget" in v and c.status == "rejected"


def test_failed_model_call_retries_then_fails_not_success(tmp_path):
    client = FakeClient(lambda p: ModelCallError("invalid JSON twice"))
    ctx, store = make_ctx(tmp_path, client)
    g = TaskGraph(store, "RUN-test", max_total=12)
    root, _ = g.add("Why is recall low?", None, [], [])
    dispatch(g, [root], ctx, "wf")
    assert root.status == "failed" and client.calls == 2
    kinds = [r["kind"] for r in store.query("SELECT kind FROM events")]
    assert "task_retry" in kinds and "task_failed" in kinds


def test_failed_tool_output_is_reported_as_error_evidence(tmp_path):
    seen = {}

    def script(prompt):
        seen["prompt"] = prompt
        return {"status": "answered", "finding": "tool failed, cannot conclude", "evidence_ids": ev_ids(prompt)[:1], "new_questions": []}

    ctx, store = make_ctx(tmp_path, FakeClient(script))
    g = TaskGraph(store, "RUN-test", max_total=12)
    root, _ = g.add("Inspect frames of a non-dev episode", None, ["episode_predictions"],
                    [{"tool": "episode_predictions", "args": {"episode_id": "test_episode"}}])
    dispatch(g, [root], ctx, "wf")
    ev = next(iter(ctx.evidence.values()))
    assert "error" in ev["result"] and "not an accessible dev episode" in ev["result"]["error"]
    assert "error" in seen["prompt"]


def test_persistence_identifies_unfinished_after_restart(tmp_path):
    ctx, store = make_ctx(tmp_path, FakeClient(lambda p: {}))
    g = TaskGraph(store, "RUN-test", max_total=12)
    a, _ = g.add("finished question about sampling", None, [], [])
    b, _ = g.add("pending question about telemetry", None, [], [])
    g.set_status(a, "done", result={"finding": "x"})
    store2 = Store(tmp_path / "db.sqlite", tmp_path / "events.jsonl")  # simulated restart
    st = unfinished_tasks(store2, "RUN-test")
    assert a.id in st["finished"] and b.id in st["unfinished"]


def test_large_profile_cannot_hide_later_evidence():
    from evoaction.agents.tools import compact, evidence_item

    huge = {"episodes": {"d1": {"n_pred": 30, "predictions": [{"error_class": "spurious or badly placed", "start_error": ">2s late",
                                                              "pad": "x" * 200} for _ in range(300)]}}}
    view = [evidence_item("EV-1", "dev_failure_profile", {}, huge), evidence_item("EV-2", "telemetry_hints", {}, {"hints": [{"t": 3.1}]})]
    text = compact(view, 24000)
    assert "EV-2" in text and '\\"t\\": 3.1' in text
