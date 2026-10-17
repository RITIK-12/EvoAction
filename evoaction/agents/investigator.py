"""Dynamic recursive investigation: task graph with bounded delegation + Nemotron investigators.

Tasks are created only from (a) the supervisor's plan or (b) `new_questions` an
investigator raises from evidence it actually gathered. The dispatcher applies
deterministic gates (depth, fan-out, total budget, duplicate/cycle detection,
evidence grounding) before a child is spawned. Nothing here pre-scripts a chain.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import deque
from typing import Any

from evoaction.agents.tools import TOOL_DOCS, ToolContext, compact, evidence_item, run_tool
from evoaction.clients.crusoe import ROLE_NEMOTRON, BudgetExceeded, ModelCallError
from evoaction.persistence import Store, new_id
from evoaction.schemas import InvestigationTask

MAX_DEPTH = 3
MAX_CHILDREN = 2
MAX_TOOL_ROUNDS = 2
MAX_TOOLS_PER_ROUND = 3


def _norm(q: str) -> str:
    q = re.sub(r"[^a-z0-9 ]", " ", q.lower())
    return re.sub(r"\s+", " ", q).strip()


def similar(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    return na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85 or jacc >= 0.75


class TaskGraph:
    def __init__(self, store: Store, run_id: str, max_total: int, max_depth: int = MAX_DEPTH, max_children: int = MAX_CHILDREN):
        self.store = store
        self.run_id = run_id
        self.max_total = max_total
        self.max_depth = max_depth
        self.max_children = max_children
        self.tasks: dict[str, InvestigationTask] = {}
        self.decisions: list[dict[str, Any]] = []

    # persistence ---------------------------------------------------------
    def _save(self, t: InvestigationTask) -> None:
        self.store.insert("investigation_tasks", {
            "id": t.id, "run_id": self.run_id, "parent_task_id": t.parent_task_id, "depth": t.depth, "question": t.question,
            "evidence_ids": t.evidence_ids, "tools_allowed": t.tools_allowed, "budget": t.budget, "status": t.status,
            "result": t.result, "new_questions": t.new_questions, "model_used": t.model_used,
            "created_at": self.store.query("SELECT created_at FROM investigation_tasks WHERE id = ?", (t.id,))[0]["created_at"]
            if self.store.query("SELECT 1 FROM investigation_tasks WHERE id = ?", (t.id,)) else time.time(),
            "updated_at": time.time()})

    def set_status(self, t: InvestigationTask, status: str, **fields: Any) -> None:
        t.status = status  # type: ignore[assignment]
        for k, v in fields.items():
            setattr(t, k, v)
        self._save(t)
        self.store.event(self.run_id, f"task_{status}", {"task_id": t.id, "depth": t.depth, "parent_task_id": t.parent_task_id,
                                                         "question": t.question})

    def ancestors(self, t: InvestigationTask) -> list[InvestigationTask]:
        out = []
        cur = self.tasks.get(t.parent_task_id) if t.parent_task_id else None
        while cur:
            out.append(cur)
            cur = self.tasks.get(cur.parent_task_id) if cur.parent_task_id else None
        return out

    def counted(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status not in ("deduplicated", "rejected"))

    # creation with gates ---------------------------------------------------
    def add(self, question: str, parent: InvestigationTask | None, tools_allowed: list[str], tool_requests: list[dict[str, Any]],
            evidence_ids: list[str] | None = None, why: str = "") -> tuple[InvestigationTask, str]:
        depth = (parent.depth + 1) if parent else 1
        t = InvestigationTask(id=new_id("TASK"), parent_task_id=parent.id if parent else None, depth=depth, question=question.strip()[:600],
                              evidence_ids=list(evidence_ids or []), tools_allowed=[x for x in tools_allowed if x in TOOL_DOCS] or list(TOOL_DOCS),
                              budget={"tool_requests": tool_requests[:MAX_TOOLS_PER_ROUND], "max_tool_rounds": MAX_TOOL_ROUNDS, "why": why[:300]},
                              run_id=self.run_id)
        verdict = "accepted"
        if parent is not None:
            ancestor_qs = [a.question for a in self.ancestors(t)]
            if any(similar(question, q) for q in ancestor_qs):
                verdict = "rejected: cycle (repeats an ancestor question)"
            elif any(similar(question, x.question) for x in self.tasks.values()):
                verdict = "deduplicated: equivalent question already investigated"
            elif depth > self.max_depth:
                verdict = f"rejected: depth {depth} > max_depth {self.max_depth}"
            elif sum(1 for x in self.tasks.values() if x.parent_task_id == parent.id and x.status not in ("deduplicated", "rejected")) >= self.max_children:
                verdict = f"rejected: parent already has {self.max_children} children"
            elif not t.evidence_ids:
                verdict = "rejected: new question cites no evidence from the parent investigation"
        elif any(similar(question, x.question) for x in self.tasks.values()):
            verdict = "deduplicated: equivalent question already investigated"
        if verdict == "accepted" and self.counted() >= self.max_total:
            verdict = f"rejected: total task budget {self.max_total} exhausted"
        self.tasks[t.id] = t
        status = "pending" if verdict == "accepted" else ("deduplicated" if verdict.startswith("deduplicated") else "rejected")
        t.status = status  # type: ignore[assignment]
        if status != "pending":
            t.result = {"dispatch_decision": verdict}
        self._save(t)
        self.decisions.append({"task_id": t.id, "parent_task_id": t.parent_task_id, "depth": depth, "question": t.question, "decision": verdict})
        self.store.event(self.run_id, "task_dispatch_decision", {"task_id": t.id, "parent_task_id": t.parent_task_id, "depth": depth,
                                                                 "question": t.question, "decision": verdict})
        return t, verdict


def unfinished_tasks(store: Store, run_id: str) -> dict[str, list[str]]:
    """Restart support: identify finished vs unfinished tasks for a run from persisted state."""
    rows = store.query("SELECT id, status FROM investigation_tasks WHERE run_id = ?", (run_id,))
    out: dict[str, list[str]] = {"finished": [], "unfinished": [], "not_dispatched": []}
    for r in rows:
        if r["status"] in ("done", "failed"):
            out["finished"].append(r["id"])
        elif r["status"] in ("pending", "running"):
            out["unfinished"].append(r["id"])
        else:
            out["not_dispatched"].append(r["id"])
    return out


INVESTIGATOR_SYSTEM = """You are an investigation agent inside EvoAction, a harness that improves its own robot-video temporal annotation workflow.
You answer ONE question using only the tool evidence provided. Rules:
- Ground every claim in the evidence records you are shown; cite their ids (EV-...). Never invent numbers or observations.
- No reference (ground-truth) annotations exist for you. Judge annotation quality only from frames, robot telemetry, the label-free self-assessment and self-consistency.
- If the evidence is insufficient and you have tool rounds left, request at most 3 tool calls instead of guessing.
- Raise a new_question ONLY if the evidence reveals a distinct, unresolved sub-problem that a separate investigation could settle. It must differ from your own question and cite evidence ids. Zero new questions is a fine answer.
Reply with a single JSON object only."""


def _investigator_prompt(task: InvestigationTask, parent: InvestigationTask | None, evidence_view: list[dict[str, Any]],
                         rounds_left: int, workflow_summary: str) -> str:
    parent_ctx = ""
    if parent is not None and parent.result:
        parent_ctx = f"Parent investigation question: {parent.question}\nParent finding: {parent.result.get('finding', '')}\n"
    tools = "\n".join(f"- {d}" for k, d in TOOL_DOCS.items() if k in task.tools_allowed)
    return f"""Question (depth {task.depth}): {task.question}
{parent_ctx}
Incumbent workflow: {workflow_summary}

Evidence records (each record's result is JSON, individually capped):
{compact(evidence_view, 24000)}

Tools you may request ({rounds_left} tool round(s) left):
{tools}

Return JSON with keys:
  "status": "answered" or "need_tools",
  "tool_requests": [{{"tool": name, "args": {{...}}}}]   (only when status is need_tools),
  "finding": short factual conclusion,
  "evidence_ids": [ids supporting the finding],
  "confidence": 0-1,
  "implication_for_workflow": what this suggests changing in the annotation workflow (operators/params/prompt), or "none",
  "new_questions": [{{"question": str, "why": str, "evidence_ids": [...], "tool_requests": [{{"tool": name, "args": {{...}}}}]}}]  (0-2 items)"""


def run_task(task: InvestigationTask, graph: TaskGraph, ctx: ToolContext, workflow_summary: str) -> None:
    graph.set_status(task, "running")
    evidence_view: list[dict[str, Any]] = []
    for ev_id in task.evidence_ids:
        if ev_id in ctx.evidence:
            e = ctx.evidence[ev_id]
            evidence_view.append(evidence_item(ev_id, e["tool"], e["args"], e["result"]))
    for req in task.budget.get("tool_requests", []):
        ev_id, res = run_tool(ctx, str(req.get("tool")), req.get("args") or {}, task.id, task.tools_allowed)
        task.evidence_ids.append(ev_id)
        evidence_view.append(evidence_item(ev_id, req.get("tool"), req.get("args"), res))
    parent = graph.tasks.get(task.parent_task_id) if task.parent_task_id else None
    result: dict[str, Any] | None = None
    attempts = 0
    rounds_left = int(task.budget.get("max_tool_rounds", MAX_TOOL_ROUNDS)) - (1 if task.budget.get("tool_requests") else 0)
    while True:
        prompt = _investigator_prompt(task, parent, evidence_view, rounds_left, workflow_summary)
        try:
            res = ctx.client.chat(ROLE_NEMOTRON, [{"role": "system", "content": INVESTIGATOR_SYSTEM}, {"role": "user", "content": prompt}],
                                  purpose="investigate", json_mode=True, max_tokens=6144, task_id=task.id)
        except BudgetExceeded as e:
            graph.set_status(task, "failed", result={"error": f"budget: {e}"})
            return
        except ModelCallError as e:
            attempts += 1
            graph.store.event(graph.run_id, "task_retry" if attempts <= 1 else "task_giving_up", {"task_id": task.id, "error": str(e)[:300]})
            if attempts <= 1:
                continue
            graph.set_status(task, "failed", result={"error": str(e)[:300]})
            return
        task.model_used = res.model
        data = res.data if isinstance(res.data, dict) else {}
        if data.get("status") == "need_tools" and rounds_left > 0 and data.get("tool_requests"):
            rounds_left -= 1
            for req in data["tool_requests"][:MAX_TOOLS_PER_ROUND]:
                ev_id, out = run_tool(ctx, str(req.get("tool")), req.get("args") or {}, task.id, task.tools_allowed)
                task.evidence_ids.append(ev_id)
                evidence_view.append(evidence_item(ev_id, req.get("tool"), req.get("args"), out))
            continue
        result = {k: data.get(k) for k in ("finding", "evidence_ids", "confidence", "implication_for_workflow")}
        result["call_id"] = res.call_id
        cited = [e for e in (data.get("evidence_ids") or []) if e in ctx.evidence]
        result["evidence_ids"] = cited
        result["uncited_ids_dropped"] = [e for e in (data.get("evidence_ids") or []) if e not in ctx.evidence]
        task.new_questions = [q.get("question", "") for q in (data.get("new_questions") or []) if isinstance(q, dict)][:MAX_CHILDREN]
        result["new_question_requests"] = [q for q in (data.get("new_questions") or []) if isinstance(q, dict)][:MAX_CHILDREN]
        break
    if not result or not result.get("finding"):
        graph.set_status(task, "failed", result={"error": "investigator returned no finding", **(result or {})})
        return
    graph.set_status(task, "done", result=result)


def dispatch(graph: TaskGraph, roots: list[InvestigationTask], ctx: ToolContext, workflow_summary: str) -> None:
    """Breadth-first execution; children are spawned only through TaskGraph.add gates."""
    queue: deque[InvestigationTask] = deque(t for t in roots if t.status == "pending")
    while queue:
        task = queue.popleft()
        run_task(task, graph, ctx, workflow_summary)
        if task.status != "done":
            continue
        for q in task.result.get("new_question_requests", []):
            ev = [e for e in (q.get("evidence_ids") or []) if e in ctx.evidence and e in task.evidence_ids]
            child, verdict = graph.add(str(q.get("question", "")), task, task.tools_allowed, q.get("tool_requests") or [],
                                       evidence_ids=ev, why=str(q.get("why", "")))
            if child.status == "pending":
                queue.append(child)


def tree(graph: TaskGraph) -> list[dict[str, Any]]:
    return [json.loads(t.model_dump_json()) for t in graph.tasks.values()]
