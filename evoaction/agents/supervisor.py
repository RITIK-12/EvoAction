"""Nemotron supervisor, strategy and critic roles."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from evoaction.agents.tools import TOOL_DOCS, ToolContext, compact, condense_profile
from evoaction.clients.crusoe import ROLE_NEMOTRON, CrusoeClient
from evoaction.persistence import Store
from evoaction.schemas import PARAM_MODELS, ExperimentProposal, WorkflowPatch
from evoaction.workflow.engine import Version, WorkflowValidationError, apply_patch

GOAL = ("Improve the incumbent robot-video temporal subtask annotation workflow so that it generalizes to unseen robot episodes "
        "(primary metric: Segment F1 at temporal IoU >= 0.75, one-to-one matching), without exceeding 1.5x the incumbent's "
        "annotation inference cost per episode.")

OPERATOR_DOCS = {
    "sample_contact_sheets": "uniformly samples frames and tiles them into timestamped contact sheets (input to initial_annotate).",
    "initial_annotate": "one Crusoe multimodal call over all contact sheets with the versioned prompt file; outputs candidate segments (optionally with per-segment confidence). thinking=false is ~10x cheaper/faster but less deliberate.",
    "state_hint_detection": "deterministic robot-telemetry change points (gripper open/close transitions, end-effector pauses). No model call. Skipped when an episode has no telemetry.",
    "inspect_transition_windows": "builds close-up frame strips (local_hz) around selected boundaries; select = all_boundaries | low_confidence | near_hints | hint_disagreement | short_segments; include_unmatched_hints adds windows at telemetry hints with no nearby boundary (candidates for missing boundaries). No model call.",
    "refine_intervals": "one Crusoe multimodal call over all selected windows with refine.md; per window it may keep, move a boundary, delete a boundary, or insert one at a hint window.",
    "merge_filter_segments": "deterministic cleanup: overlap/gap repair, minimum duration, optional tiling of the full episode.",
    "validate_export": "schema/monotonic check and export. Must be the single terminal node.",
}


def operator_library() -> str:
    lines = []
    for op, doc in OPERATOR_DOCS.items():
        schema = PARAM_MODELS[op].model_json_schema().get("properties", {})
        params = []
        for name, spec in schema.items():
            rng = []
            if "minimum" in spec:
                rng.append(f">={spec['minimum']}")
            if "maximum" in spec:
                rng.append(f"<={spec['maximum']}")
            if "enum" in spec:
                rng.append("|".join(map(str, spec["enum"])))
            if spec.get("items", {}).get("enum"):
                rng.append("list of " + "|".join(spec["items"]["enum"]))
            params.append(f"{name}={spec.get('default')!r}{' (' + ', '.join(rng) + ')' if rng else ''}")
        lines.append(f"- {op}: {doc}\n    params: {'; '.join(params) or 'none'}")
    return "\n".join(lines)


PATCH_DOC = """WorkflowPatch (JSON) fields, all optional:
  insert_nodes: [{"node": {"id": "snake_id", "operator": <operator>, "depends_on": [ids], "params": {...}, "when": "always|has_telemetry|has_boundaries|has_short_segments|has_hints"},
                  "rewire_dependents_of": <existing node id or null>}]   # nodes that depended on that id now depend on the new node
  remove_nodes: [ids]                     # dependents are bridged to the removed node's parents
  set_params: {node_id: {param: value}}
  set_when: {node_id: condition}
  set_depends_on: {node_id: [ids]}
  prompt_updates: {"prompt.md" | "refine.md": "<full new text>"}   # templates may use $instruction $duration_sec $sampling_hz $frame_count $sheet_count; refine.md also $segments_json $windows
Graph invariants checked deterministically: <=8 nodes, acyclic, exactly one initial_annotate, at most one refine_intervals (<=2 multimodal passes), every node's inputs produced upstream (inspect needs segments; refine needs segments+evidence; hint-based selection needs a state_hint_detection ancestor), a single terminal validate_export that every node feeds. No code execution; the evaluator and reference labels are not operators."""


def workflow_summary(v: Version) -> str:
    parts = []
    for n in v.workflow.nodes:
        p = ", ".join(f"{k}={val}" for k, val in n.params.items())
        dep = f" <- {','.join(n.depends_on)}" if n.depends_on else ""
        when = f" [when {n.when}]" if n.when != "always" else ""
        parts.append(f"{n.id}:{n.operator}({p}){dep}{when}")
    return " | ".join(parts) + f" | max_calls/episode={v.workflow.max_calls}"


def _chat(client: CrusoeClient, system: str, user: str, purpose: str, max_tokens: int = 8192, task_id: str | None = None):
    return client.chat(ROLE_NEMOTRON, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       purpose=purpose, json_mode=True, max_tokens=max_tokens, task_id=task_id)


SUPERVISOR_SYSTEM = f"""You are the supervisor of EvoAction, a recursive agent harness that improves its own annotation workflow.
Goal: {GOAL}
You see the incumbent's real failure profile on development episodes (reference labels are hidden; you get binned errors and counts).
Decide which 1-2 focused investigations to launch FIRST. Each is a specific question about WHY the incumbent fails, with initial tool calls.
Do not propose a fix yet and do not assume any particular fix works. Reply with one JSON object only."""


def plan_investigations(client: CrusoeClient, ctx: ToolContext, incumbent: Version, val_view: dict[str, Any],
                        history: list[dict[str, Any]], cost_profile: dict[str, Any]) -> dict[str, Any]:
    from evoaction.data.wgo import load_episode

    dev_view = ctx.incumbent_dev["agent_view"]
    dev_eps = {e: {k: load_episode(e)[k] for k in ("has_telemetry", "duration_sec", "source")} for e in ctx.dev_ids}
    user = f"""Incumbent version {incumbent.id}: {workflow_summary(incumbent)}
Incumbent prompt.md:
{incumbent.prompts.get('prompt.md', '')}

Dev episodes: {json.dumps(dev_eps)}
Dev failure profile (evidence id {ctx.incumbent_dev['eval_id']}):
{compact(condense_profile(dev_view), 9000)}

Validation aggregate for the incumbent (aggregate only): {json.dumps({k: val_view.get(k) for k in ('micro_f1', 'micro_f1_iou05', 'short_event_recall', 'mean_boundary_mae_sec', 'invalid_items', 'empty_output_episodes')})}
Measured incumbent inference cost profile: {json.dumps(cost_profile)}
Previous experiments in this lineage: {compact(history, 3000) if history else 'none'}

Tools:
{chr(10).join('- ' + d for d in TOOL_DOCS.values())}

Return JSON: {{"failure_summary": str, "investigations": [{{"question": str, "tools_allowed": [tool names], "tool_requests": [{{"tool": name, "args": {{...}}}}]}}]}}  (1-2 investigations)"""
    res = _chat(client, SUPERVISOR_SYSTEM, user, "supervisor_plan")
    data = res.data if isinstance(res.data, dict) else {}
    data["call_id"] = res.call_id
    data["model"] = res.model
    return data


STRATEGY_SYSTEM = f"""You are the strategy agent of EvoAction. Goal: {GOAL}
Convert the investigation findings into ONE falsifiable experiment: a hypothesis grounded in cited evidence ids plus an executable WorkflowPatch.
The controller (not you) will run the candidate on held-out validation episodes it never showed you, compare it with the incumbent on the same episodes,
and apply a predeclared gate: validation micro F1@0.75 delta >= 0.02, >= 50% of validation episodes non-regressing, valid outputs, and annotation inference cost <= 1.5x incumbent.
Changing workflow topology (inserting/removing/re-routing operators) is encouraged when the evidence motivates it; parameter- or prompt-only changes are allowed if better justified.
Do not repeat an experiment that was already rejected. Reply with one JSON object only."""


def _proposal_user(findings: list[dict[str, Any]], incumbent: Version, history: list[dict[str, Any]], cost_profile: dict[str, Any],
                   evidence_ids: list[str], extra: str = "") -> str:
    return f"""Incumbent {incumbent.id} workflow JSON: {json.dumps(incumbent.workflow.model_dump())}
Incumbent prompt.md:
{incumbent.prompts.get('prompt.md', '')}
Measured incumbent inference cost profile: {json.dumps(cost_profile)}

Operator library:
{operator_library()}

{PATCH_DOC}

Investigation findings (task tree, real tool evidence):
{compact(findings, 14000)}
Valid evidence ids you may cite: {evidence_ids[-60:]}
Previous experiments: {compact(history, 3000) if history else 'none'}
{extra}
Return JSON with keys: hypothesis, failure_slice, evidence_ids, objective, candidate_patch (WorkflowPatch), expected_outcome,
verification_plan (e.g. {{"metric": "segment_f1_iou075", "split": "validation", "compare_to": "incumbent"}}), stop_conditions (list of strings)."""


def _validate_proposal(data: dict[str, Any], incumbent: Version, exp_id: str, valid_evidence: set[str]) -> tuple[ExperimentProposal | None, list[str]]:
    errors: list[str] = []
    try:
        patch = WorkflowPatch(**(data.get("candidate_patch") or {}))
    except ValidationError as e:
        return None, [f"candidate_patch schema: {x['loc']}: {x['msg']}" for x in e.errors()]
    try:
        apply_patch(incumbent.workflow, incumbent.prompts, patch)
    except WorkflowValidationError as e:
        return None, [f"workflow: {x}" for x in e.errors]
    if patch == WorkflowPatch():
        errors.append("candidate_patch is empty")
    cited = [e for e in (data.get("evidence_ids") or []) if e in valid_evidence]
    if not cited:
        errors.append("proposal cites no valid evidence ids")
    if errors:
        return None, errors
    try:
        prop = ExperimentProposal(id=exp_id, parent_version=incumbent.id, hypothesis=str(data.get("hypothesis", "")),
                                  failure_slice=str(data.get("failure_slice", "")), evidence_ids=cited,
                                  objective=str(data.get("objective", "")), candidate_patch=patch,
                                  expected_outcome=str(data.get("expected_outcome", "")),
                                  verification_plan=data.get("verification_plan") or {},
                                  stop_conditions=[str(x) for x in (data.get("stop_conditions") or [])])
    except ValidationError as e:
        return None, [str(e)[:500]]
    return prop, []


CRITIC_SYSTEM = """You are a skeptical reviewer of a proposed experiment on an annotation workflow.
Check: Is the hypothesis falsifiable and specific? Does it follow from the cited evidence? Does the patch actually test the hypothesis?
Is the cost likely within 1.5x of the incumbent? Does anything rely on hidden reference labels (forbidden)? Reply with one JSON object only."""


def propose_experiment(client: CrusoeClient, store: Store, run_id: str, exp_id: str, findings: list[dict[str, Any]], incumbent: Version,
                       history: list[dict[str, Any]], cost_profile: dict[str, Any], valid_evidence: set[str]) -> dict[str, Any]:
    """Strategy -> deterministic validation (1 repair) -> critic (1 revision). Returns record with proposal or failure."""
    record: dict[str, Any] = {"exp_id": exp_id, "attempts": []}
    ev_list = sorted(valid_evidence)
    res = _chat(client, STRATEGY_SYSTEM, _proposal_user(findings, incumbent, history, cost_profile, ev_list), "strategy_propose")
    data = res.data if isinstance(res.data, dict) else {}
    prop, errors = _validate_proposal(data, incumbent, exp_id, valid_evidence)
    record["attempts"].append({"call_id": res.call_id, "errors": errors, "raw": data})
    if prop is None:
        store.event(run_id, "proposal_invalid", {"exp_id": exp_id, "errors": errors})
        res = _chat(client, STRATEGY_SYSTEM, _proposal_user(findings, incumbent, history, cost_profile, ev_list,
                    extra=f"Your previous proposal was rejected by deterministic validation: {errors}. Previous proposal: {compact(data, 3000)}. Fix it."),
                    "strategy_repair")
        data = res.data if isinstance(res.data, dict) else {}
        prop, errors = _validate_proposal(data, incumbent, exp_id, valid_evidence)
        record["attempts"].append({"call_id": res.call_id, "errors": errors, "raw": data})
        if prop is None:
            store.event(run_id, "proposal_failed", {"exp_id": exp_id, "errors": errors})
            record["status"] = "invalid"
            return record
    crit = _chat(client, CRITIC_SYSTEM, f"""Proposal: {prop.model_dump_json()}
Incumbent: {workflow_summary(incumbent)}
Cost profile: {json.dumps(cost_profile)}
Cited evidence: {compact([f for f in findings], 5000)}
Return JSON: {{"verdict": "approve" | "revise", "issues": [str], "required_changes": str}}""", "critic_review", max_tokens=4096)
    cdata = crit.data if isinstance(crit.data, dict) else {}
    record["critique"] = {"call_id": crit.call_id, **cdata}
    store.event(run_id, "proposal_critique", {"exp_id": exp_id, "verdict": cdata.get("verdict"), "issues": cdata.get("issues")})
    if str(cdata.get("verdict", "")).lower() == "revise":
        res = _chat(client, STRATEGY_SYSTEM, _proposal_user(findings, incumbent, history, cost_profile, ev_list,
                    extra=f"A reviewer asked for revisions: {json.dumps(cdata)}. Your proposal: {prop.model_dump_json()}. Return a revised proposal."),
                    "strategy_revise")
        data = res.data if isinstance(res.data, dict) else {}
        revised, errors = _validate_proposal(data, incumbent, exp_id, valid_evidence)
        record["attempts"].append({"call_id": res.call_id, "errors": errors, "raw": data, "after_critique": True})
        if revised is not None:
            prop = revised
        else:
            store.event(run_id, "proposal_revision_invalid", {"exp_id": exp_id, "errors": errors, "kept": "pre-critique proposal"})
    prop.authored_by = client.nemotron_model
    record["status"] = "valid"
    record["proposal"] = json.loads(prop.model_dump_json())
    return record
