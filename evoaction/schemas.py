"""Pydantic data contracts for workflows, experiments and investigations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OperatorName = Literal[
    "sample_contact_sheets",
    "initial_annotate",
    "inspect_transition_windows",
    "state_hint_detection",
    "refine_intervals",
    "merge_filter_segments",
    "validate_export",
]

Condition = Literal["always", "has_telemetry", "has_boundaries", "has_short_segments", "has_hints"]


class Segment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str = ""
    confidence: float | None = None


# -- operator parameter schemas (unknown keys rejected) ---------------------

class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SampleParams(_Params):
    sampling_hz: float = Field(1.0, ge=0.25, le=5.0)
    tile_px: int = Field(224, ge=128, le=448)
    frames_per_sheet: int = Field(20, ge=4, le=30)
    cols: int = Field(5, ge=2, le=8)


class AnnotateParams(_Params):
    prompt: str = "prompt.md"
    thinking: bool = True
    max_tokens: int = Field(16384, ge=512, le=32768)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class HintParams(_Params):
    signals: list[Literal["gripper", "eef_pause"]] = Field(default_factory=lambda: ["gripper"])
    min_gap_sec: float = Field(0.8, ge=0.1, le=5.0)
    pause_speed_quantile: float = Field(0.15, ge=0.01, le=0.5)
    min_pause_sec: float = Field(0.3, ge=0.1, le=3.0)


class InspectParams(_Params):
    select: Literal["all_boundaries", "low_confidence", "near_hints", "hint_disagreement", "short_segments"] = "all_boundaries"
    window_sec: float = Field(2.0, ge=0.5, le=4.0)
    local_hz: float = Field(4.0, ge=2.0, le=5.0)
    max_windows: int = Field(8, ge=1, le=16)
    include_unmatched_hints: bool = False
    tile_px: int = Field(224, ge=128, le=448)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class RefineParams(_Params):
    prompt: str = "refine.md"
    thinking: bool = True
    max_tokens: int = Field(16384, ge=512, le=32768)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    allow_insert: bool = True
    allow_delete: bool = True


class MergeParams(_Params):
    min_duration_sec: float = Field(0.4, ge=0.0, le=5.0)
    max_gap_sec: float = Field(1.0, ge=0.0, le=5.0)
    tile_full_episode: bool = True
    merge_same_label: bool = False


class ExportParams(_Params):
    pass


PARAM_MODELS: dict[str, type[_Params]] = {
    "sample_contact_sheets": SampleParams,
    "initial_annotate": AnnotateParams,
    "state_hint_detection": HintParams,
    "inspect_transition_windows": InspectParams,
    "refine_intervals": RefineParams,
    "merge_filter_segments": MergeParams,
    "validate_export": ExportParams,
}


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    operator: OperatorName
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    when: Condition = "always"


class HarnessWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[WorkflowNode]
    max_calls: int = Field(3, ge=1, le=6)  # multimodal calls per episode
    max_wall_seconds: int = Field(240, ge=10, le=900)


# -- experiment + investigation contracts -----------------------------------

class NodeInsert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: WorkflowNode
    rewire_dependents_of: str | None = None  # nodes that depended on this id now depend on the new node


class WorkflowPatch(BaseModel):
    """Bounded, typed edit to a parent workflow. No code, only graph/param/prompt changes."""

    model_config = ConfigDict(extra="forbid")
    insert_nodes: list[NodeInsert] = Field(default_factory=list)
    remove_nodes: list[str] = Field(default_factory=list)
    set_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    set_when: dict[str, Condition] = Field(default_factory=dict)
    set_depends_on: dict[str, list[str]] = Field(default_factory=dict)
    prompt_updates: dict[str, str] = Field(default_factory=dict)  # file name -> full new text (prompt.md / refine.md)

    @field_validator("prompt_updates")
    @classmethod
    def _prompt_names(cls, v: dict[str, str]) -> dict[str, str]:
        for k, text in v.items():
            if k not in ("prompt.md", "refine.md"):
                raise ValueError(f"prompt file {k!r} not allowed (prompt.md, refine.md)")
            if len(text) > 6000:
                raise ValueError(f"{k} too long")
        return v

    def is_topology_change(self) -> bool:
        return bool(self.insert_nodes or self.remove_nodes or self.set_depends_on or self.set_when)


class ExperimentProposal(BaseModel):
    id: str
    parent_version: str
    hypothesis: str
    failure_slice: str
    evidence_ids: list[str] = Field(default_factory=list)
    objective: str
    candidate_patch: WorkflowPatch
    expected_outcome: str
    verification_plan: dict[str, Any] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    authored_by: str = ""
    task_ids: list[str] = Field(default_factory=list)


TaskStatus = Literal["pending", "running", "done", "failed", "deduplicated", "rejected"]


class InvestigationTask(BaseModel):
    id: str
    parent_task_id: str | None = None
    depth: int = 0
    question: str
    evidence_ids: list[str] = Field(default_factory=list)
    tools_allowed: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = "pending"
    result: dict[str, Any] | None = None
    new_questions: list[str] = Field(default_factory=list)
    model_used: str | None = None
    run_id: str | None = None
