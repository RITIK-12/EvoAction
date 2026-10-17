"""Version N0 (Macrodata Refiner replica on Nemotron) and the default refinement playbook."""

from __future__ import annotations

from evoaction.schemas import HarnessWorkflow, WorkflowNode

# Verbatim from macrodata-labs/refiner src/refiner/robotics/subtask_annotation/segmentation.py
# (_DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE), with Refiner's "Episode instruction:" suffix.
MACRODATA_SEGMENTATION_PROMPT = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.

Episode instruction: $instruction
"""

DEFAULT_REFINE_PROMPT = """You are refining the temporal boundaries of a robot manipulation video annotation.
Task instruction: "$instruction". Video length: $duration_sec s.
Current segments (JSON): $segments_json

Each following image is a close-up strip of consecutive frames (timestamps printed in yellow) around one current boundary or one robot-telemetry hint:
$windows

Policy: a boundary belongs at the first frame where the earlier subtask's world-state change is complete (object released and at rest, gripper opened, drawer closed, pour finished).
For each window choose one action:
- "keep": the current boundary is already within 0.25 s of the correct moment
- "move": give boundary_sec, the corrected time (must lie inside the window)
- "delete": there is no real state change here; the two neighbouring segments are one event
- "insert": (hint windows only) a distinct completed state change happens here and is missing; give boundary_sec and new_subtask for the segment that starts there

Return only JSON: {"windows": [{"window_id": "W1", "action": "keep|move|delete|insert", "boundary_sec": <float or null>, "new_subtask": <string or null>, "reason": "<one short sentence>"}]}
"""


def workflow_n0() -> tuple[HarnessWorkflow, dict[str, str]]:
    """Version N0 (protocol v3): Macrodata Refiner contact-sheet segmentation replicated on Nemotron.

    0.5 s sampling, 224 px tiles, 20 frames/sheet in 5 columns, Refiner timestamp badge, JPEG q95, all sheets in
    one call, temperature 0.1, no post-processing beyond schema validation. Deviation: thinking off (Nemotron Omni
    thinking-on multi-image calls exhausted the token cap without JSON in 31/42 attempts).
    """
    wf = HarnessWorkflow(nodes=[
        WorkflowNode(id="sample", operator="sample_contact_sheets",
                     params={"sampling_hz": 2.0, "tile_px": 224, "frames_per_sheet": 20, "cols": 5, "badge": "macrodata", "jpeg_quality": 95}),
        WorkflowNode(id="annotate", operator="initial_annotate", depends_on=["sample"],
                     params={"prompt": "prompt.md", "thinking": False, "max_tokens": 8192, "temperature": 0.1}),
        WorkflowNode(id="export", operator="validate_export", depends_on=["annotate"]),
    ], max_calls=1, max_wall_seconds=300)
    return wf, {"prompt.md": MACRODATA_SEGMENTATION_PROMPT}
