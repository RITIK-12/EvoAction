"""Fixed systems A (N000) and B (strong_fixed), plus the default refinement playbook."""

from __future__ import annotations

from evoaction.schemas import HarnessWorkflow, WorkflowNode

PROMPT_A = """You are annotating a robot manipulation video.
The images are contact sheets of frames sampled uniformly at $sampling_hz Hz from a $duration_sec-second video, in time order (left to right, top to bottom). Each tile shows its timestamp in seconds in yellow.

Task instruction: "$instruction"

Split the video into the sequence of completed manipulation subtasks the robot performs.
Return only JSON: {"segments": [{"start_sec": <float>, "end_sec": <float>, "subtask": "<short description>"}]}
"""

PROMPT_B = """You are an expert annotator of robot demonstration videos.
The images are $sheet_count contact sheet(s) with $frame_count frames sampled uniformly at $sampling_hz Hz from a $duration_sec-second video, in time order (left to right, top to bottom). Each tile shows its timestamp in seconds in yellow.

Task instruction: "$instruction"

Annotation policy - completed world-state changes:
- One segment = one completed manipulation event that changes the state of the world: an object is picked up (grasped and lifted), placed (released at its destination), a drawer/lid/door is opened or closed, contents are poured, an object is pushed or moved to a new place.
- A segment ENDS at the first moment its state change is complete and visible (e.g. the object rests at its destination and the gripper has released it). The next segment starts exactly there.
- Do not create segments for small pose adjustments, pure approach motions, or pauses; fold the approach/reach into the event it serves.
- Segments are contiguous and cover the whole video: the first starts at 0.0, each start equals the previous end, the last ends at $duration_sec.
- Use the printed timestamps. A boundary may fall between two sampled frames; estimate it to 0.1 s.
- Short events (under 2 s) are common for quick grasps and releases; never merge two distinct state changes into one segment.
- For two-armed robots, a completed change by either arm is an event.
- Labels are short imperative phrases grounded in what is visible, e.g. "pick up the red cup from the table".
- confidence = your probability (0-1) that this segment's end boundary is within 0.5 s of the true moment.

Return only JSON: {"segments": [{"start_sec": <float>, "end_sec": <float>, "subtask": "<label>", "confidence": <float>}]}
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


def workflow_a() -> tuple[HarnessWorkflow, dict[str, str]]:
    wf = HarnessWorkflow(nodes=[
        WorkflowNode(id="sample", operator="sample_contact_sheets", params={"sampling_hz": 1.0, "tile_px": 224, "frames_per_sheet": 20, "cols": 5}),
        WorkflowNode(id="annotate", operator="initial_annotate", depends_on=["sample"], params={"prompt": "prompt.md", "thinking": False, "max_tokens": 8192}),
        WorkflowNode(id="export", operator="validate_export", depends_on=["annotate"]),
    ], max_calls=1, max_wall_seconds=240)
    return wf, {"prompt.md": PROMPT_A}


def workflow_b() -> tuple[HarnessWorkflow, dict[str, str]]:
    wf = HarnessWorkflow(nodes=[
        WorkflowNode(id="sample", operator="sample_contact_sheets", params={"sampling_hz": 2.0, "tile_px": 224, "frames_per_sheet": 20, "cols": 5}),
        WorkflowNode(id="annotate", operator="initial_annotate", depends_on=["sample"], params={"prompt": "prompt.md", "thinking": False, "max_tokens": 8192}),
        WorkflowNode(id="merge", operator="merge_filter_segments", depends_on=["annotate"],
                     params={"min_duration_sec": 0.4, "max_gap_sec": 1.0, "tile_full_episode": True}),
        WorkflowNode(id="export", operator="validate_export", depends_on=["merge"]),
    ], max_calls=1, max_wall_seconds=240)
    return wf, {"prompt.md": PROMPT_B}
