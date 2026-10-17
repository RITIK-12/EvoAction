You are refining the temporal boundaries of a robot manipulation video annotation.
Task instruction: "$instruction". Video length: $duration_sec s.
Current segments (JSON): $segments_json

Each following image is a close-up strip of consecutive frames (timestamps printed in yellow) around one current boundary or one robot‑telemetry hint:
$windows

Policy: a boundary belongs at the first frame where the earlier subtask's world‑state change is complete (object released and at rest, gripper opened, drawer closed, pour finished).
For each window choose one action:
- "keep": the current boundary is already within 0.25 s of the correct moment
- "move": give boundary_sec, the corrected time (must lie inside the window)
- "delete": there is no real state change here; the two neighbouring segments are one event
- "insert": (hint windows only) a distinct completed state change happens here and is missing; give boundary_sec and new_subtask for the segment that starts there

Return only JSON: {"windows": [{"window_id": "W1", "action": "keep|move|delete|insert", "boundary_sec": <float or null>, "new_subtask": <string or null>, "reason": "<one short sentence>"}]}