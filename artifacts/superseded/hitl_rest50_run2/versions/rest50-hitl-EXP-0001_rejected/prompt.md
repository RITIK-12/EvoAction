Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-15 seconds. Shorter segments are allowed only for fast pick, place, open, close, or release events that are clearly instantaneous (<1.5 sec).
- Do not output segments shorter than 1.0 second unless the action is a definitive instantaneous release or grasp.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- Prefer fewer, longer segments that correctly capture state changes over many short segments.
- If uncertain, merge adjacent short segments into a longer segment that spans the whole manipulation.

Episode instruction: $instruction