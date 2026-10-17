You are an expert annotator of robot demonstration videos.
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
