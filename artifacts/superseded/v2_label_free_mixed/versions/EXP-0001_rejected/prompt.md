You are annotating a robot manipulation video.
The images are contact sheets of frames sampled uniformly at $sampling_hz Hz from a $duration_sec-second video, in time order (left to right, top to bottom). Each tile shows its timestamp in seconds in yellow.

Task instruction: "$instruction"

Split the video into the sequence of completed manipulation subtasks the robot performs.
Return only JSON: {"segments": [{"start_sec": <float>, "end_sec": <float>, "subtask": "<short description>"}]}

Additional guidelines:
- Each subtask must span at least 1.5 seconds (or at least two sampled frames) to ensure a meaningful manipulation state change.
- Avoid predicting a subtask for every sampled frame; instead, group consecutive frames that belong to the same manipulation phase.
- If uncertain about exact boundaries, prefer longer segments that capture a clear start and end of a manipulation state change.