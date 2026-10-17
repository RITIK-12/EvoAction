You are annotating a robot manipulation video.
The images are contact sheets of frames sampled uniformly at $sampling_hz Hz from a $duration_sec-second video, in time order (left to right, top to bottom). Each tile shows its timestamp in seconds in yellow.

Task instruction: "$instruction"

Split the video into the sequence of completed manipulation subtasks the robot performs.
Return only JSON: {"segments": [{"start_sec": <float>, "end_sec": <float>, "subtask": "<short description>"}]}
