# Superseded: v1 run with ground-truth-derived agent feedback

This run (evolve `RUN-e9cd34b042`, final `final_test.json`) followed the original contract, which allowed agents to see
"development score summaries". Those summaries were computed from WGO-Bench reference labels (per-episode F1, reference
segment counts, binned offsets to reference boundaries), the supervisor saw validation F1, and promotion was gated on
validation GT F1. Agents demonstrably used this (e.g. "30 predicted vs 12 reference segments").

That violates the stricter requirement that the N+1 loop must improve **without ever seeing ground truth**, so this run
cannot support a label-free self-improvement claim. It is kept unmodified for transparency. Results: both candidates
rejected; sealed pilot test A=C 0.077, B 0.211 F1@0.75.
