# Superseded: v2 label-free run on mixed DROID+Galaxea splits

Agents saw no ground truth (label-free protocol), but versions were scored on different episode subsets
(dev/validation/test splits, pilot subsets), the baselines were our own A/B designs rather than Macrodata's
published protocol, and promotion had a per-episode 1.5x cost cap. Protocol v3 replaces this: every version
runs on the same 50 DROID episodes, version N0 replicates Macrodata Refiner's contact-sheet protocol on
Nemotron, the headline metric is Macrodata's snapped pooled Segment F1@0.75, and cost is tracked per recursion.
Kept unmodified for transparency.
