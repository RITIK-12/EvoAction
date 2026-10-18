# Superseded: first human-in-the-loop run on batch 2 (Galaxea + HomER), RUN-0310a37697

Both candidates were rejected by the valid-output rule. Root causes, from the run's logs:

1. Every valid patch set `annotate.thinking=true`. Nemotron Omni then reasoned to the 8,192-token cap and returned no JSON
   in 65/83 (EXP-0001) and 77/89 (EXP-0002) annotate calls, so 32 and 38 of 50 episodes ended `model_error`.
   This run predates the thinking fallback (a no-JSON thinking call now falls back to a non-thinking call).
2. Best-round selection used labeled-sample F1 alone, and failed episodes shrink the prediction count, which inflates F1.
   The selected rounds had 2-4 of 5 labeled episodes failed (e.g. 10 predictions for 105 human segments, F1 0.071).
   Selection now ranks rounds with zero failed labeled episodes first.
3. Recursion 1 lost rounds 1-2 to invalid patches (inserted nodes written flat instead of {"node": ...}; an invalid
   `select` value). Flat inserted nodes are now normalized.

Kept for the record; the batch was re-run after these fixes.
