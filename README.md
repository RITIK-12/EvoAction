# EvoAction

**A self-evolving, recursively delegated agent harness, built for the Crusoe × Cerebral Valley Recursive Agents Hackathon (Track 3: Version N+1).**

The working environment is robot-video temporal subtask segmentation on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), scored with **Macrodata's own evaluation protocol** so the results sit next to their published model sweep. Everything runs on NVIDIA Nemotron models served by Crusoe Managed Inference.

1. **Version N0** replicates Macrodata's Refiner annotation protocol on `Nemotron-3-Nano-Omni-30B-A3B`: their prompt word for word, 0.5 s sampling, 224 px tiles, 20 frames per sheet, their timestamp badge, all sheets in one call, and no post-processing.
2. **N → N+1 without ground truth.** A Nemotron-3-Super-120B supervisor reads N's **label-free self-assessment**. It launches investigation agents that work from robot telemetry, close-up frames and the system's own predictions, and an investigator can spawn child investigations through a gated dispatcher. A strategy agent writes a typed, executable `WorkflowPatch`, which a critic reviews. The controller runs the candidate on **the same episodes** and applies a predeclared label-free gate. Ground truth is never shown to agents or used by the gate.
3. **Measurement.** The protected evaluator scores every version against the human labels afterwards, for this README and the report only.
4. **Batches.** The 100 benchmark episodes are evaluated in two batches of 50, each with its own lineage, and pooled per recursion level into the full benchmark.

Every number below comes from persisted runs in `artifacts/` and was re-scored by an independent script (`linear_sum_assignment` matching), which gave identical true-positive counts.

## Results

Metric (Macrodata's): **Segment F1 at temporal IoU ≥ 0.75**, pooled over episodes, one-to-one matching, with the first predicted start and last predicted end snapped to the human episode boundaries. Labels are ignored. Raw (unsnapped) F1@0.75 and snapped F1@0.5 are reported alongside.

### Full benchmark: 100 episodes, pooled per recursion level

| Recursion | Versions (batch 1 + batch 2) | Snapped F1@0.75 | Raw F1@0.75 | Snapped F1@0.5 | Pred / ref | Annotation cost (100 ep) |
|---|---|---|---|---|---|---|
| 0 | N0 + N0 | **0.036** | 0.021 | 0.162 | 1498 / 743 | $0.203 |
| 1 | N001 + N0 | **0.035** | 0.027 | 0.162 | 1498 / 743 | $0.220 |
| 2 | N001 + N0 | **0.035** | 0.027 | 0.162 | 1498 / 743 | $0.220 |

The comparison below is Macrodata's original model sweep: Segment F1 at tIoU ≥ 0.75 on the full 100-episode benchmark, contact-sheet setup, before their final prompt optimization. Nemotron was not in their sweep.

| # | System | Predicted segments | Segment F1@0.75 |
|---|---|---|---|
| 1 | Gemini 3.5 Flash | 645 | 0.290 |
| 2 | Gemini Robotics ER 1.6 | 556 | 0.245 |
| 3 | GPT-5.5 (low) | 602 | 0.233 |
| 4 | GPT-5.4 (low) | 529 | 0.233 |
| 5 | Gemini 3.1 Pro Preview | 586 | 0.232 |
| 6 | Gemini 3 Flash Preview | 685 | 0.210 |
| 7 | Claude Opus 4.8 | 733 | 0.168 |
| 8 | Claude Sonnet 4.6 | 698 | 0.146 |
| 9 | Gemini 2.5 Pro | 662 | 0.145 |
| 10 | Qwen 3.5 Plus | 763 | 0.141 |
| 11 | Claude Haiku 4.5 | 570 | 0.128 |
| 12 | Gemini 3.1 Flash Lite | 757 | 0.119 |
| 13 | Qwen 3.6 Flash | 715 | 0.110 |
| 14 | GPT-5.4 mini | 317 | 0.106 |
| 15 | Gemini 2.5 Flash | 855 | 0.059 |
| 16 | Gemma 4 26B | 1080 | 0.057 |
| 17 | **EvoAction recursion 0** (N0, Nemotron-3-Nano-Omni-30B-A3B) | 1498 | **0.036** |
| 18 | **EvoAction recursion 1–2** (N001 + N0, same model) | 1498 | **0.035** |

### Batch 1: all 50 DROID episodes (150 reference segments)

| Recursion | Version | Snapped F1@0.75 | Raw F1@0.75 | Snapped F1@0.5 | Pred | Label-free score (gate) | Gate | Annotation cost (50 ep) | Recursion cost |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **N0** (Macrodata protocol on Nemotron Omni) | **0.081** | 0.013 | 0.179 | 320 | 0.336 | baseline | $0.056 | – |
| 1 | EXP-0001: telemetry hints → close-up windows → refine → merge | 0.077 | **0.043** | 0.179 | 320 | **0.418** | **PROMOTED → N001** | $0.074 | $0.057 |
| 2 | EXP-0002: 30 frames per sheet, 5 Hz windows, denser hints | 0.074 | 0.052 | 0.195 | 311 | 0.393 | REJECTED | $0.071 | $0.144 |
| final | **N001** | 0.077 | 0.043 | 0.179 | 320 | 0.418 | incumbent | $0.074 | |

### Batch 2: 25 Galaxea + 25 HomER (593 reference segments)

| Recursion | Version | Snapped F1@0.75 | Raw F1@0.75 | Snapped F1@0.5 | Pred | Label-free score (gate) | Gate | Annotation cost (50 ep) | Recursion cost |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **N0** (Macrodata protocol on Nemotron Omni) | **0.024** | 0.023 | 0.157 | 1178 | 0.493 | baseline | $0.147 | – |
| 1 | rest50-EXP-0001: telemetry hints → windows → conditional refine | 0.024 | 0.023 | 0.157 | 1178 | 0.491 | REJECTED | $0.159 | $0.069 |
| 2 | rest50-EXP-0002: same with tighter windows (4 × 1.5 s at 3 Hz) | 0.023 | 0.021 | 0.157 | 1178 | 0.497 | REJECTED (+0.003 < 0.03) | $0.201 | $0.088 |
| final | **N0** | 0.024 | 0.023 | 0.157 | 1178 | 0.493 | incumbent | $0.147 | |

## Why Nemotron N0 scores below every model in the sweep

The pipeline was checked end to end, and the shortfall comes from what the model outputs, not from the harness:
- **Frames:** all frames reach the model. The longest DROID episode produces 215 frames on 11 sheets, sent in one call, and the model correctly reports 11 images spanning 000.00 s to 107.00 s.
- **Prompt:** identical to Macrodata's.
- **Scores:** re-scored independently, with identical results.

Measured on N0's own outputs:

| Dataset | Predicted / reference segments | Median predicted segment | Median reference segment | Predicted segments ≤ 2 s | Predicted segments exactly 1.0 s | Median coverage (last predicted end ÷ reference end) | Episodes with < 50% coverage |
|---|---|---|---|---|---|---|---|
| DROID | 320 / 150 (2.1×) | 1.0 s | 7.9 s | 87% | 184 | 0.55 | 23 of 50 |
| Galaxea | 188 / 123 (1.5×) | 1.2 s | 2.9 s | 80% | 88 | 1.08 | 6 of 25 |
| HomER | 994 / 470 (2.1×) | 1.0 s | 2.9 s | 93% | 915 | 1.00 | 7 of 25 |

Counts are raw model output. Four HomER items were invalid and dropped by the evaluator, so 1,498 segments were scored.

1. **It captions the sampling grid instead of segmenting events.** 79% of all predicted segments last exactly 1.0 s, which is two frames at the 0.5 s sampling rate. The prompt explicitly asks for "2-10 seconds", but the model emits one segment per frame pair. As a result, 1,498 predictions face 743 reference segments, more than any model in the sweep (Gemma 4 26B predicted 1,080 and scored 0.057).
2. **That makes IoU ≥ 0.75 nearly unreachable.** A 1.0 s segment can only reach IoU 0.75 against a reference segment shorter than 1.33 s. DROID's shortest reference segment is 2.3 s, so raw DROID F1@0.75 is 0.013.
3. **On DROID it stops early.** In 23 of 50 episodes, predictions end before half the annotated span; on the 107 s episode they end at 17 s. The snapped headline of 0.081 is mostly snapping stretching the last predicted segment to the episode end. Raw F1 is 0.013.
4. **Likely model-side causes:**
   - **Thinking off.** It's off because thinking-on multi-image calls exhausted the token cap and returned no JSON in 31 of 42 attempts.
   - **A small model.** It has 3B active parameters (MoE).
   - **Little visual detail per frame.** About 570 prompt tokens per 20-frame sheet on this provider, or about 28 tokens per frame. Gemma 4 26B, the closest open model in the sweep, shows the same pattern: the most predicted segments and the lowest score.

**What the agents saw without labels.**
- The DROID supervisor diagnosed "uniformly spaced early boundaries (approximately 1-second intervals)". Investigators confirmed that predictions stop after the first sheet and sit on a 1-second grid.
- Their patches added a telemetry-guided refinement stage. On DROID that moved boundaries within the predicted span: raw F1@0.75 rose 3.3×, from 3 to 10 exact matches.
- It did not change granularity or coverage, which are what limit the score.

**Note on further fixes.** Any fix designed now (for example per-sheet calls stitched together, or fewer frames per sheet) would be informed by this ground-truth analysis. It would have to be reported as a human-designed baseline variant, not as an agent-found N+1.

## What the N+1 loop did (no ground truth)

- **Batch 1, recursion 1.** EXP-0001 inserted a telemetry-guided refinement stage, a **topology change** from 3 nodes to 7. It was promoted on the label-free gate: +0.082, with 96% of episodes non-regressing. Measured afterwards, raw F1@0.75 went from 0.013 to 0.043, while the snapped headline lost one match (19 → 18).
- **Batch 1, recursion 2.** EXP-0002 changed sampling parameters only, and the label-free score fell. The candidate was rejected, then frozen and kept. Measured afterwards, it had the best raw F1@0.75 (0.052) and F1@0.5 (0.195) of any DROID version.
- **Batch 2.** Both candidates were telemetry-guided refinement variants, and both were rejected. The label-free deltas were −0.002 and +0.003, against a +0.03 threshold. Ground truth confirmed neither helped.
- **Honest reading.**
  - The label-free score rewarded EXP-0001 mostly through segment-duration variability (0.33 → 0.91). Telemetry agreement stayed at 0.188.
  - The proxy tracks raw boundary accuracy but not the snapped headline.
  - With 150 or 593 reference segments, one-match changes are noise.
  - Investigation depth reached 1 in both batches: no investigator raised an evidence-backed follow-up question that passed the dispatcher's gates.

## Protocol

- **Same episodes for every version.** Batch sets are immutable and hashed (`artifacts/episode_sets/{droid50,rest50}.json`). There are no dev/validation/test splits, because nothing is trained. All N, N+1, N+2 numbers in a table come from the same episodes.
- **What agents see:**
  - the incumbent's predictions
  - video frames, through the Crusoe vision model
  - robot telemetry
  - sampling metadata
  - a predeclared **label-free self-score** (`evoaction/selfcheck/label_free.py`), which combines:
    - F1 between predicted boundaries and gripper open/close events (0.35)
    - vision-model confirmation that a state change completes at each boundary (0.35)
    - segment-duration variability (0.15)
    - temporal resolution (0.15)
- **Promotion gate (controller only):** label-free score +0.03 or more, at least 50% of episodes non-regressing, and valid outputs. There is no per-episode cost cap. **Total cost per recursion** (agents plus candidate run) and per version is recorded and reported.
- **Ground truth** is read only by `evoaction/evaluation/protected.py`. Guards:
  - tests plant sentinel ground-truth values and check they never appear in an agent prompt
  - a static test checks that agents and self-check code never touch the protected store
  - a post-run scan of all 24 cached agent responses from both batches found no ground-truth vocabulary; the only "f1" in agent inputs is the label-free `telemetry_agreement.f1`

## Faithfulness to Macrodata's setup, and deviations

| | Macrodata (published Refiner code and post) | EvoAction N0 |
|---|---|---|
| Prompt | Refiner segmentation prompt | identical, plus the episode instruction line |
| Sampling | 0.5 s, 224 px tiles, 20 per sheet, 5 columns, all sheets in one call | identical |
| Timestamp badge | black box top-left, white `000.00s`, 17 pt | identical |
| JPEG / temperature | quality 95, subsampling 2 / 0.1 | identical |
| Post-processing | drop invalid, clamp, sort | identical |
| Scoring | snapped pooled Segment F1@0.75 | identical (raw also reported) |
| Model | 16 VLMs (sweep above) | Nemotron-3-Nano-Omni-30B-A3B, `thinking=false` |

Deviations, recorded honestly:
- **Thinking off.** With thinking on, 31 of 42 multi-image calls exhausted the token cap while reasoning and returned no JSON.
- **Prompt version.** The Refiner prompt includes Macrodata's duration prior ("Most segments should be 2-10 seconds"). The sweep above was run before their final prompt optimization, so if anything N0 uses the stronger prompt.
- **Context tokens.** Prompt tokens grow linearly at about 570 per sheet: 1, 2, 4 and 11 sheets give 601, 1173, 2307 and 6138 tokens. Agents are told this measured fact, and `frames_per_sheet` is a parameter they can change.
- **Matching.** We use maximum-cardinality bipartite matching at IoU ≥ 0.75; the post doesn't specify the matcher. Our snapping drops a segment if snapping inverts it. That affects 17 batch-2 episodes and no DROID episode. True-positive counts are identical either way.
- **Failed calls are not cached.** In the standalone baseline files, one episode per batch failed with invalid JSON on both attempts. When each lineage re-ran N0 as recursion 0, it succeeded, so the lineage N0 rows (every episode OK) are the reference.
- **In-sample selection.** The gate selects on the same episodes it reports. It is label-free, so no labels leak, but improvements are not a held-out generalization claim.
- **DROID's 50 episodes cover 26 tasks**, mostly in twin pairs with the same instruction and duration.

## Quick start

```bash
uv sync --python 3.12
cp .env.example .env                                 # add CRUSOE_API_KEY
mkdir -p data/wgo                                    # full WGO-Bench parquet (1.4 GB, gitignored)
curl -L -o data/wgo/annotations.parquet https://huggingface.co/datasets/macrodata/WGO-Bench/resolve/main/data/annotations.parquet
uv run python -m evoaction setup                     # ingest all 100 episodes, lock droid50 and rest50
uv run python -m evoaction baseline --set droid50    # N0 on batch 1
uv run python -m evoaction evolve --set droid50      # label-free N -> N+1 -> N+2 on batch 1
uv run python -m evoaction evolve --set rest50       # same for batch 2
uv run python -m evoaction report                    # artifacts/report.html
```

Runs used `MAX_VISION_CALLS=1000–1200 RESERVED_TEST_VISION_CALLS=0 MAX_NEMOTRON_CALLS=80 WORKERS=8`. Tests: `uv run --group dev pytest -q`.

## Models (Crusoe Managed Inference, confirmed live via `GET /v1/models`)

| Role | Model |
|---|---|
| Supervisor, investigators, strategy, critic | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Annotation, refinement, frame inspection, label-free verification | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` |

`nvidia/NVIDIA-Nemotron-3.5-Super-VL` was also probed. It accepts at most one image per request, so it cannot take multi-sheet input.

## Workflow graph (the thing that evolves)

Agents submit typed, Pydantic-validated `WorkflowPatch` objects over an operator allowlist. No generated code is executed.

| Operator | Behaviour |
|---|---|
| `sample_contact_sheets` | Uniform frames tiled into timestamped contact sheets |
| `initial_annotate` | One multimodal call over all sheets with the versioned `prompt.md` |
| `state_hint_detection` | Deterministic gripper and end-effector change points from telemetry |
| `inspect_transition_windows` | Close-up frame strips around boundaries or hints |
| `refine_intervals` | One multimodal call that keeps, moves, deletes or inserts boundaries (`refine.md`) |
| `merge_filter_segments` | Deterministic overlap and gap repair, minimum duration |
| `validate_export` | Schema and monotonicity checks, then export (single terminal node) |

Version directories in `artifacts/versions/` are immutable and SHA-256 hashed. Rejected candidates are frozen and kept, and promotion copies the candidate to a new `N00k` directory. Recursion is bounded at depth 3, 2 children per parent and 12 tasks. Inference is cached on model, full input and parameters, so re-running an unchanged version is free and every candidate is compared on identical upstream outputs.

## Layout

```
evoaction/data/wgo.py          WGO ingest (agent-visible media vs evaluator-only labels), batch episode sets
evoaction/media/               deterministic contact sheets (Macrodata badge)
evoaction/evaluation/          pure metrics + protected evaluator (only gold reader)
evoaction/selfcheck/           label-free self-assessment
evoaction/workflow/            operators, graph validation/execution, version store, N0
evoaction/agents/              tools, recursive investigator task graph, supervisor/strategy/critic
evoaction/experiments/         evolution controller (label-free gate, promotion, rollback, per-recursion cost)
evoaction/report.py            static HTML evidence report
artifacts/                     episode sets, baselines, lineage*, evolution*, versions, runs, events, SQLite DB, report.html
```

Apache-2.0. WGO-Bench data is CC-BY-NC-SA-4.0 and is not redistributed here (`data/` is gitignored).
