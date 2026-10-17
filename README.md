# EvoAction

**A self-evolving, recursively delegated agent harness, built for the Crusoe × Cerebral Valley Recursive Agents Hackathon (Track 3: Version N+1).**

Robot manipulation subtask annotation is the working environment, because it is hard and can be scored objectively. Given a robot demonstration video, the task is to output timestamped segments, one per completed world-state change, in the style of WGO-Bench. Starting from a one-line goal and an incumbent annotation workflow, EvoAction:

1. Runs the incumbent on real robot episodes using a Crusoe-hosted NVIDIA Nemotron vision model.
2. Has a **Nemotron supervisor** read the incumbent's **label-free self-assessment** and launch focused **investigation agents**. Agents **never see ground truth**. An investigator that finds a distinct, unresolved sub-problem can request a child investigation. The dispatcher spawns it only if it passes novelty, cycle, depth, fan-out, budget and evidence-citation checks.
3. Turns findings into a falsifiable **`ExperimentProposal`** with a typed **workflow patch** that can change topology, parameters or prompts. A Nemotron critic reviews it, and invalid patches get one bounded repair.
4. Runs the candidate on held-out validation episodes. The **controller** then applies a predeclared **label-free** promotion rule and **promotes or rejects** the candidate. Ground truth is used only for a shadow measurement recorded after the decision. Rejected versions are frozen and kept; the incumbent is reloaded from its immutable artifacts.
5. Freezes **A** (incumbent `N000`), **B** (a strong fixed pipeline with no agents) and **C** (the best evolved harness, or the unchanged incumbent if nothing was promoted). All three run **once** on a sealed held-out split, and the run reports quality, inference cost and optimization overhead.

Nothing is preordained. Promotion, rejection, grandchild spawning and benchmark wins are empirical outcomes, and `artifacts/report.html` shows only what actually happened.

## Results (v2 no-ground-truth protocol; every number comes from `artifacts/`)

**Version N** (`N000`, scored against ground truth by the experimenters only):

| | dev (5 episodes) | validation (4 episodes) |
|---|---|---|
| F1@0.75 | 0.081 | 0.098 |
| label-free self-score | 0.543 | 0.511 |

**N+1 loop** (`RUN-83d7bcb960`, agents saw no ground truth):
- The Nemotron supervisor diagnosed evenly spaced ~1 s boundaries with no visual confirmation and poor telemetry alignment. It worked from label-free evidence only.
- It launched 4 depth-1 investigations. No investigator raised an evidence-backed follow-up question, so the maximum depth reached was 1.
- Ground-truth vocabulary appears 0 times in agent-written text.

| Experiment | Change | Label-free score (gate) | Decision | Shadow GT F1@0.75 (measured afterwards) |
|---|---|---|---|---|
| EXP-0001 | prompt: longer subtasks | 0.511 → 0.366 | REJECTED (score fell) | 0.098 → 0.098 (F1@0.5 0.098 → 0.341) |
| EXP-0002 | topology: 2 Hz + gripper hints → windows → refine → merge | 0.511 → 0.602 | REJECTED (cost 2.56× > 1.5×) | 0.098 → 0.140 |

**Sealed held-out test:** the 3 episodes never evaluated before (`robointer_droid_000038, galaxea_033, galaxea_039`). A, B and C were frozen first and the test ran once.

| System | F1@0.75 | F1@0.5 | Label-free self-score | Annotation cost |
|---|---|---|---|---|
| A · N000 | 0.037 | 0.222 | 0.241 | $0.0040 |
| B · strong fixed | 0.000 | 0.091 | 0.316 | $0.0031 |
| C · unchanged incumbent (no candidate promoted) | 0.037 | 0.222 | 0.241 | $0.0040 |

Continuity check on the v1 pilot test episodes, which were already scored once: A 0.077, B 0.211, C 0.077 (F1@0.75).

**What this shows, honestly:**
- EvoAction ran a real N → N+1 loop with no ground truth. EXP-0002 changed the topology using robot telemetry, improved the label-free score, and, measured afterwards, also improved validation F1@0.75 (0.098 → 0.140). The predeclared 1.5× cost cap rejected it anyway, so C is the unchanged incumbent.
- The label-free proxy is informative but imperfect. It agreed with ground truth on EXP-0002 and disagreed on EXP-0001 and on the A-vs-B ranking of the fresh test.
- B vs A flips between test sets: B wins on the pilot test, A wins on the fresh test. With 3–4 episodes per split, sample variance dominates and nothing is statistically established.

## No-ground-truth protocol (v2)

N+1 must improve the workflow **without ever looking at ground truth**. Version N's annotations are scored against the WGO-Bench labels, but those metrics are for the experimenters only.

- **What agents see:** their own predictions, video frames (through the Crusoe vision model), robot telemetry, sampling metadata, and a predeclared **label-free self-score** (`evoaction/selfcheck/label_free.py`). The score combines:
  - agreement between predicted boundaries and gripper open/close events (0.35)
  - Crusoe vision-model confirmation that a state change completes at each boundary (0.35)
  - variability of segment durations, since evenly spaced segments are an artifact of tiled sampling (0.15)
  - temporal resolution: no segments shorter than two sampled frames (0.15)
- **Promotion rule:** label-free self-score improves by at least 0.03, at least 50% of validation episodes do not regress on that score, outputs are valid, and annotation cost is at most 1.5×.
- **Ground truth** is read only by `evoaction/evaluation/protected.py`. It produces a **shadow measurement** after each decision, which goes to the report only, and the sealed final test. That answers the real question: did the label-free self-improvement actually improve accuracy?
- **Guards:**
  - A test plants sentinel ground-truth values and checks they never appear in any agent prompt.
  - A static test checks that `agents/` and `selfcheck/` never reference the protected store.
  - The history agents see excludes shadow measurements.
- **Superseded run:** an earlier run in which agents saw ground-truth-derived development summaries (`v1`) is archived in `artifacts/superseded/v1_gt_feedback/`, with an explanation.

## Quick start

```bash
uv sync --python 3.12
cp .env.example .env   # add CRUSOE_API_KEY
uv run python scripts/preflight.py          # live model discovery + Nemotron JSON + multimodal checks
uv run python -m evoaction run-demo         # baselines -> evolve -> sealed final test -> report
open artifacts/report.html
```

Individual stages: `python -m evoaction baselines | evolve | final | report` (add `--full` to use the full splits instead of the pilot subsets). Tests: `uv run --group dev pytest -q`.

## Models (Crusoe Managed Inference, confirmed live via `GET /v1/models`)

| Role | Model |
|---|---|
| Supervisor, investigators, strategy, critic | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Visual annotation, boundary refinement, frame inspection | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` |

Near-miss IDs in `.env` are resolved to the live ID; for example, `nvidia/Nemotron-3-Super-120B-A12B` resolves to `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`. `nvidia/NVIDIA-Nemotron-3.5-Super-VL` was also probed, but it accepts at most one image per request, so it cannot take multi-sheet input.

## Data and splits

- [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) (CC-BY-NC-SA-4.0). The parquet holds one episode per row group, so episodes are range-read individually instead of downloading the full ~1.4 GB file.
- Ingest separates **agent-visible** media, instructions and robot telemetry (`data/cache/episodes/`) from **evaluator-only** reference labels (`data/cache/protected/`). Only `evoaction/evaluation/protected.py` reads the protected directory.
- `artifacts/splits.json` is immutable and hashed: robot episodes only (DROID + Galaxea, which have telemetry), at most 60 s and 30 MB each, at least 3 reference subtasks. Near-duplicate instructions are grouped so a group never crosses splits. The full splits are 5 dev / 4 validation / 6 test; the pilots are 2 / 2 / 3. Each split and each pilot contains both robot sources and events shorter than 2 s. The seed rule was chosen from episode composition only, before any inference on these splits.

## Evaluation protocol (`evoaction-temporal-v1`)

- Temporal IoU with **maximum-cardinality one-to-one bipartite matching** on the IoU ≥ 0.75 adjacency (`scipy.sparse.csgraph.maximum_bipartite_matching`). It is neither greedy nor max-IoU-then-threshold.
- Primary metric: **micro Segment F1@0.75**, plus paired per-episode F1, short-event recall (reference subtasks under 2 s), and invalid/empty output rates.
- Report-only metrics: F1@0.5, mean boundary error in seconds, and an evaluator-only **outer-boundary-snapped** variant. The raw score is always reported separately. Snapped predictions are never written back.
- Temporal F1 measures boundaries only. **It does not assess whether subtask labels are semantically correct.**
- **Access policy (v2):** agents receive nothing derived from labels on any split. The evaluator scores dev and validation only for shadow measurement. The test split stays sealed until A/B/C are frozen with checksums, then is evaluated exactly once, on the 3 test episodes never evaluated before.

## Workflow graph (the thing that evolves)

The operator allowlist is Pydantic-validated, and agents can only submit typed `WorkflowPatch` objects. No generated code is ever executed.

| Operator | Behaviour |
|---|---|
| `sample_contact_sheets` | Uniform frames, tiled into timestamped contact sheets |
| `initial_annotate` | One Crusoe multimodal call over all sheets, using the versioned `prompt.md` |
| `state_hint_detection` | Deterministic gripper and end-effector change points from robot telemetry |
| `inspect_transition_windows` | Close-up frame strips around selected boundaries or hints |
| `refine_intervals` | One Crusoe multimodal call that keeps, moves, deletes or inserts boundaries, using `refine.md` |
| `merge_filter_segments` | Deterministic overlap and gap repair, minimum duration, full-episode tiling |
| `validate_export` | Schema and monotonicity checks, then export (the single terminal node) |

The validator enforces: at most 8 nodes, no cycles, exactly one annotate node, at most 2 multimodal passes per interval, every input produced upstream, and a single terminal export. Each version directory under `artifacts/versions/` contains `workflow.json`, `prompt.md`, an optional `refine.md`, `config.json`, `manifest.json` with SHA-256 hashes, and, once evaluated, `experiment.json` and `validation_report.json`. Version files are frozen after evaluation.

## Promotion rule (applied by the controller, never by an agent)

The rule is label-free (see the protocol above) and predeclared before the v2 run. It is an operational heuristic, not statistical significance.

## Budgets and safety

Call, spend and time budgets are set in `.env`: `MAX_NEMOTRON_CALLS`, `MAX_VISION_CALLS`, `MAX_SPEND_USD`, `RESERVED_TEST_VISION_CALLS` and others. They are checked before every request, and denials are persisted as budget decisions. Recursion is bounded at depth 3, 2 children per parent and 12 tasks, with at most one like-for-like retry. Results are cached on model, full input, prompt and node configuration, so identical upstream calls are reused and candidates are compared on common inputs. Credentials are never logged.

## Decisions and deviations, recorded honestly

- **Fixed baselines use `thinking=false`.** With thinking on, 31 of 42 thinking-length multi-image vision calls ran out of tokens while reasoning and never produced JSON. Setting `reasoning_effort` low or medium did not limit this, and these calls took 40–170 s. Thinking-off calls returned valid JSON in 2–11 s but tend toward evenly spaced boundaries. Two earlier baseline configurations (8k and 16k token caps with thinking on) are archived under `artifacts/versions/_superseded/` and marked superseded. They were retired before any agent optimization or validation-based selection. `thinking` remains an operator parameter the agents can change, and they are told this measured trade-off.
- Default `REQUEST_TIMEOUT_SECONDS` is 180, because reasoning calls can exceed 90 s.
- WGO-Bench's official scorer could not be verified, so the outer-snapped score is labeled as an evaluator-only variant.
- The pilot splits are small, so results are **feasibility evidence**, not established generalization.
- The v2 evolve run used `MAX_VISION_CALLS=96` because label-free visual verification adds one vision call per assessed episode. Total spend stays around $1–2 of the $35 cap.
- I stopped at the predeclared 2 experiments. A third was allowed by budget, but I had already seen the shadow ground-truth measurements, and running more experiments because of them would let ground truth steer the process.

## Layout

```
scripts/preflight.py          live Crusoe + data preflight -> artifacts/preflight.json
evoaction/clients/crusoe.py   model discovery, budgets, cache, usage
evoaction/data/wgo.py         WGO ingest with gold separation, locked splits
evoaction/media/              deterministic contact sheets
evoaction/evaluation/         pure metrics + protected evaluator (only gold reader)
evoaction/workflow/           operators, graph validation/execution, version store, A/B baselines
evoaction/agents/             tools, recursive investigator task graph, Nemotron supervisor/strategy/critic
evoaction/experiments/        evolution controller (gate, promotion, rollback), sealed final evaluation
evoaction/report.py           static HTML evidence report
artifacts/                    preflight, splits, versions, lineage, evolution, final_test, report.html
```

Apache-2.0. WGO-Bench data is CC-BY-NC-SA-4.0 and is not redistributed here.
