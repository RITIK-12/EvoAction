# EvoAction

**Self-improving agentic harness for data annotation.** Built for the Crusoe × Cerebral Valley Recursive Agents Hackathon (Track 3: Version N+1). Runs entirely on NVIDIA Nemotron models served by Crusoe Managed Inference.

## Problem

Physical AI needs robot video labeled into subtasks: when each step starts, when it ends, and what it is. Labeling by hand is slow and expensive. Off-the-shelf VLMs do poorly: on WGO-Bench the best published model reaches a Segment F1 of 0.29. Prompts and pipelines get tuned by hand, one round at a time.

## Solution

A harness that improves its own annotation pipeline the way an annotation team does: label a small sample, find the mistakes, fix the process, run again.

```
N  ──run on all episodes──▶ predictions
        │
        ▼
root-cause analysis on a fixed 10% human-labeled sample
(segment counts and lengths, 1-second-grid fraction, coverage, best IoU per human segment)
        │
        ▼
main agent (Nemotron-3-Super-120B) ranks the root causes and writes a typed WorkflowPatch:
rewrite the prompt, change parameters, or add/remove pipeline operators
        │
        ▼
try on the labeled sample ─▶ refine (up to 3 rounds) ─▶ best round runs on all episodes
        │
        ▼
gate (labeled-sample F1 must improve, every episode must produce valid output) ─▶ N+1
```

- **Annotator:** `Nemotron-3-Nano-Omni-30B-A3B`, reading timestamped contact sheets of the video.
- **N0** replicates Macrodata's published annotation protocol exactly: same prompt, sampling, tiles and timestamp badge.
- **Operators the agent can wire together:** contact-sheet sampling, annotate, telemetry hint detection, close-up boundary windows, refine, and merge/filter. The pipeline is a validated DAG, and no generated code is ever executed.
- **Versions** are immutable and hashed. Rejected versions are kept. Every model call, decision and score is logged to SQLite and JSONL.
- The same 10% labeled sample is reused every recursion. The other 90% is never labeled for the agent or the gate; it's held out and gives the honest score.

## Dataset

[WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) by Macrodata: 100 robot manipulation episodes (50 DROID, 25 Galaxea, 25 HomER) with 743 human-annotated subtask segments. Licence CC-BY-NC-SA-4.0, not redistributed here.

Metric (Macrodata's): **Segment F1 at temporal IoU ≥ 0.75**, pooled over episodes, with the first start and last end snapped to the human episode boundaries.

## Results: DROID, 50 episodes

| Version | What the agent changed | Segment F1@0.75, held-out 45 | Segment F1@0.75, all 50 |
|---|---|---|---|
| N0 | Macrodata protocol on Nemotron Omni | 0.069 | 0.081 |
| N1 | added telemetry hints → close-up windows → refine → merge (≥ 2 s, full coverage) | 0.066 | 0.086 |
| **N2** | rewrote the annotate and refine prompts ("most segments 3–12 s"), 30 frames per sheet, refine every boundary, merge ≥ 3 s and join same-label neighbours | **0.182** | **0.196** |

- N2 is 2.6× N0 on the held-out episodes.
- **The gate rejected N2:** it passed the labeled-sample check (0.381 vs 0.276), but 1 of 50 episodes returned invalid JSON, so the strict valid-output rule rejected it. The gated incumbent is still N1.
- Re-running N1 gave 0.079 on held-out, so expect about ±0.015 of run-to-run noise.
- The Galaxea + HomER batch is still running.

## Compared with Macrodata's model sweep

Macrodata's original sweep: Segment F1@0.75, contact-sheet setup, all 100 episodes. **Our row is DROID only (45 held-out episodes)**, because the second batch isn't finished.

| # | System | Segment F1@0.75 |
|---|---|---|
| 1 | Gemini 3.5 Flash | 0.290 |
| 2 | Gemini Robotics ER 1.6 | 0.245 |
| 3 | GPT-5.5 (low) | 0.233 |
| 4 | GPT-5.4 (low) | 0.233 |
| 5 | Gemini 3.1 Pro Preview | 0.232 |
| 6 | Gemini 3 Flash Preview | 0.210 |
| 7 | **EvoAction N2 (Nemotron Omni 30B-A3B + Nemotron Super 120B)** | **0.182** |
| 8 | Claude Opus 4.8 | 0.168 |
| 9 | Claude Sonnet 4.6 | 0.146 |
| 10 | Gemini 2.5 Pro | 0.145 |
| 11 | Qwen 3.5 Plus | 0.141 |
| 12 | Claude Haiku 4.5 | 0.128 |
| 13 | Gemini 3.1 Flash Lite | 0.119 |
| 14 | Qwen 3.6 Flash | 0.110 |
| 15 | GPT-5.4 mini | 0.106 |
| 16 | Gemini 2.5 Flash | 0.059 |
| 17 | Gemma 4 26B | 0.057 |

## Run it

```bash
uv sync --python 3.12
cp .env.example .env                                  # add CRUSOE_API_KEY
mkdir -p data/wgo && curl -L -o data/wgo/annotations.parquet \
  https://huggingface.co/datasets/macrodata/WGO-Bench/resolve/main/data/annotations.parquet
uv run python -m evoaction setup                      # ingest 100 episodes, lock batches + 10% labeled samples
uv run python -m evoaction baseline --set droid50     # N0
uv run python -m evoaction evolve --set droid50 --hitl   # N -> N+1 -> N+2
uv run python -m evoaction report                     # artifacts/report.html
uv run python -m evoaction demo                       # live UI: pick an episode, watch N0 -> N1 -> N2 label it
```

## Layout

```
evoaction/
  clients/crusoe.py        Crusoe API client: model discovery, budgets, cache, retries
  data/wgo.py              dataset ingest (labels kept apart), batches, 10% labeled samples
  media/contact_sheets.py  timestamped contact sheets
  workflow/                operators, DAG validation and execution, immutable version store
  agents/                  main agent (root causes + patches), investigators, critic
  experiments/evolution.py recursion controller: root-cause analysis, refinement rounds, gate, promotion
  evaluation/              Segment F1 metrics; the protected evaluator is the only code that reads labels
  selfcheck/               label-free self-assessment (used in the no-label comparison runs)
  demo.py                  live demo server (own DB under data/demo_runs/)
  report.py                HTML evidence report
scripts/                   preflight, completing an interrupted candidate
artifacts/                 episode sets, lineages, versions, runs, events.jsonl, evoaction.db, report.html
```

Apache-2.0.
