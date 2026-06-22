# AI-Accelerated EDA Placement & Routing Engine

> Turn a plain-English circuit description into an optimized, fully-routed,
> manufacturable PCB layout — through an 11-stage AI pipeline, served as a
> full-stack web app with a grounded design copilot.

[![CI](https://github.com/amalnair2204/eda_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/amalnair2204/eda_engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)
![RL](https://img.shields.io/badge/RL-PPO%20(SB3)-red)
![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Describe a circuit in natural language — *"an ESP32 reading three analog sensors
through an ADC"* — and the engine returns a placed, routed, electrically-analyzed,
and fabrication-ready PCB. It pairs an LLM netlist generator with both classical
and learned optimization: a genetic-algorithm placer **and** a reinforcement-learning
placement agent, single- and multi-layer maze routing with via insertion,
industry-standard manufacturing export, and a retrieval-augmented copilot that
answers design questions grounded in real datasheets.

This is a pure-software portfolio project — no physical hardware — built to
demonstrate LLM orchestration, reinforcement learning, classical optimization,
and full-stack delivery in one coherent system.

## Demo

<!-- Add screenshots to docs/ and uncomment:
![UI](docs/ui.png)
![Multi-layer routing](docs/routing.png)
![Gerber preview](docs/gerber_preview.png)
-->

## Pipeline

```
Plain-English prompt
  → Phase 0   Groq (LLaMA 3.3 70B)         →  JSON netlist
  → Phase 1   Parser + CircuitGraph
  → Phase 2/7 Placement      ┌ Genetic Algorithm  (near-optimal global search)
                             └ RL Agent (PPO)      (fast amortized inference)
  → Phase 3/8 Routing        ┌ Single-layer Lee's maze router
                             └ Multi-layer + vias  (0 same-layer crossings)
  → Phase 4   Analytics (HPWL, per-layer crossings, vias, parasitics, DRC/ERC)
  → Phase 9   Manufacturing export (Gerber RS-274X · Excellon · BOM · KiCad)
  → Phase 5/6 FastAPI + WebSocket web app at localhost:8000
  → Phase 10  RAG design copilot (grounded, cited answers over datasheets)
```

Placement strategy (`ga` | `rl`) and router (`single` | `multi`) are selectable
at runtime — they share a common Strategy interface, so each is a drop-in
alternative.

## Placement Benchmark — Learned vs Classical

Measured on the sample netlist, comparing all three placers head-to-head:

| Placer | HPWL | Overlaps | Time (s) | Routing completion |
|--------|-----:|---------:|---------:|-------------------:|
| Random | 104.5 | 0 | 0.000 | 100.0% |
| Genetic Algorithm | **33.0** | 0 | 1.039 | 86.1% |
| RL Agent (PPO) | 41.0 | 0 | **0.021** | 88.9% |

**Takeaway:** the RL policy learns placement — **~61% lower wirelength than random**
— and runs **~50× faster than the GA** (0.021s vs 1.04s) by trading ~24% HPWL.
This is the core tradeoff in modern EDA: a metaheuristic does a slow, near-optimal
per-instance search, while a learned policy pays its cost once during training and
then places almost instantly. The GA wins on raw quality; the RL agent wins on
amortized speed.

## Key Features

- **Dual placement engines** — genetic algorithm and a PPO reinforcement-learning
  agent (masked action space, sequential placement MDP), benchmarked head-to-head.
- **Multi-layer routing with vias** — Lee's algorithm extended to a 3D grid;
  layer-direction biasing and via-cost minimization drive same-layer crossings to
  zero on the sample netlist.
- **Manufacturing export** — generates Gerber RS-274X (one file per copper layer),
  Excellon drill files, a grouped BOM CSV, and a KiCad-importable netlist, bundled
  fab-ready. Validated by re-parsing every generated file.
- **RAG design copilot** — ask "suggest a lower-power alternative for U1" and get a
  cited answer grounded in a datasheet knowledge base plus your *current* design.
  Refuses to fabricate specs not in the knowledge base.
- **Deterministic guardrails** — datasheet-sourced pinouts (never LLM-inferred);
  auto-injected decoupling caps; DRC/ERC checks.
- **Full-stack** — FastAPI + WebSocket backend, browser UI, real-time progress.

## Tech Stack

| Layer | Tools |
|-------|-------|
| LLM | Groq API (LLaMA 3.3 70B) |
| Reinforcement learning | PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib (MaskablePPO) |
| Optimization & graph | NumPy, NetworkX |
| Routing | Custom Lee's / A* on a 2D/3D integer grid |
| Manufacturing | gerbonara (Gerber / Excellon / render) |
| RAG | sentence-transformers (embeddings), ChromaDB (vector store), pypdf |
| API / UI | FastAPI, Uvicorn, WebSockets, HTML/CSS/JS |
| Viz & tests | Matplotlib, Pytest (108 tests) |

## Getting Started

```bash
# Clone
git clone https://github.com/amalnair2204/eda_engine.git
cd eda_engine

# Environment
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# Configure — copy the template and add your Groq key
copy .env.example .env             # Windows  (cp on macOS/Linux)
# then edit .env and set GROQ_API_KEY

# Build the copilot's knowledge base (one time)
python ingest_knowledge.py

# Run
python -m uvicorn app:app --reload
# open http://localhost:8000
```

> `.env` holds a live Groq key and is gitignored. The Chroma vector store is a
> runtime artifact rebuilt by `ingest_knowledge.py`, so it is not committed either.

## Run with Docker

The image is Linux-based (`python:3.12-slim`) and reads `HOST`/`PORT` from the
environment. **Secrets are never baked into the image** — `GROQ_API_KEY` is
passed at runtime.

```bash
# Build
docker build -t eda-engine .

# Run — pass the Groq key as a runtime env var (NOT in the image)
docker run --rm -p 8000:8000 -e GROQ_API_KEY=your_groq_key eda-engine
# open http://localhost:8000

# Override the bind port if needed
docker run --rm -p 9000:9000 -e PORT=9000 -e GROQ_API_KEY=your_groq_key eda-engine
```

Or with Docker Compose (reads `GROQ_API_KEY` / `GROQ_MODEL` from your shell or a
local `.env`):

```bash
GROQ_API_KEY=your_groq_key docker compose up --build
```

The RAG embedding model downloads on first use into the container's
`HF_HOME` cache. To pre-build the vector store inside the container, run
`python ingest_knowledge.py` (e.g. `docker run --rm eda-engine python ingest_knowledge.py`
with a mounted volume, or as a one-off start step).

## Deployment

The container runs anywhere that hosts a Docker image (Render, Railway, Fly.io,
a VM, etc.). Generic recipe:

- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT` (the image's
  default `CMD` already does this; most PaaS inject `$PORT`).
- **Required env var:** `GROQ_API_KEY` (set as a secret in the host's dashboard —
  never commit it).
- **Optional env vars:** `GROQ_MODEL` (default `llama-3.3-70b-versatile`),
  `HOST` (default `0.0.0.0`), `PORT` (default `8000`).
- **Note:** the image is large (torch + sentence-transformers). Pick an instance
  with enough memory/disk; first request may be slow while the embedding model
  downloads.

## Usage

1. Open `localhost:8000`.
2. Enter a circuit description in plain English.
3. Pick a placement strategy (**GA** or **RL Agent**) and a router (**single** or
   **multi-layer**).
4. Watch the pipeline stream phase-by-phase progress over WebSocket.
5. Inspect the routed layout and the analytics report (electrical metrics +
   DRC/ERC).
6. Download fabrication files (Gerber / drill / BOM / KiCad) from the export button.
7. Ask the design copilot questions about your board — it answers with citations.

### Benchmarks & training
```bash
python -m benchmark_placement     # random vs GA vs RL  → outputs/
python -m benchmark_routing       # single vs multi-layer → outputs/
python -m train_phase7_rl         # retrain the RL policy → models/
python -m pytest tests/ -v        # full test suite
```

## Project Structure

```
eda_engine/
├── phase0_groq_translator.py     # LLM → JSON netlist
├── phase1_eda_engine.py          # Parser, CircuitGraph, core models + Strategy protocols
├── phase2_genetic_placer.py      # GA placement
├── phase3_router.py              # Single-layer maze router
├── phase4_analytics.py           # EEE metrics + DRC/ERC (per-layer)
├── phase7_rl_placer.py           # RL placement agent (PPO) + Gymnasium env
├── phase8_multilayer_router.py   # Multi-layer routing + vias
├── phase9_export.py              # Gerber / Excellon / BOM / KiCad export
├── phase10_rag_copilot.py        # Grounded RAG copilot
├── train_phase7_rl.py            # RL training entrypoint
├── ingest_knowledge.py           # Build the copilot's vector store
├── benchmark_placement.py        # GA vs RL vs random
├── benchmark_routing.py          # single vs multi-layer
├── app.py                        # FastAPI backend + WebSocket
├── frontend/                     # Web UI
├── knowledge/                    # Datasheet / design-rule corpus (RAG source)
├── models/                       # Trained RL policy
├── tests/                        # 108 pytest tests
├── CLAUDE.md                     # Architecture + conventions (single source of truth)
└── requirements.txt
```

## What This Demonstrates

- Combining an LLM with both classical (GA, maze routing) and learned (RL) methods
  in one production pipeline — and honestly benchmarking them against each other.
- Formulating PCB placement as a sequential-decision RL problem with a masked
  action space, in the spirit of RL chip-floorplanning research.
- Designing around LLM failure modes (hallucinated pins, fabricated specs) with
  deterministic guardrails and a grounded, cited RAG layer.
- End-to-end delivery: algorithm core → FastAPI service → real-time web UI →
  manufacturable output.

## License

MIT
