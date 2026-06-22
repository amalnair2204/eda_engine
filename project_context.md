# Project Context & Handover — AI-Accelerated EDA Placement & Routing Engine

> **Purpose:** Single-document briefing to onboard a fresh AI instance (or human) onto this codebase. Read this together with `CLAUDE.md` (the authoritative architecture/conventions file).
>
> **Last updated:** 2026-06-22
> **Repo:** `github.com/amalnair2204/eda_engine` (branch `main`, CI passing)
> **Latest commit at handover:** `14b0c1f`

---

## 1. What This Project Is

A **pure-software, end-to-end AI-accelerated EDA (Electronic Design Automation) placement & routing engine**, written entirely in Python. There is **no physical hardware**. It turns a plain-English circuit description into a placed, routed, electrically-analyzed, and fabrication-ready PCB layout, served as a full-stack web app with a grounded design copilot.

It is a **portfolio piece** demonstrating: LLM orchestration, reinforcement learning, classical optimization (genetic algorithms, maze routing), multi-objective optimization, RAG, and full-stack delivery + deployment.

---

## 2. System Architecture

12-phase pipeline. Each phase exposes a `run_phaseN(input) -> output` function (Pipeline pattern) so the FastAPI layer can call them uniformly. The `CircuitGraph` object is the backbone and is treated as immutable except for component `x`/`y` positions.

```
Plain-English prompt
  → Phase 0   Groq (LLaMA 3.3 70B / gpt-oss)  →  JSON netlist
  → Phase 1   Parser + CircuitGraph (typed dataclasses)
  → Phase 2/7 Placement   ┌ Genetic Algorithm   (slow, near-optimal global search)
                          └ RL Agent (MaskablePPO) (fast amortized inference)
  → Phase 3/8 Routing     ┌ Single-layer Lee's maze router
                          └ Multi-layer + vias  (drives same-layer crossings → 0)
  → Phase 4   Analytics (HPWL, per-layer crossings, vias, parasitics, DRC/ERC)
  → Phase 9   Manufacturing export (Gerber RS-274X · Excellon · BOM · KiCad netlist)
  → Phase 5/6 FastAPI + WebSocket web app @ localhost:8000
  → Phase 10  RAG design copilot (grounded, cited answers over datasheets)
  → Phase 11  Design-space exploration (Pareto-optimal layout search)
  → Phase 12  Docker + CI/CD deployment
```

### Design patterns in use
- **Strategy** — placers (`ga` | `rl`) and routers (`single` | `multi`) are swappable behind a common interface; selectable at runtime from the UI.
- **Factory** — `NetlistParser`.
- **Adapter** — `CircuitGraph`.
- **Pipeline** — `run_phaseN()` functions.
- **Observer** — Phase 4 analytics watches Phase 3 output.

### Core data models (defined in `phase1_eda_engine.py`, imported everywhere — never redefined)
`Pin`, `Component`, `Net`, `GridMetadata`, `Netlist`, `CircuitGraph`, `GraphEdge`.

### Tech stack
| Layer | Tools |
|-------|-------|
| LLM | Groq API (LLaMA 3.3 70B; gpt-oss-120b supported) |
| RL | PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib (MaskablePPO) |
| Optimization & graph | NumPy, NetworkX |
| Routing | Custom Lee's / A* on 2D/3D integer grid |
| Manufacturing | gerbonara (Gerber / Excellon / render) |
| RAG | sentence-transformers (embeddings), ChromaDB (vector store), pypdf |
| API / UI | FastAPI, Uvicorn, WebSockets, HTML/CSS/JS |
| Viz & tests | Matplotlib (Agg), Pytest |
| Deploy | Docker (python:3.12-slim), GitHub Actions CI |

---

## 3. Features Fully Implemented

All 12 phases complete and tested. **122 pytest tests, all passing. CI green on `main`.**

| # | Feature | File(s) | Notes |
|---|---------|---------|-------|
| 0 | Groq LLM → JSON netlist | `phase0_groq_translator.py` | JSON mode, `max_tokens=4096`, finish_reason logging, auto-retry when `nets` missing |
| 1 | Parser + CircuitGraph + visualizer | `phase1_eda_engine.py` | Core typed models live here |
| 2 | Genetic Algorithm placer | `phase2_genetic_placer.py` | Minimizes HPWL |
| 3 | Single-layer Lee's/A* maze router | `phase3_router.py` | Note: mutates `graph.metadata` on grid expansion |
| 4 | Analytics engine | `phase4_analytics.py` | HPWL, crossings, parasitics, DRC/ERC; `AnalyticsEngine(...).compute()` is read-only |
| 5 | UI shell | `frontend/` | HTML/CSS/JS |
| 6 | FastAPI integration | `app.py` | REST + WebSocket, server-side `_LAST_ROUTED` cache |
| 7 | RL placement agent (PPO) | `phase7_rl_placer.py`, `train_phase7_rl.py` | Trained model at `models/phase7_rl_placer.zip` |
| 8 | Multi-layer router + vias | `phase8_multilayer_router.py` | 3D grid, via-cost minimization |
| 9 | Manufacturing export | `phase9_export.py` | Per-board Gerber/Excellon/BOM/KiCad; fab `.zip` bundles all four |
| 10 | RAG design copilot | `phase10_rag_copilot.py`, `ingest_knowledge.py` | Cited answers; refuses specs not in KB |
| 11 | Design-space exploration | `phase11_explorer.py` | Sweeps placer×router×params, Pareto front, recommendation |
| 12 | Docker + CI/CD | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml` | Secrets via runtime env only |

### API endpoints (`app.py`)
`GET /` · `GET /health` · `POST /generate` · `WS /ws/generate` · `POST /copilot` · `WS /copilot/stream` · `POST /export` · `POST /explore` · `GET /export/download/{filename}` · `GET /outputs/{filename}` · `GET /netlists/generated/{filename}`

### Phase 11 detail (most recent feature work)
- Objectives minimized: **HPWL, crossings, trace length, runtime**. Candidates with routing completion < 100% are treated as dominated/invalid.
- Non-dominated sorting → Pareto set; recommends a tradeoff from that set.
- Outputs: `outputs/phase11_pareto.png` (scatter) + `outputs/phase11_results.md` (table).
- Deep-copies the graph per candidate (Phase 3 mutates metadata) to protect siblings/caller.
- Uses `AnalyticsEngine(...).compute()` directly (no file writes per candidate).

### Phase 12 detail (just shipped)
- `Dockerfile`: `python:3.12-slim`, layered install (requirements before source), `libgomp1` for torch, `HF_HOME` cache for sentence-transformers, env-based `HOST`/`PORT`, **no secrets baked in**.
- `docker-compose.yml`: single `eda-engine` service, port 8000, `GROQ_API_KEY`/`GROQ_MODEL` from host env or optional local `.env` (`required: false`).
- `.github/workflows/ci.yml`: push/PR to `main` → Python 3.12, cached pip, lenient ruff (`--select E9,F63,F7,F82`), `python -m pytest`. **No secrets**; a dummy `GROQ_API_KEY` is set because all Groq calls in tests are mocked.
- `app.py` gained an env-based `_host_port()` helper + `__main__` block (hosting only — no phase logic touched).

---

## 4. Currently Working On

**Nothing in flight.** Phases 0–12 are complete, committed, pushed, and CI-verified. The codebase is at a stable checkpoint. The most recent activity was administrative: committing/pushing the whole project, updating the `CLAUDE.md` phase tracker, and verifying CI + README badge are green.

---

## 5. Known Bugs / Caveats

No open functional bugs. Items to be aware of:

- **Groq free tier rate limit.** The free `on_demand` tier allows only ~8000 tokens/minute. `max_tokens` is capped at **4096** for this reason. On `finish_reason="length"` the translator logs a hint to simplify the prompt or upgrade the tier — it does not silently fail.
- **`gpt-oss-120b` reasoning field.** Phase 0 parses only `message.content` (the JSON), never the separate `message.reasoning` field. A single automatic retry fires if the parsed netlist is missing/empty `nets`.
- **RL placer requires a trained model.** `RLPlacer` raises `FileNotFoundError` without `models/phase7_rl_placer.zip` (present in repo, but gitignored — see below). Phase 11 catches this and skips RL candidates gracefully.
- **`models/` and `vectorstore/` are gitignored** (binary/runtime artifacts). A fresh clone must retrain (`python -m train_phase7_rl`) or it will skip RL, and rebuild the copilot KB (`python ingest_knowledge.py`).
- **`netlists/generated/` is now gitignored** (runtime Groq outputs); previously-tracked files were removed from the index.
- **`phase3_router` mutates `graph.metadata`** on grid expansion — any caller running multiple layouts off one graph must deep-copy first (Phase 11 already does).
- **Large Docker image** (torch + sentence-transformers). First request may be slow while the embedding model downloads into `HF_HOME`.
- **Windows dev note:** use `python -m pytest` (not bare `pytest`); `gh` CLI is not installed on this machine — CI status was checked via the GitHub API/badge.

---

## 6. Next Immediate Steps (Suggested)

None are required for correctness; these are natural follow-ons:

1. **Demo assets** — README references `docs/ui.png`, `docs/routing.png`, `docs/gerber_preview.png` (commented out). Capture and add screenshots, uncomment the Demo section.
2. **Surface Phase 11 in the UI more richly** — the "Explore design space" button exists; could render the Pareto scatter (`phase11_pareto.png`) inline rather than just the table.
3. **Deployment dry-run** — actually deploy the container to a PaaS (Render/Railway/Fly.io) and confirm `$PORT` injection + `GROQ_API_KEY` secret wiring end-to-end.
4. **Consider committing a small pre-trained RL model** or documenting the retrain step more prominently, so cloners don't silently lose RL placement.

---

## 7. How to Run

```bash
# Local
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
copy .env.example .env                            # set GROQ_API_KEY
python ingest_knowledge.py                        # build copilot KB (one-time)
python -m uvicorn app:app --reload                # → http://localhost:8000

# Docker (secret passed at runtime, never baked in)
docker build -t eda-engine .
docker run --rm -p 8000:8000 -e GROQ_API_KEY=your_groq_key eda-engine

# Tests
python -m pytest            # 122 tests
```

**Hard constraints for future work (from project conventions):** never commit `.env`; never bake secrets into the image; never redefine the core data models outside `phase1_eda_engine.py`; never merge two phases into one file; update the `CLAUDE.md` Phase Status Tracker at the end of every phase.
