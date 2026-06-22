# CLAUDE.md — AI-Accelerated EDA Placement & Routing Engine
> This file is read automatically by Claude Code at the start of every session.
> It is the single source of truth for architecture, conventions, and phase status.

---

## Project Overview
A pure-software, end-to-end AI-accelerated Electronic Design Automation (EDA)
Placement and Routing Engine built entirely in Python. There is NO physical
hardware involved. This is a portfolio piece demonstrating mastery of:
- LLM orchestration (Groq API)
- Spatial optimization (Genetic Algorithm)
- Pathfinding (Lee's Algorithm / A*)
- Software design patterns at scale

---

## End-to-End Pipeline
```
Plain English Prompt
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 0 — Groq API Translator  │  English → JSON Netlist
│  phase0_groq_translator.py      │  Groq SDK (LLaMA 3.3 70B)
└─────────────────────────────────┘
        │  structured JSON netlist
        ▼
┌─────────────────────────────────┐
│  PHASE 1 — Parser & Graph       │  JSON → typed Python objects → graph
│  phase1_eda_engine.py           │  NetlistParser, CircuitGraph, InitialPlacer
└─────────────────────────────────┘
        │  CircuitGraph object
        ▼
┌─────────────────────────────────┐
│  PHASE 2 — Genetic Algorithm    │  Spatial placement optimizer
│  phase2_genetic_placer.py       │  Minimizes HPWL fitness function
└─────────────────────────────────┘
        │  optimized component positions
        ▼
┌─────────────────────────────────┐
│  PHASE 3 — Maze Router          │  Plots copper trace paths
│  phase3_router.py               │  Lee's Algorithm / A* variant
└─────────────────────────────────┘
        │  routed trace paths
        ▼
┌─────────────────────────────────┐
│  PHASE 4 — Analytics Engine     │  Computes EEE metrics
│  phase4_analytics.py            │  HPWL, crossings, parasitic capacitance
└─────────────────────────────────┘
        │  metrics dict
        ▼
┌─────────────────────────────────┐
│  PHASE 5 — UI Shell             │  Designed in Claude Design
│  frontend/                      │  Exported as HTML/CSS
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 6 — Full Integration     │  FastAPI backend + wired frontend
│  app.py                         │  Claude Code integration
└─────────────────────────────────┘
```

---

## Phase Status Tracker
| Phase | Name                          | File                        | Status        |
|-------|-------------------------------|-----------------------------|---------------|
| **0** | Groq API Translator           | phase0_groq_translator.py   | ✅ Complete    |
| 1     | Parser + Graph + Visualizer   | phase1_eda_engine.py        | ✅ Complete    |
| 2     | Genetic Algorithm Placer      | phase2_genetic_placer.py    | ✅ Complete    |
| 3     | Lee's / A* Maze Router        | phase3_router.py            | ✅ Complete    |
| 4     | Analytics Engine              | phase4_analytics.py         | ✅ Complete    |
| 5     | UI Shell (Claude Design)      | frontend/                   | ✅ Complete    |
| 6     | Full Integration              | app.py                      | ✅ Complete    |
| 7     | RL Placement Agent (PPO)      | phase7_rl_placer.py         | ✅ Complete    |
| 8     | Multi-Layer Router + Vias     | phase8_multilayer_router.py | ✅ Complete    |
| 9     | Manufacturing Export          | phase9_export.py            | ✅ Complete    |
| 10    | RAG Design Copilot            | phase10_rag_copilot.py      | ✅ Complete    |
| 11    | Design-Space Exploration      | phase11_explorer.py         | ✅ Complete    |
| 12    | Docker + CI/CD Deployment     | Dockerfile, .github/        | ✅ Complete    |

**Update this table at the end of every phase.**

---

## Project File Structure
```
eda_engine/
├── CLAUDE.md                        ← this file (Claude Code reads on startup)
├── README.md                        ← human-facing project docs
├── requirements.txt                 ← all pip dependencies
├── .env                             ← API keys (never commit this)
├── .env.example                     ← safe template to commit
├── .gitignore
│
├── phase0_groq_translator.py        ← Phase 0: LLM → JSON netlist
├── phase1_eda_engine.py             ← Phase 1: Parser, Graph, Visualizer
├── phase2_genetic_placer.py         ← Phase 2: GA optimizer
├── phase3_router.py                 ← Phase 3: Maze router
├── phase4_analytics.py              ← Phase 4: EEE metrics
│
├── netlists/
│   ├── sample_netlist.json          ← mock Groq API output (ESP32 + LED)
│   └── generated/                   ← runtime Groq outputs land here
│
├── outputs/
│   ├── phase1_output.png
│   ├── phase2_output.png
│   └── phase3_output.png
│
├── frontend/                        ← Phase 5: Claude Design export
│   ├── index.html
│   ├── styles.css
│   └── canvas.js
│
├── tests/
│   ├── test_phase0.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
│
└── app.py                           ← Phase 6: FastAPI integration server
```

---

## Core Data Models
> These are defined in phase1_eda_engine.py and imported by ALL other phases.
> NEVER redefine them elsewhere. NEVER mutate Pin or Net after parsing.

```
Pin          — id, pin_type, net, abs_x, abs_y
Component    — id, comp_type, name, pins, footprint(w,h), x, y, properties
Net          — id, net_type, connected_pins[(comp_id, pin_id)]
GridMetadata — width, height, unit
Netlist      — metadata, components[], nets[]
CircuitGraph — nodes{id:Component}, edges[GraphEdge], adjacency{id:set}
GraphEdge    — net_id, net_type, source(comp,pin), target(comp,pin), weight
```

**Phase handoff contract:**
- Phase 0 → Phase 1: raw JSON dict (or string)
- Phase 1 → Phase 2: CircuitGraph object (placement at seed positions)
- Phase 2 → Phase 3: CircuitGraph object (placement optimized, HPWL minimized)
- Phase 3 → Phase 4: CircuitGraph + list of routed trace paths
- Phase 4 → Phase 6: metrics dict {hpwl, crossings, capacitance, trace_length}

---

## Tech Stack
| Layer         | Library/Tool          | Purpose                          |
|---------------|-----------------------|----------------------------------|
| LLM           | groq                  | Groq API calls (LLaMA 3 etc.)    |
| Data models   | dataclasses           | Core typed models                |
| Graph         | built-in + networkx   | CircuitGraph + export            |
| Optimization  | numpy                 | GA fitness calculations          |
| Routing       | built-in              | Lee's / A* on 2D int grid        |
| Visualization | matplotlib            | All phase output canvases        |
| API server    | fastapi + uvicorn     | Phase 6 backend                  |
| Env vars      | python-dotenv         | API key management               |
| Testing       | pytest                | All phases                       |

---

## Environment Variables (.env)
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GRID_WIDTH=24
GRID_HEIGHT=20
GRID_UNIT=mm
GA_GENERATIONS=200
GA_POPULATION=50
```

---

## Coding Conventions
1. **One class per logical concern.** Never mix parsing logic with graph logic.
2. **Design patterns to honour:**
   - Factory → NetlistParser
   - Adapter → CircuitGraph
   - Strategy → Placer, Router (swappable algorithms)
   - Pipeline → run_phaseN() functions
   - Observer → Phase 4 analytics watches Phase 3 output
3. **Every phase exposes a `run_phaseN(input) -> output` function** — this is
   what Phase 6 (FastAPI) will call.
4. **No hardcoded paths.** Use `pathlib.Path` everywhere.
5. **All outputs** (PNGs, generated netlists) go into `outputs/` or
   `netlists/generated/` — never in the project root.
6. **Docstrings on every class and public method.**
7. **Type hints on every function signature.**
8. **Never commit `.env`.** Always update `.env.example` instead.

---

## EEE Domain Constraints (Engineering Rules)
The engine must respect these real electrical engineering rules:
- Power nets (VCC, 3V3, 5V) must never cross Ground nets
- Signal traces must not run parallel for more than 3 grid cells (crosstalk risk)
- Decoupling capacitors must be placed within 2 grid cells of their IC's power pin
- High-frequency components (MCU, IC) should be placed away from the grid edges
- Minimum trace separation: 1 grid cell (represents DRC clearance rule)

---

## How to Run Each Phase (Quick Reference)
```bash
# Phase 0
python phase0_groq_translator.py --prompt "Connect an ESP32 to an LED via a resistor"

# Phase 1
python phase1_eda_engine.py

# Phase 2 (after Phase 1)
python phase2_genetic_placer.py

# Phase 3 (after Phase 2)
python phase3_router.py

# Phase 4 (after Phase 3)
python phase4_analytics.py

# All tests
pytest tests/ -v

# Phase 6 (full stack)
uvicorn app:app --reload
```

---

## Project Complete
All 6 phases implemented and tested. Run with `uvicorn app:app --reload`.

---

## Notes for Claude Code
- Always read this CLAUDE.md at the start of every session before touching any file.
- Always check the Phase Status Tracker before starting work.
- Never skip a phase or merge two phases into one file.
- After completing any phase, update the Status Tracker in this file.
- After completing any phase, tell the user exactly how to test it manually.
- The `CircuitGraph` object is the backbone of this entire project. Treat it
  as immutable except for `Component.x` and `Component.y` values.