# EDA Engine — AI-Accelerated PCB Placement & Routing

A pure-software, end-to-end AI-accelerated Electronic Design Automation (EDA) tool.
Type a plain-English circuit description; the engine translates it into a validated
netlist, optimises component placement with a Genetic Algorithm, routes copper traces
with Lee's maze router, and reports real electrical properties — all from a web UI.

---

## Architecture

```
Plain English Prompt
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 0 — Groq API Translator  │  English → JSON Netlist (LLaMA 3.3 70B)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 1 — Parser & Graph       │  JSON → CircuitGraph (NetworkX)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 2 — Genetic Algorithm    │  Minimise HPWL over 200 generations
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 3 — Maze Router          │  Lee's Algorithm / BFS on 2D grid
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 4 — Analytics Engine     │  R, C, delay, DRC rule check
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 6 — FastAPI + Frontend   │  WebSocket streaming, SVG canvas
└─────────────────────────────────┘
```

---

## Tech Stack

| Layer        | Tool / Library       | Purpose                              |
|--------------|----------------------|--------------------------------------|
| LLM          | Groq SDK + LLaMA 3.3 | Prompt → JSON netlist                |
| Graph        | Built-in + NetworkX  | CircuitGraph adjacency model         |
| Optimisation | NumPy                | GA fitness (HPWL) calculations       |
| Routing      | Pure Python          | BFS maze router on 2D int grid       |
| Visualisation| Matplotlib           | Phase PNG outputs (dark-mode canvas) |
| Backend      | FastAPI + Uvicorn    | REST + WebSocket API                 |
| Frontend     | Vanilla HTML/CSS/JS  | SVG circuit renderer, live metrics   |
| Env vars     | python-dotenv        | API key management                   |
| Testing      | Pytest               | 52 tests across all phases           |

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd eda_engine

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your Groq API key:
#   GROQ_API_KEY=gsk_...
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Run

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## How to use

1. Type a plain-English circuit description in the **Circuit Prompt** box.
   Example: *"Connect an ESP32 to a status LED via a 330 ohm resistor"*
2. Click **Generate Circuit**.
3. Watch the 5-step pipeline progress in real time.
4. The SVG canvas renders the routed PCB layout when complete.
5. Metrics cards show resistance, capacitance, delay, and crossings.
6. The DRC panel lists any EEE rule violations.

---

## Running individual phases

```bash
# Phase 0 — Groq API translation
python phase0_groq_translator.py --prompt "Connect an ESP32 to an LED via a resistor"

# Phase 1 — Parse + graph (uses sample_netlist.json)
python phase1_eda_engine.py

# Phase 2 — GA placement optimizer
python phase2_genetic_placer.py

# Phase 3 — Maze router
python phase3_router.py

# Phase 4 — Analytics engine
python phase4_analytics.py

# All tests
pytest tests/ -v
```

---

## Phase descriptions

**Phase 0 — LLM Translation**
The Groq API (LLaMA 3.3 70B) converts a plain-English circuit description into
a validated JSON netlist.  A structured system prompt and two few-shot examples
enforce the schema; the output is validated before passing downstream.

**Phase 1 — Parser & Graph Builder**
Parses the JSON netlist into typed Python dataclasses (`Component`, `Pin`, `Net`).
Builds a `CircuitGraph` via star-expansion per net.  `InitialPlacer` assigns
non-overlapping seed positions respecting EEE edge-clearance rules.

**Phase 2 — Genetic Algorithm Placer**
Evolves 50 candidate layouts over 200 generations, minimising a weighted fitness
function: HPWL × 1.0, overlap penalty × 10.0, thermal penalty × 2.0, edge
proximity × 1.5, and a decoupling-cap proximity reward × −1.5.  Typical
HPWL improvement: 40–60%.

**Phase 3 — Maze Router**
Routes copper trace paths using Lee's Algorithm (BFS wavefront expansion +
backtrace).  Nets are prioritised POWER → GROUND → SIGNAL.  A three-level
detour strategy (normal → DRC-relaxed → full crossing) ensures connectivity
even in dense layouts.

**Phase 4 — Analytics Engine**
Computes per-trace DC resistance, parasitic capacitance (parallel-plate
microstrip model), and signal propagation delay (FR4 time-of-flight).  Runs
five EEE design-rule checks: wire crossings, trace length, decoupling cap
proximity, unrouted nets, and power trace width.

**Phase 6 — FastAPI Integration**
FastAPI serves the frontend SPA and exposes `POST /generate` (synchronous)
and `WebSocket /ws/generate` (streaming, per-phase progress messages).
The SVG canvas in the browser renders components, traces, and pins from
the `layout` JSON returned by the backend.

---

## Sample output

![Phase 3 Routed Layout](outputs/phase3_output.png)

---

## Environment variables

| Variable        | Default                    | Description           |
|-----------------|----------------------------|-----------------------|
| `GROQ_API_KEY`  | *(required)*               | Groq API key          |
| `GROQ_MODEL`    | `llama-3.3-70b-versatile`  | Groq model to use     |
| `GRID_WIDTH`    | `24`                       | Board grid columns    |
| `GRID_HEIGHT`   | `20`                       | Board grid rows       |
| `GA_GENERATIONS`| `200`                      | GA evolution cycles   |
| `GA_POPULATION` | `50`                       | GA population size    |
