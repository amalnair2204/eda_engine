# Handoff Prompt — AI-Accelerated EDA Placement & Routing Engine

> Paste this into a new chat with any AI assistant to continue seamlessly.

---

## Goal
Build, fix, and ship a portfolio project called **"AI-Accelerated EDA Placement & Routing Engine"** — a pure-software Python system that converts plain English circuit descriptions into optimized, fully-routed PCB layouts via a 5-phase AI pipeline, deployed as a full-stack local web app. Final goals: push to GitHub, polish for resume/portfolio use.

## Who I Am
Altrin — final-year EEE student at BITS Pilani Dubai, targeting Applied AI Developer roles in Dubai. GitHub: `altrin7311`. Workflow: I orchestrate Claude Code with structured prompts per phase, run them, then verify with pytest before moving on.

## Architecture (Pipeline)
```
Plain English Prompt
   → Phase 0: Groq API (LLaMA 3.3 70B) → JSON Netlist
   → Phase 1: NetlistParser + CircuitGraph + InitialPlacer
   → Phase 2: Genetic Algorithm placement optimizer (HPWL minimization)
   → Phase 3: Lee's Algorithm maze router
   → Phase 4: EEE Analytics Engine (resistance, capacitance, signal delay, DRC/ERC)
   → Phase 5: UI shell (Claude Design → frontend/index.html)
   → Phase 6: FastAPI backend (app.py) + WebSocket — web app at localhost:8000
```

## Current Status
- ✅ Phases 0–6 all complete, 30+ pytest tests passing
- ✅ Resume bullet points and summary finalized
- 🔄 A major fix prompt was given to Claude Code addressing:
  1. Pin accuracy (MCP3008 showing 7 pins vs 16 actual, wrong positioning)
  2. Wire crossings (36 → target 0)
  3. Routing completion (83.3% → target 100%, 3 unrouted nets)
  4. Missing decoupling cap warnings + power trace false positives
- ❓ UNCONFIRMED whether that fix prompt was run/passed yet — **biggest open thread**
- 🔄 GitHub push in progress — not confirmed complete. Last blocker: uvicorn occupying terminal, resolved by opening 2nd VS Code terminal

## Key Decisions
- LLM is **Groq** (groq.com, LLaMA 3.3 70B), **NOT Grok** (xAI) — corrected across all files
- Pin accuracy via static `component_library.py` (datasheet-sourced), not LLM-inferred
- Routing fixes are engine-level (2× grid resolution, rip-up-and-reroute, auto grid expansion) — apply to any circuit
- Groq system prompt forces all pins listed (unconnected = "NC") + auto-injects 100nF decoupling caps on every IC
- `CLAUDE.md` at project root is persistent Claude Code memory — every prompt starts "Read CLAUDE.md fully first"

## What to Avoid
- Don't confuse Grok vs Groq
- On this Windows machine use `python -m pytest` and `python -m uvicorn` (not bare commands — PATH issue)
- Never push `.env` (live Groq key) — verify it's absent after push
- Don't redesign UI when wiring backend; don't skip/merge phases

## Next Best Step
Confirm with me:
- **(a)** Did the Fix 1–5 prompt finish in Claude Code and hit targets (0 crossings, 100% completion, 16 labelled MCP3008 pins, no decoupling warning, all tests passing)?
- **(b)** Did the GitHub push complete with `.env` confirmed absent?

Then either debug remaining fixes or move to final polish (README screenshots, repo description, portfolio writeup).

## How to Respond
- One-line confirmation you understood this handoff
- Ask only the 2 confirmation questions above
- Continue from where I say we are; don't re-explain the architecture back to me

---

## Tech Stack Reference
Python 3.12 · FastAPI · Uvicorn · Groq API (LLaMA 3.3 70B) · NumPy · NetworkX · Matplotlib · WebSockets · HTML/CSS/JS · Pytest · python-dotenv
