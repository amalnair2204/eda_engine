"""
Phase 6 — FastAPI Integration Server

Routes
------
GET  /                          Serve frontend/index.html
GET  /health                    Health check
POST /generate                  Synchronous full pipeline run
WS   /ws/generate               Streaming pipeline with per-phase messages
GET  /outputs/{filename}        Serve output PNGs
GET  /netlists/generated/{...}  Serve generated netlists
"""

from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="EDA Engine", version="1.0.0", description="AI-Accelerated PCB P&R")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

# Single worker prevents concurrent matplotlib / NumPy calls
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for POST /generate."""
    prompt: str


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _graph_to_layout(graph, traces: list) -> dict:
    """Serialise CircuitGraph + RoutedTrace list to a JSON-safe layout dict."""
    components = []
    for cid, comp in graph.nodes.items():
        pins = [
            {
                "id":       pin.id,
                "abs_x":   float(pin.abs_x),
                "abs_y":   float(pin.abs_y),
                "net":      pin.net,
                "pin_type": pin.pin_type,
            }
            for pin in comp.pins
        ]
        components.append({
            "id":     comp.id,
            "type":   comp.comp_type,
            "name":   comp.name,
            "x":      comp.x,
            "y":      comp.y,
            "width":  comp.footprint.width,
            "height": comp.footprint.height,
            "pins":   pins,
        })

    trace_list = [
        {
            "net_id":      t.net_id,
            "net_type":    t.net_type,
            "source_comp": t.source_comp,
            "source_pin":  t.source_pin,
            "target_comp": t.target_comp,
            "target_pin":  t.target_pin,
            "path":        [[x, y] for x, y in t.path],
            "length":      t.length,
        }
        for t in traces
    ]

    return {
        "components": components,
        "traces":     trace_list,
        "grid": {
            "width":  graph.metadata.width,
            "height": graph.metadata.height,
        },
    }


def _board_to_metrics(board) -> dict:
    """Serialise BoardMetrics to a JSON-safe dict for the frontend."""
    return {
        "total_traces_routed":   board.total_traces_routed,
        "total_traces_failed":   board.total_traces_failed,
        "total_trace_length_mm": board.total_trace_length_mm,
        "longest_trace_mm":      board.longest_trace_mm,
        "shortest_trace_mm":     board.shortest_trace_mm,
        "wire_crossing_count":   board.wire_crossing_count,
        "routing_completion_pct": board.routing_completion_pct,
        "hpwl_mm":               board.hpwl_mm,
        "total_resistance_ohms": board.total_resistance_ohms,
        "total_capacitance_pf":  board.total_capacitance_pf,
        "max_signal_delay_ps":   board.max_signal_delay_ps,
        "violations":            board.violations,
    }


# ---------------------------------------------------------------------------
# Synchronous pipeline runner (runs in thread executor)
# ---------------------------------------------------------------------------

def _run_full_pipeline(prompt: str) -> dict:
    """Execute Phase 0-4 synchronously.  Called via run_in_executor."""
    from phase0_groq_translator import run_phase0
    from phase1_eda_engine import CircuitGraph, InitialPlacer, NetlistParser
    from phase2_genetic_placer import run_phase2
    from phase3_router import run_phase3
    from phase4_analytics import run_phase4

    netlist_dict = run_phase0(prompt)

    parser  = NetlistParser()
    netlist = parser.parse(netlist_dict)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    graph = run_phase2(graph)

    graph, traces, p3 = run_phase3(graph)

    board = run_phase4(graph, traces, p3)

    return {
        "status":  "complete",
        "netlist": netlist_dict,
        "layout":  _graph_to_layout(graph, traces),
        "metrics": _board_to_metrics(board),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_model=None)
async def root():
    """Serve the frontend SPA."""
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h2>frontend/index.html not found — run the server from the project root.</h2>",
            status_code=404,
        )
    return FileResponse(index)


@app.get("/health")
async def health() -> dict:
    """Liveness / readiness check."""
    return {"status": "ok", "pipeline": "ready"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """Run the full Phase 0-4 pipeline and return one JSON response."""
    if not req.prompt.strip():
        return {"status": "error", "message": "Prompt cannot be empty"}

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run_full_pipeline, req.prompt)
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "message": str(exc)}


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket) -> None:
    """WebSocket endpoint — streams one JSON message per completed phase."""
    await websocket.accept()

    async def send(msg: dict) -> None:
        await websocket.send_text(json.dumps(msg))

    try:
        data   = await websocket.receive_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            await send({"phase": -1, "status": "error", "message": "Prompt is empty"})
            return

        loop = asyncio.get_event_loop()

        # ── Phase 0 — LLM translation ─────────────────────────────────
        await send({"phase": 0, "status": "running",
                    "message": "Calling Groq API (LLaMA 3.3 70B)..."})
        try:
            from phase0_groq_translator import run_phase0
            netlist_dict = await loop.run_in_executor(_executor, run_phase0, prompt)
            inner  = netlist_dict.get("netlist", netlist_dict)
            n_comp = len(inner.get("components", []))
            n_nets = len(inner.get("nets", []))
            await send({
                "phase": 0, "status": "complete",
                "message": f"Netlist generated — {n_comp} components, {n_nets} nets",
                "netlist": netlist_dict,
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Phase 0 (LLM) failed: {exc}"})
            return

        # ── Phase 1 — Parse + Graph ───────────────────────────────────
        await send({"phase": 1, "status": "running",
                    "message": "Parsing netlist and building circuit graph..."})
        try:
            from phase1_eda_engine import (
                CircuitGraph, InitialPlacer, NetlistParser,
                half_perimeter_wire_length,
            )
            parser  = NetlistParser()
            netlist = parser.parse(netlist_dict)
            InitialPlacer(netlist.metadata).place(netlist)
            graph = CircuitGraph.from_netlist(netlist)
            hpwl  = half_perimeter_wire_length(graph)
            await send({
                "phase": 1, "status": "complete",
                "message": f"Graph built — {len(graph.nodes)} nodes, HPWL: {hpwl:.1f} mm",
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Phase 1 (Graph) failed: {exc}"})
            return

        # ── Phase 2 — Genetic Algorithm ───────────────────────────────
        await send({"phase": 2, "status": "running",
                    "message": "Running Genetic Algorithm placement optimizer..."})
        try:
            from phase2_genetic_placer import run_phase2
            hpwl_before = half_perimeter_wire_length(graph)
            graph = await loop.run_in_executor(_executor, run_phase2, graph)
            hpwl_after  = half_perimeter_wire_length(graph)
            pct = (hpwl_before - hpwl_after) / max(hpwl_before, 1e-9) * 100
            await send({
                "phase": 2, "status": "complete",
                "message": (
                    f"GA complete — HPWL {hpwl_before:.1f} → {hpwl_after:.1f} mm"
                    f" ({pct:.1f}% improvement)"
                ),
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Phase 2 (GA) failed: {exc}"})
            return

        # ── Phase 3 — Maze Router ─────────────────────────────────────
        await send({"phase": 3, "status": "running",
                    "message": "Running Lee's Algorithm maze router..."})
        try:
            from phase3_router import run_phase3
            graph, traces, p3 = await loop.run_in_executor(
                _executor, run_phase3, graph
            )
            routed   = p3.get("total_routed", 0)
            failed   = p3.get("total_failed", 0)
            crossings = p3.get("crossing_count", 0)
            await send({
                "phase": 3, "status": "complete",
                "message": (
                    f"Routing complete — {routed} segments, "
                    f"{failed} failed, {crossings} crossings"
                ),
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Phase 3 (Router) failed: {exc}"})
            return

        # ── Phase 4 — Analytics ───────────────────────────────────────
        await send({"phase": 4, "status": "running",
                    "message": "Computing EEE electrical analytics..."})
        try:
            from phase4_analytics import run_phase4

            def _p4() -> object:
                return run_phase4(graph, traces, p3)

            board = await loop.run_in_executor(_executor, _p4)
            n_viol = len(board.violations)
            await send({
                "phase": 4, "status": "complete",
                "message": f"Analytics ready — {n_viol} violation(s) found",
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Phase 4 (Analytics) failed: {exc}"})
            return

        # ── Final result ──────────────────────────────────────────────
        await send({
            "phase": -1,
            "status": "result",
            "data": {
                "status":  "complete",
                "netlist": netlist_dict,
                "layout":  _graph_to_layout(graph, traces),
                "metrics": _board_to_metrics(board),
            },
        })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            await send({"phase": -1, "status": "error", "message": str(exc)})
        except Exception:
            pass


@app.get("/outputs/{filename}")
async def serve_output(filename: str) -> FileResponse:
    """Serve files from outputs/ (phase PNG images, etc.)."""
    path = _OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {filename}")
    return FileResponse(path)


@app.get("/netlists/generated/{filename}")
async def serve_netlist(filename: str) -> FileResponse:
    """Serve generated netlist JSON files."""
    path = _PROJECT_ROOT / "netlists" / "generated" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {filename}")
    return FileResponse(path)


# Mount static assets AFTER all named routes
_FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
