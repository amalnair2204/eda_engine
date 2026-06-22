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
import os
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

# Most-recent routed board, cached for the manufacturing export endpoint.
# Populated at the end of the routing step in both /generate and /ws/generate.
_LAST_ROUTED: dict = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CopilotRequest(BaseModel):
    """Request body for POST /copilot."""
    query: str
    history: list = []


class GenerateRequest(BaseModel):
    """Request body for POST /generate."""
    prompt: str
    placer: str = "ga"   # "ga" (default, Genetic Algorithm) | "rl" (Phase 7 RL)
    router: str = "single"  # "single" (default, Phase 3) | "multi" (Phase 8)


class ExploreRequest(BaseModel):
    """Request body for POST /explore (all fields optional)."""
    config: dict | None = None   # overrides phase11 _default_config() when set


# Valid placement strategies, mapping the request value to a human label.
_VALID_PLACERS = {"ga": "Genetic Algorithm", "rl": "RL agent"}

# Valid routing strategies, mapping the request value to a human label.
_VALID_ROUTERS = {"single": "single-layer router", "multi": "multi-layer router"}


def _resolve_router(name: str):
    """Return the Router strategy function for a request value.

    Args:
        name: "single" or "multi".

    Returns:
        Callable(CircuitGraph) -> (graph, traces, metrics) — run_phase3 or run_phase8.

    Raises:
        ValueError: If name is not a recognised router.
    """
    if name == "single":
        from phase3_router import run_phase3
        return run_phase3
    if name == "multi":
        from phase8_multilayer_router import run_phase8
        return run_phase8
    raise ValueError(f"Unknown router '{name}'. Use 'single' or 'multi'.")


def _resolve_placer(name: str):
    """Return the Placer strategy function for a request value.

    Args:
        name: "ga" or "rl".

    Returns:
        Callable(CircuitGraph) -> CircuitGraph (run_phase2 or run_phase7).

    Raises:
        ValueError: If name is not a recognised placer.
    """
    if name == "ga":
        from phase2_genetic_placer import run_phase2
        return run_phase2
    if name == "rl":
        from phase7_rl_placer import run_phase7
        return run_phase7
    raise ValueError(f"Unknown placer '{name}'. Use 'ga' or 'rl'.")


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
        "via_count":             getattr(board, "via_count", 0),
        "per_layer_crossings":   getattr(board, "per_layer_crossings", {}),
    }


# ---------------------------------------------------------------------------
# Synchronous pipeline runner (runs in thread executor)
# ---------------------------------------------------------------------------

def _run_full_pipeline(prompt: str, placer: str = "ga", router: str = "single") -> dict:
    """Execute Phase 0-4 synchronously.  Called via run_in_executor.

    Args:
        prompt: Plain-English circuit description.
        placer: Placement strategy — "ga" (default) or "rl".
        router: Routing strategy — "single" (default) or "multi".
    """
    from phase0_groq_translator import run_phase0
    from phase1_eda_engine import CircuitGraph, InitialPlacer, NetlistParser
    from phase4_analytics import run_phase4

    place = _resolve_placer(placer)
    route = _resolve_router(router)

    netlist_dict = run_phase0(prompt)

    parser  = NetlistParser()
    netlist = parser.parse(netlist_dict)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    graph = place(graph)

    graph, traces, p3 = route(graph)
    # Cache for /export and /explore (netlist_dict enables a fresh re-place).
    _LAST_ROUTED.update(graph=graph, traces=traces, netlist=netlist_dict)

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

    # Unknown placer → 400 with a clear message (ga remains the default).
    if req.placer not in _VALID_PLACERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown placer '{req.placer}'. Use 'ga' or 'rl'.",
        )

    # Unknown router → 400 with a clear message (single remains the default).
    if req.router not in _VALID_ROUTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown router '{req.router}'. Use 'single' or 'multi'.",
        )

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _run_full_pipeline, req.prompt, req.placer, req.router
        )
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
        placer = data.get("placer", "ga").strip().lower()
        router = data.get("router", "single").strip().lower()

        if not prompt:
            await send({"phase": -1, "status": "error", "message": "Prompt is empty"})
            return

        if placer not in _VALID_PLACERS:
            await send({"phase": -1, "status": "error",
                        "message": f"Unknown placer '{placer}'. Use 'ga' or 'rl'."})
            return

        if router not in _VALID_ROUTERS:
            await send({"phase": -1, "status": "error",
                        "message": f"Unknown router '{router}'. Use 'single' or 'multi'."})
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

        # ── Phase 2 / 7 — Placement (GA default, RL optional) ─────────
        _placer_label = _VALID_PLACERS[placer]
        await send({"phase": 2, "status": "running",
                    "message": f"Running {_placer_label} placement optimizer..."})
        try:
            place = _resolve_placer(placer)
            hpwl_before = half_perimeter_wire_length(graph)
            graph = await loop.run_in_executor(_executor, place, graph)
            hpwl_after  = half_perimeter_wire_length(graph)
            pct = (hpwl_before - hpwl_after) / max(hpwl_before, 1e-9) * 100
            await send({
                "phase": 2, "status": "complete",
                "message": (
                    f"{_placer_label} complete — HPWL {hpwl_before:.1f} → "
                    f"{hpwl_after:.1f} mm ({pct:.1f}% improvement)"
                ),
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Placement ({_placer_label}) failed: {exc}"})
            return

        # ── Phase 3 / 8 — Router (single-layer default, multi optional) ──
        _router_label = _VALID_ROUTERS[router]
        await send({"phase": 3, "status": "running",
                    "message": f"Running {_router_label}..."})
        try:
            route = _resolve_router(router)
            graph, traces, p3 = await loop.run_in_executor(_executor, route, graph)
            # Cache for /export and /explore.
            _LAST_ROUTED.update(graph=graph, traces=traces, netlist=netlist_dict)
            routed   = p3.get("total_routed", 0)
            failed   = p3.get("total_failed", 0)
            crossings = p3.get("crossing_count", 0)
            vias      = p3.get("via_count", 0)
            via_str   = f", {vias} vias" if router == "multi" else ""
            await send({
                "phase": 3, "status": "complete",
                "message": (
                    f"Routing complete — {routed} segments, "
                    f"{failed} failed, {crossings} same-layer crossings{via_str}"
                ),
            })
        except Exception as exc:
            await send({"phase": -1, "status": "error",
                        "message": f"Router ({_router_label}) failed: {exc}"})
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


@app.post("/copilot")
async def copilot(req: CopilotRequest) -> dict:
    """RAG design copilot — grounded, cited answer to a design question.

    Uses the most-recently routed board (if any) as read-only design context.
    Returns {answer, citations, sources}.  Empty query → 400.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    from phase10_rag_copilot import run_phase10

    graph = _LAST_ROUTED.get("graph")

    def _ask() -> dict:
        return run_phase10(req.query, circuit_graph=graph, history=req.history)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _ask)
    return {
        "status":    "complete",
        "answer":    result["answer"],
        "citations": result["citations"],
        "sources":   [c["source"] for c in result["retrieved_chunks"]],
    }


@app.websocket("/copilot/stream")
async def copilot_stream(websocket: WebSocket) -> None:
    """WebSocket copilot — streams answer tokens, then a final cited result."""
    await websocket.accept()

    async def send(msg: dict) -> None:
        await websocket.send_text(json.dumps(msg))

    try:
        data    = await websocket.receive_json()
        query   = (data.get("query") or "").strip()
        history = data.get("history", [])
        if not query:
            await send({"type": "error", "message": "Query must not be empty."})
            return

        from phase10_rag_copilot import stream_phase10

        graph = _LAST_ROUTED.get("graph")
        loop  = asyncio.get_event_loop()
        gen   = stream_phase10(query, circuit_graph=graph, history=history)

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            item = await loop.run_in_executor(_executor, _next)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                await send({"type": "token", "token": payload})
            elif kind == "done":
                await send({"type": "done",
                            "answer": payload["answer"],
                            "citations": payload["citations"],
                            "sources": [c["source"] for c in payload["retrieved_chunks"]]})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            await send({"type": "error", "message": str(exc)})
        except Exception:
            pass


@app.post("/export")
async def export() -> dict:
    """Run Phase 9 manufacturing export on the most-recently routed board.

    Returns a JSON manifest of the produced files plus a download URL for the
    fab-ready Gerber/drill zip.  Generate a board first (POST /generate or the
    WebSocket) — the routed result is cached server-side for export.
    """
    if not _LAST_ROUTED.get("graph"):
        raise HTTPException(
            status_code=400,
            detail="No routed board to export — run /generate first.",
        )

    from phase9_export import run_phase9

    def _do_export() -> dict:
        return run_phase9(_LAST_ROUTED["graph"], _LAST_ROUTED["traces"])

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _do_export)

    board = result["board"]
    manifest = {
        "status":         "complete",
        "board":          board,
        "layers":         result["layers"],
        "drill_hits":     result["drill_hits"],
        "bom_total_qty":  result["bom_total_qty"],
        "unrouted_nets":  result["unrouted_nets"],
        "completion_pct": result["completion_pct"],
        "files": {
            "copper_gerbers": [p.name for p in result["copper_gerbers"].values()],
            "outline_gerber": result["outline_gerber"].name,
            "drill":          result["drill"].name,
            "bom":            result["bom"].name,
            "kicad_netlist":  result["kicad_netlist"].name,
            "zip":            result["zip"].name,
        },
        "zip_url": f"/export/download/{result['zip'].name}",
    }
    return manifest


def _explore_to_json(result: dict) -> dict:
    """Serialise a run_phase11 result to a JSON-safe dict for the frontend."""
    rec = result.get("recommendation")
    return {
        "status":         "complete",
        "objectives":     result["objectives"],
        "candidates":     result["candidates"],   # already JSON-friendly dicts
        "pareto_ids":     [c["id"] for c in result["pareto"]],
        "recommendation": rec,                     # dict with "rationale" or None
        "pareto_png_url": f"/outputs/{Path(result['pareto_png']).name}",
        "results_md":     Path(result["results_md"]).name,
    }


@app.post("/explore")
async def explore(req: ExploreRequest) -> dict:
    """Phase 11 design-space exploration over the most-recently generated netlist.

    Sweeps multiple placement/routing strategies, scores each on the stated
    objectives, and returns the candidate table, the Pareto-optimal set, and a
    recommended trade-off.  Generate a board first (POST /generate or the
    WebSocket) — the netlist is cached server-side for re-exploration.
    """
    if not _LAST_ROUTED.get("netlist"):
        raise HTTPException(
            status_code=400,
            detail="No netlist to explore — run /generate first.",
        )

    netlist_dict = _LAST_ROUTED["netlist"]
    config = req.config or None

    def _do_explore() -> dict:
        from phase1_eda_engine import CircuitGraph, InitialPlacer, NetlistParser
        from phase11_explorer import run_phase11
        # Fresh graph from the cached netlist — clean seed positions, original
        # grid (Phase 11 deep-copies per candidate, so this stays read-only).
        netlist = NetlistParser().parse(netlist_dict)
        InitialPlacer(netlist.metadata).place(netlist)
        graph = CircuitGraph.from_netlist(netlist)
        return run_phase11(graph, config)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _do_explore)
    return _explore_to_json(result)


@app.get("/export/download/{filename}")
async def export_download(filename: str) -> FileResponse:
    """Serve a generated manufacturing file (e.g. the fab-ready zip)."""
    path = (_PROJECT_ROOT / "outputs" / "manufacturing" / filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {filename}")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=filename)


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


# ---------------------------------------------------------------------------
# Hosting entry-point (env-based host/port for containers / PaaS)
# ---------------------------------------------------------------------------

def _host_port() -> tuple[str, int]:
    """Resolve the bind (host, port) from env for container / PaaS hosting.

    Reads HOST (default "0.0.0.0") and PORT (default 8000).  A non-integer PORT
    falls back to 8000 rather than crashing the server on a bad env value.

    Returns:
        (host, port) tuple suitable for uvicorn.run().
    """
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        port = 8000
    return host, port


if __name__ == "__main__":
    import uvicorn

    _host, _port = _host_port()
    uvicorn.run(app, host=_host, port=_port)
