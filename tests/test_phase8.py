"""Tests for Phase 8 — Multi-Layer routing with vias.

One dedicated test per behaviour (no bundling).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import app as app_module
from app import app

from phase1_eda_engine import (
    CircuitGraph,
    Component,
    Footprint,
    GraphEdge,
    GridMetadata,
    InitialPlacer,
    NetlistParser,
    Pin,
)
from phase8_multilayer_router import (
    CELL_BLOCKED,
    CELL_FREE,
    CELL_ROUTED,
    LayeredLeeRouter,
    MultiLayerGrid,
    MultiLayerNetRouter,
    compute_layer_crossings,
    run_phase8,
)

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_graph() -> CircuitGraph:
    """Standard 4-component graph from sample_netlist.json (placed)."""
    raw = json.loads(_SAMPLE_JSON.read_text(encoding="utf-8"))
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    return CircuitGraph.from_netlist(netlist)


def _pt_comp(cid: str, x: int, y: int) -> Component:
    """A 1x1 component with one centred pin at (x, y)."""
    pin = Pin(id="P", pin_type="PASSIVE", net="")
    pin.abs_x, pin.abs_y = float(x), float(y)
    return Component(id=cid, comp_type="IC", name=cid, pins=[pin],
                     footprint=Footprint(width=1, height=1), x=x, y=y)


def _crossing_graph() -> CircuitGraph:
    """Two nets whose straight paths cross: H (horizontal) and V (vertical)."""
    meta = GridMetadata(width=24, height=20, unit="mm", name="cross")
    a, b = _pt_comp("A", 2, 9),  _pt_comp("B", 21, 9)   # net H, y=9
    c, d = _pt_comp("C", 11, 2), _pt_comp("D", 11, 17)  # net V, x=11
    edges = [
        GraphEdge(net_id="H", net_type="SIGNAL", source=("A", "P"), target=("B", "P")),
        GraphEdge(net_id="V", net_type="SIGNAL", source=("C", "P"), target=("D", "P")),
    ]
    return CircuitGraph(
        nodes={"A": a, "B": b, "C": c, "D": d},
        edges=edges,
        adjacency={"A": {"B"}, "B": {"A"}, "C": {"D"}, "D": {"C"}},
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# test_crossing_case_zero_same_layer_crossings
# ---------------------------------------------------------------------------

def test_crossing_case_zero_same_layer_crossings(monkeypatch) -> None:
    """Two crossing nets route with 0 same-layer crossings using 2 layers."""
    monkeypatch.setenv("ROUTING_LAYERS", "2")
    graph = _crossing_graph()
    _g, traces, metrics = run_phase8(graph)
    assert metrics["total_failed"] == 0, "both nets must route"
    assert metrics["same_layer_crossings"] == 0
    # Independently recompute to be sure.
    same_layer, _ = compute_layer_crossings(traces)
    assert same_layer == 0


# ---------------------------------------------------------------------------
# test_via_only_at_transition_and_blocks_all_layers
# ---------------------------------------------------------------------------

def test_via_only_at_transition_and_blocks_all_layers() -> None:
    """A via appears only where a path changes layer, and blocks all layers."""
    grid = MultiLayerGrid(9, 3, 2)
    # Force layer-0 start (block layer 1 near source) then a layer-0 mid wall.
    for y in range(3):
        grid.grid[1, y, 1] = CELL_ROUTED   # wall on layer 1 at x=1
        grid.grid[0, y, 4] = CELL_ROUTED   # wall on layer 0 at x=4
    pins = {(0, 1), (8, 1)}
    nodes = LayeredLeeRouter(grid).route((0, 1), (8, 1), "netA", pins)
    assert nodes is not None
    vias = grid.mark_trace(nodes, "netA", "SIGNAL", pins)
    assert vias, "scenario must force at least one via"

    # Every via corresponds to a real layer transition in the path.
    transitions = {
        (x0, y0)
        for (x0, y0, l0), (x1, y1, l1) in zip(nodes, nodes[1:])
        if x0 == x1 and y0 == y1 and l0 != l1
    }
    assert set(vias) == transitions

    # The via cell is blocked for other nets on EVERY layer.
    for vx, vy in vias:
        assert grid.via_blocked_for(vx, vy, "netB")
        for L in range(grid.layers):
            assert grid.grid[L, vy, vx] != CELL_FREE


# ---------------------------------------------------------------------------
# test_run_phase8_full_completion_on_sample
# ---------------------------------------------------------------------------

def test_run_phase8_full_completion_on_sample(sample_graph: CircuitGraph) -> None:
    """run_phase8 achieves 100% routing completion on the sample netlist."""
    _g, _traces, metrics = run_phase8(sample_graph)
    assert metrics["total_failed"] == 0
    assert metrics["completion_pct"] == 100.0


# ---------------------------------------------------------------------------
# test_min_trace_separation_per_layer
# ---------------------------------------------------------------------------

def test_min_trace_separation_per_layer() -> None:
    """A routed trace claims 1-cell clearance on its layer only (not others)."""
    grid = MultiLayerGrid(10, 5, 2)
    pins = {(0, 2), (9, 2)}
    nodes = LayeredLeeRouter(grid).route((0, 2), (9, 2), "n", pins)
    assert nodes is not None
    grid.mark_trace(nodes, "n", "SIGNAL", pins)

    # Pick an interior cell on layer 0 (the straight trace runs along y=2).
    ix, iy, il = nodes[len(nodes) // 2]
    assert grid.grid[il, iy, ix] == CELL_ROUTED
    # A same-layer orthogonal neighbour must be BLOCKED clearance...
    blocked_neighbours = [
        grid.grid[il, iy + dy, ix + dx]
        for dx, dy in ((0, -1), (0, 1))
        if grid.in_bounds(ix + dx, iy + dy)
    ]
    assert CELL_BLOCKED in blocked_neighbours
    # ...but the OTHER layer at the same neighbour stays free (independent DRC).
    other = 1 - il
    assert grid.grid[other, iy + 1, ix] == CELL_FREE


# ---------------------------------------------------------------------------
# test_phase4_reports_via_and_per_layer_crossings
# ---------------------------------------------------------------------------

def test_phase4_reports_via_and_per_layer_crossings(sample_graph: CircuitGraph) -> None:
    """Phase 4 returns via_count + per_layer_crossings AND the original keys."""
    from phase4_analytics import run_phase4

    graph, traces, p3 = run_phase8(sample_graph)
    board = run_phase4(graph, traces, p3)

    # New additive fields.
    assert hasattr(board, "via_count")
    assert isinstance(board.via_count, int)
    assert hasattr(board, "per_layer_crossings")
    assert isinstance(board.per_layer_crossings, dict)

    # Original metric keys still present and populated.
    for attr in ("hpwl_mm", "wire_crossing_count", "routing_completion_pct",
                 "total_trace_length_mm", "total_resistance_ohms"):
        assert hasattr(board, attr)


# ---------------------------------------------------------------------------
# test_phase4_backward_compatible_single_layer
# ---------------------------------------------------------------------------

def test_phase4_backward_compatible_single_layer(sample_graph: CircuitGraph) -> None:
    """Phase 4 still consumes single-layer Phase 3 output (paths => layer 0)."""
    from phase3_router import run_phase3
    from phase4_analytics import run_phase4

    graph, traces, p3 = run_phase3(sample_graph)
    board = run_phase4(graph, traces, p3)
    assert board.via_count == 0
    assert isinstance(board.per_layer_crossings, dict)


# ---------------------------------------------------------------------------
# test_app_accepts_router_single
# ---------------------------------------------------------------------------

def test_app_accepts_router_single() -> None:
    """POST /generate with router=single is accepted and forwards 'single'."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate",
                           json={"prompt": "blink LED", "router": "single"})
    assert resp.status_code == 200
    assert m.call_args.args[2] == "single"


# ---------------------------------------------------------------------------
# test_app_accepts_router_multi
# ---------------------------------------------------------------------------

def test_app_accepts_router_multi() -> None:
    """POST /generate with router=multi is accepted and forwards 'multi'."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate",
                           json={"prompt": "blink LED", "router": "multi"})
    assert resp.status_code == 200
    assert m.call_args.args[2] == "multi"


# ---------------------------------------------------------------------------
# test_app_rejects_garbage_router
# ---------------------------------------------------------------------------

def test_app_rejects_garbage_router() -> None:
    """POST /generate with an unknown router returns HTTP 400."""
    resp = client.post("/generate",
                       json={"prompt": "blink LED", "router": "bogus"})
    assert resp.status_code == 400
    assert "router" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# test_app_default_router_is_single
# ---------------------------------------------------------------------------

def test_app_default_router_is_single() -> None:
    """Omitting router defaults to 'single' (single-layer stays default)."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate", json={"prompt": "blink LED"})
    assert resp.status_code == 200
    assert m.call_args.args[2] == "single"
