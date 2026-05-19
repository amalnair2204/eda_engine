"""Tests for Phase 3 — Lee's Algorithm Maze Router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
from phase3_router import (
    CELL_BLOCKED,
    CELL_COMPONENT,
    CELL_FREE,
    CELL_ROUTED,
    LeeRouter,
    NetRouter,
    RoutedTrace,
    RoutingGrid,
    run_phase3,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"


@pytest.fixture
def sample_graph() -> CircuitGraph:
    """Phase-1+2 placed graph from sample_netlist.json."""
    from phase2_genetic_placer import run_phase2

    with _SAMPLE_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)
    return run_phase2(graph, pop_size=10, n_generations=20)


def _empty_grid(w: int = 20, h: int = 20) -> RoutingGrid:
    """Return an all-FREE RoutingGrid of the given dimensions."""
    rg = RoutingGrid(w, h)
    rg.grid[:] = CELL_FREE
    return rg


def _graph_with_two_comps() -> CircuitGraph:
    """Minimal graph: two 2x2 MCU components connected by one POWER edge."""
    meta = GridMetadata(width=20, height=20, unit="mm", name="test")
    c1 = Component(
        id="C1", comp_type="MCU", name="MCU1",
        pins=[Pin(id="VCC", pin_type="POWER", net="VCC",
                  abs_x=3.0, abs_y=2.0)],
        footprint=Footprint(width=2, height=2),
        x=2, y=2, properties={},
    )
    c2 = Component(
        id="C2", comp_type="MCU", name="MCU2",
        pins=[Pin(id="VCC", pin_type="POWER", net="VCC",
                  abs_x=12.0, abs_y=2.0)],
        footprint=Footprint(width=2, height=2),
        x=12, y=2, properties={},
    )
    edge = GraphEdge(
        net_id="VCC", net_type="POWER",
        source=("C1", "VCC"), target=("C2", "VCC"),
    )
    return CircuitGraph(
        nodes={"C1": c1, "C2": c2},
        edges=[edge],
        adjacency={"C1": {"C2"}, "C2": {"C1"}},
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# test_routing_grid_initializes_correctly
# ---------------------------------------------------------------------------

def test_routing_grid_initializes_correctly() -> None:
    """Component footprint cells are marked CELL_COMPONENT after init."""
    graph = _graph_with_two_comps()
    rg = RoutingGrid(20, 20)
    rg.initialize_from_graph(graph)

    # C1 at (2,2) footprint 2x2 → cells (2,2),(3,2),(2,3),(3,3)
    for dx in range(2):
        for dy in range(2):
            assert rg.grid[2 + dy, 2 + dx] == CELL_COMPONENT, \
                f"Expected COMPONENT at ({2+dx},{2+dy})"

    # A cell outside any component should be FREE
    assert rg.grid[0, 0] == CELL_FREE


# ---------------------------------------------------------------------------
# test_routing_grid_in_bounds
# ---------------------------------------------------------------------------

def test_routing_grid_in_bounds() -> None:
    """Boundary cells are correctly identified."""
    rg = RoutingGrid(10, 10)
    assert rg.in_bounds(0, 0)
    assert rg.in_bounds(9, 9)
    assert not rg.in_bounds(-1, 0)
    assert not rg.in_bounds(0, -1)
    assert not rg.in_bounds(10, 0)
    assert not rg.in_bounds(0, 10)


# ---------------------------------------------------------------------------
# test_routing_grid_mark_trace
# ---------------------------------------------------------------------------

def test_routing_grid_mark_trace() -> None:
    """Intermediate path cells become CELL_ROUTED after mark_trace."""
    rg   = _empty_grid(10, 10)
    path = [(1, 5), (2, 5), (3, 5), (4, 5)]
    rg.mark_trace(path)

    # Intermediate cells (2,5) and (3,5) must be ROUTED
    assert rg.grid[5, 2] == CELL_ROUTED
    assert rg.grid[5, 3] == CELL_ROUTED
    # Endpoints should NOT be changed to ROUTED
    assert rg.grid[5, 1] != CELL_ROUTED
    assert rg.grid[5, 4] != CELL_ROUTED


# ---------------------------------------------------------------------------
# test_lee_router_finds_path
# ---------------------------------------------------------------------------

def test_lee_router_finds_path() -> None:
    """Router finds a path across an open grid."""
    rg   = _empty_grid(15, 15)
    router = LeeRouter(rg)
    path = router.route(1, 7, 13, 7)

    assert path is not None
    assert len(path) >= 2
    assert path[0]  == (1, 7)
    assert path[-1] == (13, 7)


# ---------------------------------------------------------------------------
# test_lee_router_navigates_obstacle
# ---------------------------------------------------------------------------

def test_lee_router_navigates_obstacle() -> None:
    """Router finds a path around a COMPONENT obstacle."""
    rg = _empty_grid(12, 12)
    # Block column x=6 from y=2 to y=9 — leaves a gap at y=0..1 to route around
    for y in range(2, 10):
        rg.grid[y, 6] = CELL_COMPONENT

    router = LeeRouter(rg)
    path = router.route(2, 6, 10, 6)

    assert path is not None
    # Intermediate cells must not pass through the blocked column mid-section
    blocked_cells = {(6, y) for y in range(2, 10)}
    assert all(cell not in blocked_cells for cell in path[1:-1])


# ---------------------------------------------------------------------------
# test_lee_router_returns_none_when_blocked
# ---------------------------------------------------------------------------

def test_lee_router_returns_none_when_blocked() -> None:
    """Router returns None when target is completely surrounded by obstacles."""
    rg = _empty_grid(10, 10)
    # Surround (5,5) on all four sides with COMPONENT
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rg.grid[5 + dy, 5 + dx] = CELL_COMPONENT
    # Also block a 3x3 ring to ensure the BFS cannot get adjacent
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dx) == 2 or abs(dy) == 2:
                rg.grid[5 + dy, 5 + dx] = CELL_COMPONENT

    router = LeeRouter(rg)
    path   = router.route(0, 0, 5, 5)
    assert path is None


# ---------------------------------------------------------------------------
# test_path_connects_source_to_target
# ---------------------------------------------------------------------------

def test_path_connects_source_to_target() -> None:
    """Path returned by the router starts at source and ends at target."""
    rg     = _empty_grid(12, 12)
    router = LeeRouter(rg)
    path   = router.route(0, 0, 11, 11)

    assert path is not None
    assert path[0]  == (0, 0)
    assert path[-1] == (11, 11)


# ---------------------------------------------------------------------------
# test_net_priority_order
# ---------------------------------------------------------------------------

def test_net_priority_order() -> None:
    """POWER nets come before SIGNAL nets in the routing order."""
    graph = _graph_with_two_comps()
    # Add a SIGNAL edge between the same components
    signal_edge = GraphEdge(
        net_id="SIG", net_type="SIGNAL",
        source=("C1", "VCC"), target=("C2", "VCC"),
    )
    graph.edges.append(signal_edge)

    rg     = RoutingGrid(20, 20)
    nr     = NetRouter(graph, rg)
    ordered = nr._prioritize_nets()
    types  = [n.net_type for n in ordered]

    power_idx  = next(i for i, t in enumerate(types) if t == "POWER")
    signal_idx = next(i for i, t in enumerate(types) if t == "SIGNAL")
    assert power_idx < signal_idx


# ---------------------------------------------------------------------------
# test_route_all_returns_traces
# ---------------------------------------------------------------------------

def test_route_all_returns_traces(sample_graph: CircuitGraph) -> None:
    """run_phase3 returns at least one RoutedTrace object."""
    _graph, traces, _metrics = run_phase3(sample_graph)
    assert len(traces) >= 1
    assert all(isinstance(t, RoutedTrace) for t in traces)


# ---------------------------------------------------------------------------
# test_metrics_dict_has_required_keys
# ---------------------------------------------------------------------------

def test_metrics_dict_has_required_keys(sample_graph: CircuitGraph) -> None:
    """Metrics dict contains all keys Phase 4 expects."""
    required = {
        "total_routed", "total_failed", "total_length",
        "crossing_count", "longest_trace", "shortest_trace", "failed_routes",
    }
    _graph, _traces, metrics = run_phase3(sample_graph)
    assert required.issubset(metrics.keys())


# ---------------------------------------------------------------------------
# test_no_trace_exits_grid
# ---------------------------------------------------------------------------

def test_no_trace_exits_grid(sample_graph: CircuitGraph) -> None:
    """Every cell in every routed trace path lies within grid bounds."""
    gw = sample_graph.metadata.width
    gh = sample_graph.metadata.height
    _graph, traces, _metrics = run_phase3(sample_graph)

    for trace in traces:
        for x, y in trace.path:
            assert 0 <= x < gw, f"x={x} out of grid width {gw}"
            assert 0 <= y < gh, f"y={y} out of grid height {gh}"


# ---------------------------------------------------------------------------
# test_crossing_count_is_non_negative
# ---------------------------------------------------------------------------

def test_crossing_count_is_non_negative(sample_graph: CircuitGraph) -> None:
    """Crossing count in metrics is always >= 0."""
    _graph, _traces, metrics = run_phase3(sample_graph)
    assert metrics["crossing_count"] >= 0
