"""Tests for Phase 4 — Analytics Engine."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

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
from phase3_router import RoutedTrace
from phase4_analytics import (
    AnalyticsEngine,
    BoardMetrics,
    CELL_SIZE_MM,
    ReportGenerator,
    TraceMetrics,
    run_phase4,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"


def _make_trace(
    net_id: str = "SIG",
    net_type: str = "SIGNAL",
    length_cells: int = 6,
) -> RoutedTrace:
    """Return a minimal RoutedTrace with the given length."""
    path = [(i, 0) for i in range(length_cells)]
    return RoutedTrace(
        net_id=net_id, net_type=net_type,
        source_comp="C1", source_pin="P1",
        target_comp="C2", target_pin="P2",
        path=path, length=length_cells,
    )


def _make_graph() -> CircuitGraph:
    """Minimal graph with two MCU components and one SIGNAL edge."""
    meta = GridMetadata(width=20, height=20, unit="mm", name="TestDesign")
    c1 = Component(
        id="U1", comp_type="MCU", name="MCU1",
        pins=[Pin(id="SIG", pin_type="OUTPUT", net="SIG",
                  abs_x=2.0, abs_y=2.0)],
        footprint=Footprint(width=2, height=2),
        x=2, y=2, properties={},
    )
    c2 = Component(
        id="R1", comp_type="RESISTOR", name="RES1",
        pins=[Pin(id="P1", pin_type="PASSIVE", net="SIG",
                  abs_x=10.0, abs_y=2.0)],
        footprint=Footprint(width=1, height=1),
        x=10, y=2, properties={},
    )
    edge = GraphEdge(
        net_id="SIG", net_type="SIGNAL",
        source=("U1", "SIG"), target=("R1", "P1"),
    )
    return CircuitGraph(
        nodes={"U1": c1, "R1": c2},
        edges=[edge],
        adjacency={"U1": {"R1"}, "R1": {"U1"}},
        metadata=meta,
    )


def _make_engine(
    traces: list[RoutedTrace] | None = None,
    crossings: int = 0,
    failed: int = 0,
    graph: CircuitGraph | None = None,
) -> AnalyticsEngine:
    """Return an AnalyticsEngine with controllable metrics."""
    if traces is None:
        traces = [_make_trace()]
    if graph is None:
        graph = _make_graph()
    p3 = {
        "total_routed":   len(traces),
        "total_failed":   failed,
        "total_length":   sum(t.length for t in traces),
        "crossing_count": crossings,
        "longest_trace":  max((t.length for t in traces), default=0),
        "shortest_trace": min((t.length for t in traces), default=0),
        "failed_routes":  [],
    }
    return AnalyticsEngine(graph, traces, p3)


@pytest.fixture
def sample_graph_routed():
    """Full Phase 1+2+3 pipeline on sample netlist (fast reduced params)."""
    from phase2_genetic_placer import run_phase2
    from phase3_router import run_phase3

    with _SAMPLE_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)
    graph = run_phase2(graph, pop_size=10, n_generations=20)
    graph, traces, p3 = run_phase3(graph)
    return graph, traces, p3


# ---------------------------------------------------------------------------
# test_trace_length_mm_calculation
# ---------------------------------------------------------------------------

def test_trace_length_mm_calculation() -> None:
    """6 cells at CELL_SIZE_MM=1.0 should equal exactly 6.0 mm."""
    engine = _make_engine()
    trace  = _make_trace(length_cells=6)
    assert engine._calc_trace_length_mm(trace) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# test_resistance_is_positive
# ---------------------------------------------------------------------------

def test_resistance_is_positive() -> None:
    """Resistance must be positive for any non-zero trace length."""
    engine = _make_engine()
    assert engine._calc_resistance(5.0) > 0.0
    assert engine._calc_resistance(1.0) > 0.0


# ---------------------------------------------------------------------------
# test_capacitance_is_positive
# ---------------------------------------------------------------------------

def test_capacitance_is_positive() -> None:
    """Parasitic capacitance must be positive for any non-zero trace length."""
    engine = _make_engine()
    assert engine._calc_parasitic_capacitance(5.0) > 0.0


# ---------------------------------------------------------------------------
# test_signal_delay_is_positive
# ---------------------------------------------------------------------------

def test_signal_delay_is_positive() -> None:
    """Signal delay must be positive for any non-zero trace length."""
    engine = _make_engine()
    assert engine._calc_signal_delay(5.0) > 0.0


# ---------------------------------------------------------------------------
# test_longer_trace_has_higher_resistance
# ---------------------------------------------------------------------------

def test_longer_trace_has_higher_resistance() -> None:
    """10 mm trace must have strictly higher resistance than 5 mm trace."""
    engine = _make_engine()
    assert engine._calc_resistance(10.0) > engine._calc_resistance(5.0)


# ---------------------------------------------------------------------------
# test_longer_trace_has_higher_capacitance
# ---------------------------------------------------------------------------

def test_longer_trace_has_higher_capacitance() -> None:
    """10 mm trace must have strictly higher capacitance than 5 mm trace."""
    engine = _make_engine()
    assert engine._calc_parasitic_capacitance(10.0) > engine._calc_parasitic_capacitance(5.0)


# ---------------------------------------------------------------------------
# test_routing_completion_pct_all_routed
# ---------------------------------------------------------------------------

def test_routing_completion_pct_all_routed() -> None:
    """4 routed / 0 failed -> 100.0% routing completion."""
    traces  = [_make_trace() for _ in range(4)]
    engine  = _make_engine(traces=traces, failed=0)
    metrics = engine.compute()
    assert metrics.routing_completion_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# test_routing_completion_pct_partial
# ---------------------------------------------------------------------------

def test_routing_completion_pct_partial() -> None:
    """3 routed / 1 failed -> 75.0% routing completion."""
    traces  = [_make_trace() for _ in range(3)]
    engine  = _make_engine(traces=traces, failed=1)
    metrics = engine.compute()
    assert metrics.routing_completion_pct == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# test_violation_detected_for_crossings
# ---------------------------------------------------------------------------

def test_violation_detected_for_crossings() -> None:
    """crossing_count=2 must produce a DRC FAIL violation string."""
    engine     = _make_engine(crossings=2)
    violations = engine._check_eee_violations()
    assert any("crossing" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# test_no_violation_when_clean
# ---------------------------------------------------------------------------

def test_no_violation_when_clean() -> None:
    """Zero crossings, no failures, short signal traces -> no violations."""
    trace  = _make_trace(length_cells=5)       # 5 mm — well below 20 mm limit
    engine = _make_engine(traces=[trace], crossings=0, failed=0)
    violations = engine._check_eee_violations()
    # Filter out decoupling-cap warnings (our test graph has no CAPACITOR)
    non_cap = [v for v in violations if "decoupling" not in v.lower()]
    assert len(non_cap) == 0


# ---------------------------------------------------------------------------
# test_board_metrics_has_all_fields
# ---------------------------------------------------------------------------

def test_board_metrics_has_all_fields() -> None:
    """BoardMetrics object must have every required field."""
    engine  = _make_engine()
    metrics = engine.compute()
    required = {
        "design_name", "hpwl_mm", "component_count", "net_count",
        "total_traces_routed", "total_traces_failed", "total_trace_length_mm",
        "longest_trace_mm", "shortest_trace_mm", "wire_crossing_count",
        "routing_completion_pct", "total_resistance_ohms", "total_capacitance_pf",
        "max_signal_delay_ps", "trace_metrics", "violations",
    }
    assert required.issubset(dataclasses.asdict(metrics).keys())


# ---------------------------------------------------------------------------
# test_json_report_is_valid
# ---------------------------------------------------------------------------

def test_json_report_is_valid(tmp_path: Path) -> None:
    """Saved JSON must be parseable and contain all required top-level keys."""
    engine   = _make_engine()
    metrics  = engine.compute()
    reporter = ReportGenerator(metrics)
    out      = tmp_path / "report.json"
    reporter.save_json_report(out)

    with out.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for key in ("design_name", "hpwl_mm", "total_traces_routed",
                "total_capacitance_pf", "trace_metrics", "violations"):
        assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# test_run_phase4_returns_board_metrics
# ---------------------------------------------------------------------------

def test_run_phase4_returns_board_metrics(sample_graph_routed) -> None:
    """run_phase4() must return a BoardMetrics instance."""
    graph, traces, p3 = sample_graph_routed
    result = run_phase4(graph, traces, p3)
    assert isinstance(result, BoardMetrics)
