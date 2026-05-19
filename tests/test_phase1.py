"""
Tests for Phase 1 — Netlist Parser, Circuit Graph Builder, and Visualizer.

All tests use the sample_netlist.json (ESP32 + LED circuit, 4 components, 4 nets).
The visualizer test redirects the output path so it does not depend on outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_eda_engine import (
    CircuitGraph,
    GridMetadata,
    InitialPlacer,
    NetlistParser,
    half_perimeter_wire_length,
    visualize,
    _SAMPLE_JSON,
)

# ---------------------------------------------------------------------------
# Shared fixture: parsed netlist from sample_netlist.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_netlist() -> dict:
    with _SAMPLE_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def netlist(raw_netlist):
    return NetlistParser().parse(raw_netlist)


@pytest.fixture(scope="module")
def placed_netlist(raw_netlist):
    nl = NetlistParser().parse(raw_netlist)
    InitialPlacer(nl.metadata).place(nl)
    return nl


@pytest.fixture(scope="module")
def graph(placed_netlist):
    return CircuitGraph.from_netlist(placed_netlist)


# ---------------------------------------------------------------------------
# 1. Parser — component count
# ---------------------------------------------------------------------------

def test_parser_components_count(netlist):
    """Sample netlist has exactly 4 components."""
    assert len(netlist.components) == 4


# ---------------------------------------------------------------------------
# 2. Parser — net count
# ---------------------------------------------------------------------------

def test_parser_nets_count(netlist):
    """Sample netlist has exactly 4 nets."""
    assert len(netlist.nets) == 4


# ---------------------------------------------------------------------------
# 3. Parser — all pin types valid
# ---------------------------------------------------------------------------

_VALID_PIN_TYPES = {"OUTPUT", "INPUT", "PASSIVE", "POWER", "BIDIRECTIONAL"}

def test_parser_pin_types_valid(netlist):
    """Every pin in every component must have a recognised pin_type."""
    for comp in netlist.components:
        for pin in comp.pins:
            assert pin.pin_type in _VALID_PIN_TYPES, (
                f"Component {comp.id} pin {pin.id} has invalid type {pin.pin_type!r}"
            )


# ---------------------------------------------------------------------------
# 4. Graph — adjacency: U1 must connect to R1 and C1
# ---------------------------------------------------------------------------

def test_graph_adjacency_u1_connected(graph):
    """U1 (MCU) must be adjacent to both R1 and C1 after star-expansion."""
    assert "U1" in graph.adjacency, "U1 not found in graph adjacency"
    assert "R1" in graph.adjacency["U1"], "U1 should be adjacent to R1"
    assert "C1" in graph.adjacency["U1"], "U1 should be adjacent to C1"


# ---------------------------------------------------------------------------
# 5. Graph — edge count
# ---------------------------------------------------------------------------

def test_graph_edge_count(graph):
    """Star-expansion of 4 nets (2+3+2+2 pins) should yield 5 edges.

    VCC (2 pins): 1 edge
    GND (3 pins): 2 edges  (star from U1 to D1 and C1)
    LED_DRIVE (2 pins): 1 edge
    LED_ANODE (2 pins): 1 edge
    Total: 5
    """
    assert len(graph.edges) == 5


# ---------------------------------------------------------------------------
# 6. Placer — no overlaps
# ---------------------------------------------------------------------------

def test_placer_no_overlaps(placed_netlist):
    """After placement, no two component footprints may overlap."""
    placer = InitialPlacer(placed_netlist.metadata)
    comps = placed_netlist.components
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            assert not placer._overlaps(a, b), (
                f"Components {a.id} and {b.id} overlap after placement"
            )


# ---------------------------------------------------------------------------
# 7. Placer — all components in bounds
# ---------------------------------------------------------------------------

def test_placer_in_bounds(placed_netlist):
    """All component footprints must lie entirely within the 24x20 grid."""
    placer = InitialPlacer(placed_netlist.metadata)
    for comp in placed_netlist.components:
        assert placer._in_bounds(comp), (
            f"Component {comp.id} at ({comp.x},{comp.y}) with footprint "
            f"{comp.footprint.width}x{comp.footprint.height} is out of bounds"
        )


# ---------------------------------------------------------------------------
# 8. HPWL — positive float
# ---------------------------------------------------------------------------

def test_hpwl_positive(graph):
    """HPWL must be a positive float for any non-trivial placement."""
    hpwl = half_perimeter_wire_length(graph)
    assert isinstance(hpwl, float), "HPWL must be a float"
    assert hpwl > 0.0, f"HPWL must be positive, got {hpwl}"


def test_hpwl_sample_value(graph):
    """With the sample netlist seed positions, HPWL should equal 27.0."""
    hpwl = half_perimeter_wire_length(graph)
    assert abs(hpwl - 27.0) < 1e-6, f"Expected HPWL=27.0, got {hpwl}"


# ---------------------------------------------------------------------------
# 9. Visualizer — saves PNG to the given path
# ---------------------------------------------------------------------------

def test_visualizer_saves_png(graph, tmp_path):
    """visualize() must write a .png file at the given output path."""
    out = tmp_path / "phase1_output.png"
    result = visualize(graph, out)
    assert result == out
    assert out.exists(), "PNG file was not created"
    assert out.stat().st_size > 10_000, "PNG file is suspiciously small"


# ---------------------------------------------------------------------------
# 10. run_phase1 integration — creates outputs/phase1_output.png
# ---------------------------------------------------------------------------

def test_visualizer_saves_png_via_pipeline(raw_netlist, tmp_path, monkeypatch):
    """run_phase1 with a monkeypatched output dir must write phase1_output.png."""
    monkeypatch.setattr("phase1_eda_engine._OUTPUT_DIR", tmp_path)

    from phase1_eda_engine import run_phase1
    graph = run_phase1(raw_netlist)

    expected = tmp_path / "phase1_output.png"
    assert expected.exists(), "run_phase1 did not create the PNG"
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 5


# ---------------------------------------------------------------------------
# 11. Parser — component IDs match expected values
# ---------------------------------------------------------------------------

def test_parser_component_ids(netlist):
    """The four expected component IDs must be present."""
    ids = {c.id for c in netlist.components}
    assert ids == {"U1", "R1", "D1", "C1"}


# ---------------------------------------------------------------------------
# 12. Parser — net types are recognised values
# ---------------------------------------------------------------------------

_VALID_NET_TYPES = {"POWER", "SIGNAL", "GROUND"}

def test_parser_net_types_valid(netlist):
    """Every net must have a recognised net_type."""
    for net in netlist.nets:
        assert net.net_type in _VALID_NET_TYPES, (
            f"Net {net.id!r} has invalid type {net.net_type!r}"
        )


# ---------------------------------------------------------------------------
# 13. Placer — pin abs positions are set (not default 0)
# ---------------------------------------------------------------------------

def test_placer_pin_positions_set(placed_netlist):
    """After placement, at least some pins must have non-zero abs_x or abs_y."""
    any_nonzero = any(
        pin.abs_x != 0.0 or pin.abs_y != 0.0
        for comp in placed_netlist.components
        for pin in comp.pins
    )
    assert any_nonzero, "No pin positions were updated after placement"


# ---------------------------------------------------------------------------
# 14. CircuitGraph — nodes dict keyed by component ID
# ---------------------------------------------------------------------------

def test_graph_nodes_keyed_by_id(graph):
    """graph.nodes must be a dict where keys match component IDs."""
    for cid, comp in graph.nodes.items():
        assert cid == comp.id, f"Node key {cid!r} does not match comp.id {comp.id!r}"


# ---------------------------------------------------------------------------
# 15. Fallback placer — produces valid placement for a pathological input
# ---------------------------------------------------------------------------

def test_placer_fallback_on_invalid_positions():
    """InitialPlacer must fix a netlist where all components are stacked at (0,0)."""
    nl = NetlistParser().parse({
        "netlist": {
            "metadata": {"grid": {"width": 24, "height": 20, "unit": "mm"}},
            "components": [
                {"id": "U1", "type": "MCU",      "name": "ESP32",
                 "footprint": {"width": 4, "height": 6}, "x": 0, "y": 0,
                 "pins": [{"id": "VCC", "type": "POWER", "net": "VCC"}]},
                {"id": "R1", "type": "RESISTOR",  "name": "R1",
                 "footprint": {"width": 1, "height": 2}, "x": 0, "y": 0,
                 "pins": [{"id": "P1", "type": "PASSIVE", "net": "VCC"}]},
                {"id": "C1", "type": "CAPACITOR", "name": "C1",
                 "footprint": {"width": 1, "height": 1}, "x": 0, "y": 0,
                 "pins": [{"id": "P1", "type": "PASSIVE", "net": "VCC"}]},
            ],
            "nets": [
                {"id": "VCC", "type": "POWER", "connected_pins": [
                    {"component_id": "U1", "pin_id": "VCC"},
                    {"component_id": "R1", "pin_id": "P1"},
                    {"component_id": "C1", "pin_id": "P1"},
                ]}
            ],
        }
    })
    placer = InitialPlacer(nl.metadata)
    placer.place(nl)

    # After fallback, must be valid
    comps = nl.components
    for comp in comps:
        assert placer._in_bounds(comp), f"{comp.id} is out of bounds after fallback"
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            assert not placer._overlaps(a, b), f"{a.id} and {b.id} overlap after fallback"
