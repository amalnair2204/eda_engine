"""Tests for Phase 9 — Manufacturing export (Gerber / Drill / BOM / KiCad).

One dedicated test per behaviour (no bundling).  A module-scoped fixture runs
the placement + routing + export pipeline once and the tests inspect its output.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest
import gerbonara as gn

from phase1_eda_engine import NetlistParser, InitialPlacer, CircuitGraph
from phase2_genetic_placer import run_phase2
from phase8_multilayer_router import run_phase8
from phase9_export import run_phase9, derive_refdes

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"


@pytest.fixture(scope="module")
def exported():
    """Place + route (multi-layer) + export the sample board once."""
    raw = json.loads(_SAMPLE_JSON.read_text(encoding="utf-8"))
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)
    graph = run_phase2(graph, pop_size=20, n_generations=30)
    graph, traces, _m = run_phase8(graph)
    result = run_phase9(graph, traces)
    return graph, traces, result


# ---------------------------------------------------------------------------
# test_gerbers_reparse_and_layer_count
# ---------------------------------------------------------------------------

def test_gerbers_reparse_and_layer_count(exported) -> None:
    """Every copper Gerber re-parses; file count == number of routed layers."""
    graph, traces, result = exported

    # Re-parse every copper Gerber with gerbonara (no exceptions).
    for path in result["copper_gerbers"].values():
        gf = gn.GerberFile.open(str(path))
        assert gf is not None
        list(gf.objects)   # force evaluation

    routed_layers = {L for t in traces for L in getattr(t, "layers", [0])}
    routed_layers.add(0)
    assert len(result["copper_gerbers"]) == len(routed_layers)


# ---------------------------------------------------------------------------
# test_drill_parses_and_hit_count
# ---------------------------------------------------------------------------

def test_drill_parses_and_hit_count(exported) -> None:
    """Drill file parses; hit count == via count + through-hole pad count."""
    graph, traces, result = exported

    ef = gn.ExcellonFile.open(str(result["drill"]))
    total_hits = sum(ef.hit_count().values())

    n_pins = sum(len(c.pins) for c in graph.nodes.values())
    n_vias = len({tuple(v) for t in traces for v in getattr(t, "vias", [])})
    assert total_hits == n_pins + n_vias
    assert result["drill_hits"] == n_pins + n_vias
    assert len(ef.drill_sizes()) >= 1   # tool sizes defined


# ---------------------------------------------------------------------------
# test_bom_grouped_and_total_quantity
# ---------------------------------------------------------------------------

def test_bom_grouped_and_total_quantity(exported) -> None:
    """bom.csv has one row per unique part; total quantity == component count."""
    graph, _traces, result = exported
    rows = list(csv.DictReader(result["bom"].open(encoding="utf-8")))
    assert rows, "BOM must not be empty"

    # One row per unique (Type, Value, Footprint).
    keys = {(r["Type"], r["Value"], r["Footprint"]) for r in rows}
    assert len(keys) == len(rows)

    total_qty = sum(int(r["Quantity"]) for r in rows)
    assert total_qty == len(graph.nodes)


# ---------------------------------------------------------------------------
# test_kicad_netlist_contains_all_components_and_nets
# ---------------------------------------------------------------------------

def test_kicad_netlist_contains_all_components_and_nets(exported) -> None:
    """KiCad .net lists every component ref and every net name in the graph."""
    graph, _traces, result = exported
    text = result["kicad_netlist"].read_text(encoding="utf-8")

    assert text.lstrip().startswith("(export")

    for ref in derive_refdes(graph).values():
        assert f'(ref "{ref}")' in text, f"missing component ref {ref}"

    net_names = {e.net_id for e in graph.edges}
    net_names |= {p.net for c in graph.nodes.values() for p in c.pins if p.net}
    for name in net_names:
        assert f'(name "{name}")' in text, f"missing net {name}"


# ---------------------------------------------------------------------------
# test_fab_zip_contains_gerbers_and_drill
# ---------------------------------------------------------------------------

def test_fab_zip_contains_gerbers_and_drill(exported) -> None:
    """The fab zip exists and contains copper + outline Gerbers + the drill."""
    _graph, _traces, result = exported
    zpath = result["zip"]
    assert zpath.exists()

    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    # All copper layers present.
    for p in result["copper_gerbers"].values():
        assert p.name in names
    # Outline + drill present.
    assert result["outline_gerber"].name in names
    assert result["drill"].name in names
    assert any(n.endswith("Edge_Cuts.gbr") for n in names)


# ---------------------------------------------------------------------------
# test_run_phase9_reports_unrouted_for_partial_board
# ---------------------------------------------------------------------------

def test_run_phase9_reports_unrouted_for_partial_board(exported) -> None:
    """run_phase9 reports the unrouted-net count for a partially-routed board."""
    graph, traces, _result = exported
    # Drop all but the first trace → most nets now incomplete.
    partial = run_phase9(graph, traces[:1])
    assert partial["unrouted_nets"] > 0
    assert partial["completion_pct"] < 100.0
