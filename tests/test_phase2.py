"""Tests for Phase 2 — Genetic Algorithm Placement Optimizer."""

from __future__ import annotations

import copy
import json
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
from phase2_genetic_placer import (
    Chromosome,
    FitnessEvaluator,
    GeneticPlacer,
    run_phase2,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"


@pytest.fixture
def sample_graph() -> CircuitGraph:
    """Standard 4-component graph loaded from sample_netlist.json."""
    with _SAMPLE_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    return CircuitGraph.from_netlist(netlist)


def _two_mcu_graph() -> CircuitGraph:
    """Minimal graph with two MCU components on the same VCC net."""
    meta = GridMetadata(width=20, height=20, unit="mm", name="test")
    mcu1 = Component(
        id="U1", comp_type="MCU", name="MCU1",
        pins=[Pin(id="VCC", pin_type="POWER", net="VCC")],
        footprint=Footprint(width=3, height=3),
        x=4, y=4, properties={},
    )
    mcu2 = Component(
        id="U2", comp_type="MCU", name="MCU2",
        pins=[Pin(id="VCC", pin_type="POWER", net="VCC")],
        footprint=Footprint(width=3, height=3),
        x=12, y=12, properties={},
    )
    edge = GraphEdge(
        net_id="VCC", net_type="POWER",
        source=("U1", "VCC"), target=("U2", "VCC"),
    )
    return CircuitGraph(
        nodes={"U1": mcu1, "U2": mcu2},
        edges=[edge],
        adjacency={"U1": {"U2"}, "U2": {"U1"}},
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# test_chromosome_creation
# ---------------------------------------------------------------------------

def test_chromosome_creation(sample_graph: CircuitGraph) -> None:
    """Chromosome holds exactly one (x, y) entry per component."""
    placer = GeneticPlacer(sample_graph, pop_size=5, n_generations=1)
    chrom  = placer._random_chromosome()
    assert set(chrom.positions.keys()) == set(sample_graph.nodes.keys())
    assert len(chrom.positions) == len(sample_graph.nodes)
    for pos in chrom.positions.values():
        assert len(pos) == 2


# ---------------------------------------------------------------------------
# test_fitness_overlap_penalty
# ---------------------------------------------------------------------------

def test_fitness_overlap_penalty(sample_graph: CircuitGraph) -> None:
    """Two overlapping components produce a higher fitness than when separated."""
    evaluator = FitnessEvaluator(sample_graph)
    comp_ids  = list(sample_graph.nodes.keys())
    a_id, b_id = comp_ids[0], comp_ids[1]

    # Baseline: use current Phase 1 positions (no overlap)
    base_pos = {cid: (comp.x, comp.y) for cid, comp in sample_graph.nodes.items()}
    base = Chromosome(positions=copy.deepcopy(base_pos))
    evaluator.evaluate(base)

    # Overlapping: place both at (0, 0)
    overlap_pos = copy.deepcopy(base_pos)
    overlap_pos[a_id] = (0, 0)
    overlap_pos[b_id] = (0, 0)
    overlapping = Chromosome(positions=overlap_pos)
    evaluator.evaluate(overlapping)

    assert overlapping.fitness > base.fitness


# ---------------------------------------------------------------------------
# test_fitness_hpwl_closer_is_better
# ---------------------------------------------------------------------------

def test_fitness_hpwl_closer_is_better(sample_graph: CircuitGraph) -> None:
    """Components on shared nets placed closer together score lower HPWL."""
    evaluator = FitnessEvaluator(sample_graph)
    base_pos  = {cid: (comp.x, comp.y) for cid, comp in sample_graph.nodes.items()}

    # Near: all components at roughly the same spot (no overlap — offset by 1)
    near_pos = {}
    for i, (cid, comp) in enumerate(sample_graph.nodes.items()):
        near_pos[cid] = (i * (comp.footprint.width + 1), 0)
    near = Chromosome(positions=near_pos)

    # Far: components spread to opposite corners
    far_pos = copy.deepcopy(base_pos)
    comp_list = list(sample_graph.nodes.values())
    gw = sample_graph.metadata.width
    gh = sample_graph.metadata.height
    for i, comp in enumerate(comp_list):
        if i % 2 == 0:
            far_pos[comp.id] = (0, 0)
        else:
            far_pos[comp.id] = (
                max(0, gw - comp.footprint.width),
                max(0, gh - comp.footprint.height),
            )
    far = Chromosome(positions=far_pos)

    evaluator.evaluate(near)
    evaluator.evaluate(far)

    assert near.fitness < far.fitness


# ---------------------------------------------------------------------------
# test_fitness_thermal_penalty
# ---------------------------------------------------------------------------

def test_fitness_thermal_penalty() -> None:
    """Two MCUs within 3 cells incur a thermal penalty; far apart do not."""
    graph     = _two_mcu_graph()
    evaluator = FitnessEvaluator(graph)

    # Close: centers ~1 cell apart (both at adjacent positions)
    close = Chromosome(positions={"U1": (0, 0), "U2": (1, 0)})
    evaluator.evaluate(close)

    # Far: centers > 3 cells apart
    far = Chromosome(positions={"U1": (0, 0), "U2": (10, 10)})
    evaluator.evaluate(far)

    assert close.fitness > far.fitness


# ---------------------------------------------------------------------------
# test_no_overlaps_in_initial_population
# ---------------------------------------------------------------------------

def test_no_overlaps_in_initial_population(sample_graph: CircuitGraph) -> None:
    """Every chromosome in the initial population is overlap-free."""
    placer     = GeneticPlacer(sample_graph, pop_size=20, n_generations=1)
    population = placer._initialize_population()
    evaluator  = FitnessEvaluator(sample_graph)
    comps      = list(sample_graph.nodes.values())

    for chrom in population:
        for i, a in enumerate(comps):
            for b in comps[i + 1:]:
                assert not evaluator.overlaps_at(
                    a, chrom.positions[a.id],
                    b, chrom.positions[b.id],
                ), f"Overlap between {a.id} and {b.id} in initial population"


# ---------------------------------------------------------------------------
# test_crossover_preserves_all_components
# ---------------------------------------------------------------------------

def test_crossover_preserves_all_components(sample_graph: CircuitGraph) -> None:
    """Each child from crossover contains exactly the same component IDs as parents."""
    placer = GeneticPlacer(sample_graph, pop_size=5, n_generations=1)
    pa     = placer._random_chromosome()
    pb     = placer._random_chromosome()
    c1, c2 = placer._crossover(pa, pb)

    expected = set(pa.positions.keys())
    assert set(c1.positions.keys()) == expected
    assert set(c2.positions.keys()) == expected


# ---------------------------------------------------------------------------
# test_mutation_stays_in_bounds
# ---------------------------------------------------------------------------

def test_mutation_stays_in_bounds(sample_graph: CircuitGraph) -> None:
    """All positions after mutation lie within the grid for every component."""
    placer = GeneticPlacer(sample_graph, pop_size=5, n_generations=1,
                           mutation_rate=1.0, seed=0)
    original = placer._random_chromosome()

    for _ in range(20):
        mutated = placer._mutate(original)
        for cid, (x, y) in mutated.positions.items():
            comp = sample_graph.nodes[cid]
            assert x >= 0, f"{cid}.x={x} < 0"
            assert y >= 0, f"{cid}.y={y} < 0"
            assert x + comp.footprint.width  <= sample_graph.metadata.width,  \
                f"{cid} exceeds grid width"
            assert y + comp.footprint.height <= sample_graph.metadata.height, \
                f"{cid} exceeds grid height"


# ---------------------------------------------------------------------------
# test_repair_resolves_overlaps
# ---------------------------------------------------------------------------

def test_repair_resolves_overlaps(sample_graph: CircuitGraph) -> None:
    """After repair, no pair of components shares overlapping cells."""
    placer    = GeneticPlacer(sample_graph, pop_size=5, n_generations=1)
    evaluator = FitnessEvaluator(sample_graph)

    # Force an overlap by placing all components at (0, 0)
    bad_pos   = {cid: (0, 0) for cid in sample_graph.nodes}
    bad_chrom = Chromosome(positions=bad_pos)
    repaired  = placer._repair_overlaps(bad_chrom)

    comps = list(sample_graph.nodes.values())
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            assert not evaluator.overlaps_at(
                a, repaired.positions[a.id],
                b, repaired.positions[b.id],
            ), f"Overlap persists between {a.id} and {b.id} after repair"


# ---------------------------------------------------------------------------
# test_ga_improves_fitness
# ---------------------------------------------------------------------------

def test_ga_improves_fitness(sample_graph: CircuitGraph) -> None:
    """Fitness after 50 GA generations is lower than (or equal to) Gen 0 fitness."""
    placer  = GeneticPlacer(sample_graph, pop_size=20, n_generations=50, seed=7)
    _best, history = placer.run()
    assert history[-1] <= history[0], (
        f"GA made things worse: Gen0={history[0]:.2f}, Gen50={history[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# test_run_phase2_returns_graph
# ---------------------------------------------------------------------------

def test_run_phase2_returns_graph(sample_graph: CircuitGraph) -> None:
    """run_phase2() returns a CircuitGraph instance."""
    result = run_phase2(sample_graph, pop_size=10, n_generations=10)
    assert isinstance(result, CircuitGraph)


# ---------------------------------------------------------------------------
# test_run_phase2_hpwl_improves
# ---------------------------------------------------------------------------

def test_run_phase2_hpwl_improves() -> None:
    """HPWL after Phase 2 is lower than or equal to the Phase 1 seed HPWL."""
    from phase1_eda_engine import half_perimeter_wire_length

    with _SAMPLE_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    hpwl_before = half_perimeter_wire_length(graph)
    run_phase2(graph, pop_size=20, n_generations=50)
    hpwl_after  = half_perimeter_wire_length(graph)

    assert hpwl_after <= hpwl_before + 0.01, (
        f"HPWL did not improve: before={hpwl_before:.2f}, after={hpwl_after:.2f}"
    )
