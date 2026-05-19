"""
Phase 2 — Genetic Algorithm Placement Optimizer

Sections
--------
1. Constants & colour palette
2. Chromosome dataclass
3. FitnessEvaluator  — EEE-aware weighted scoring
4. GeneticPlacer     — GA engine: init, selection, crossover, mutation, repair
5. Visualizer        — two-panel dark-mode canvas + convergence chart
6. run_phase2()      — pipeline entry-point
7. CLI entry-point
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import numpy as np

from phase1_eda_engine import (
    CircuitGraph,
    Component,
    half_perimeter_wire_length,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Colour palette — mirrors Phase 1 dark-mode aesthetic
# ---------------------------------------------------------------------------
_BG       = "#0f0f1a"
_PANEL_BG = "#16162a"
_GRID_C   = "#1e1e3a"
_TEXT_C   = "#e0e0ff"
_DIM_C    = "#888899"
_EMERALD  = "#00C97A"

_COMP_COLORS: dict[str, str] = {
    "MCU":       "#1E90FF",
    "RESISTOR":  "#FF8C00",
    "CAPACITOR": "#00CED1",
    "LED":       "#FFD700",
    "IC":        "#9370DB",
    "POWER":     "#FF4444",
}
_NET_COLORS: dict[str, str] = {
    "POWER":  "#FF6B6B",
    "GROUND": "#69FF69",
    "SIGNAL": "#69B4FF",
}

# ---------------------------------------------------------------------------
# GA hyper-parameters
# ---------------------------------------------------------------------------
POP_SIZE      = 50
N_GENERATIONS = 200
MUTATION_RATE = 0.15
ELITE_FRAC    = 0.1
TOURNAMENT_K  = 5

# ---------------------------------------------------------------------------
# EEE component categories
# ---------------------------------------------------------------------------
_HEAT_SOURCES = {"MCU", "IC"}
_HIGH_FREQ    = {"MCU", "IC"}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Chromosome
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chromosome:
    """One candidate placement — a complete assignment of (x, y) to every component.

    Attributes:
        positions: Maps component_id → top-left grid coordinate (x, y).
        fitness:   Weighted penalty score; lower is better.  Set by FitnessEvaluator.
    """

    positions: dict[str, tuple[int, int]]
    fitness: float = field(default_factory=lambda: float("inf"))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — FitnessEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class FitnessEvaluator:
    """EEE-aware fitness scorer for GA chromosomes.

    Scoring terms (lower total = better placement):
      1. HPWL           — half-perimeter wire length          (weight  1.0)
      2. Overlap        — footprint collision cells × 100     (weight 10.0)
      3. Thermal        — heat-source proximity < 3 cells     (weight  2.0)
      4. Edge proximity — MCU/IC within 1 cell of boundary    (weight  1.5)
      5. Decoupling cap — cap within 2 cells of MCU/IC power  (weight -1.5, reward)
    """

    _W_HPWL    =  1.0
    _W_OVERLAP = 10.0
    _W_THERMAL =  2.0
    _W_EDGE    =  1.5
    _W_DECAP   = -1.5   # negative = reward (subtracted from score)

    def __init__(self, graph: CircuitGraph) -> None:
        """Pre-compute net membership and reward structures from the graph.

        Args:
            graph: CircuitGraph produced by Phase 1.
        """
        self._graph   = graph
        self._grid_w  = graph.metadata.width
        self._grid_h  = graph.metadata.height

        # net_id → set of comp_ids appearing in that net's edges
        self._net_comps: dict[str, set[str]] = defaultdict(set)
        # net_id → net_type
        self._net_types: dict[str, str] = {}
        # comp_id → list of net_ids it participates in
        self._comp_nets: dict[str, list[str]] = defaultdict(list)

        for edge in graph.edges:
            self._net_comps[edge.net_id].add(edge.source[0])
            self._net_comps[edge.net_id].add(edge.target[0])
            self._net_types[edge.net_id] = edge.net_type

        for net_id, comp_ids in self._net_comps.items():
            for cid in comp_ids:
                if net_id not in self._comp_nets[cid]:
                    self._comp_nets[cid].append(net_id)

        power_nets: set[str] = {
            nid for nid, ntype in self._net_types.items() if ntype == "POWER"
        }

        # Pre-compute (cap_id, [mcu_ic_ids_on_same_power_net]) for decoupling reward
        self._decap_pairs: list[tuple[str, list[str]]] = []
        for cid, comp in graph.nodes.items():
            if comp.comp_type != "CAPACITOR":
                continue
            for net_id in self._comp_nets.get(cid, []):
                if net_id not in power_nets:
                    continue
                mcu_ic = [
                    oid for oid in self._net_comps[net_id]
                    if graph.nodes[oid].comp_type in _HEAT_SOURCES and oid != cid
                ]
                if mcu_ic:
                    self._decap_pairs.append((cid, mcu_ic))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, chromosome: Chromosome) -> float:
        """Score a chromosome, set chromosome.fitness, and return the score.

        Args:
            chromosome: Candidate layout to score.

        Returns:
            Weighted fitness score (lower is better).
        """
        score = (
            self._W_HPWL    * self._hpwl(chromosome)
            + self._W_OVERLAP * self._overlap_penalty(chromosome)
            + self._W_THERMAL * self._thermal_penalty(chromosome)
            + self._W_EDGE    * self._edge_proximity_penalty(chromosome)
            + self._W_DECAP   * self._decap_reward(chromosome)
        )
        chromosome.fitness = score
        return score

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center(comp: Component, pos: tuple[int, int]) -> tuple[float, float]:
        """Return the grid center of a component placed at pos."""
        return (
            pos[0] + comp.footprint.width  / 2.0,
            pos[1] + comp.footprint.height / 2.0,
        )

    @staticmethod
    def overlaps_at(
        a: Component, pos_a: tuple[int, int],
        b: Component, pos_b: tuple[int, int],
    ) -> bool:
        """Return True if component footprints overlap at the given positions."""
        ax, ay = pos_a
        bx, by = pos_b
        return not (
            ax + a.footprint.width  <= bx
            or bx + b.footprint.width  <= ax
            or ay + a.footprint.height <= by
            or by + b.footprint.height <= ay
        )

    # ------------------------------------------------------------------
    # Penalty / reward terms
    # ------------------------------------------------------------------

    def _hpwl(self, chromosome: Chromosome) -> float:
        """HPWL summed across all nets using chromosome positions."""
        total = 0.0
        for net_id, comp_ids in self._net_comps.items():
            xs, ys = [], []
            for cid in comp_ids:
                comp = self._graph.nodes[cid]
                cx, cy = self._center(comp, chromosome.positions[cid])
                xs.append(cx)
                ys.append(cy)
            if xs:
                total += (max(xs) - min(xs)) + (max(ys) - min(ys))
        return total

    def _overlap_penalty(self, chromosome: Chromosome) -> float:
        """100.0 per overlapping cell for every component pair."""
        penalty = 0.0
        comps = list(self._graph.nodes.values())
        for i, a in enumerate(comps):
            pos_a = chromosome.positions[a.id]
            for b in comps[i + 1:]:
                pos_b = chromosome.positions[b.id]
                if not self.overlaps_at(a, pos_a, b, pos_b):
                    continue
                ax, ay = pos_a
                bx, by = pos_b
                ox = min(ax + a.footprint.width,  bx + b.footprint.width)  - max(ax, bx)
                oy = min(ay + a.footprint.height, by + b.footprint.height) - max(ay, by)
                penalty += 100.0 * max(0, ox) * max(0, oy)
        return penalty

    def _thermal_penalty(self, chromosome: Chromosome) -> float:
        """50.0 for each pair of heat-source components within 3 grid cells."""
        penalty = 0.0
        heat = [c for c in self._graph.nodes.values() if c.comp_type in _HEAT_SOURCES]
        for i, a in enumerate(heat):
            cx_a, cy_a = self._center(a, chromosome.positions[a.id])
            for b in heat[i + 1:]:
                cx_b, cy_b = self._center(b, chromosome.positions[b.id])
                if math.hypot(cx_a - cx_b, cy_a - cy_b) < 3.0:
                    penalty += 50.0
        return penalty

    def _edge_proximity_penalty(self, chromosome: Chromosome) -> float:
        """30.0 for each MCU/IC component within 1 cell of any grid boundary."""
        penalty = 0.0
        gw, gh = self._grid_w, self._grid_h
        for cid, comp in self._graph.nodes.items():
            if comp.comp_type not in _HIGH_FREQ:
                continue
            x, y = chromosome.positions[cid]
            fw, fh = comp.footprint.width, comp.footprint.height
            if x < 1 or x + fw > gw - 1 or y < 1 or y + fh > gh - 1:
                penalty += 30.0
        return penalty

    def _decap_reward(self, chromosome: Chromosome) -> float:
        """20.0 per decoupling cap within 2 cells of its MCU/IC power companion."""
        reward = 0.0
        for cap_id, mcu_ids in self._decap_pairs:
            cap_comp = self._graph.nodes[cap_id]
            cx_cap, cy_cap = self._center(cap_comp, chromosome.positions[cap_id])
            for mcu_id in mcu_ids:
                mcu_comp = self._graph.nodes[mcu_id]
                cx_m, cy_m = self._center(mcu_comp, chromosome.positions[mcu_id])
                if math.hypot(cx_cap - cx_m, cy_cap - cy_m) <= 2.0:
                    reward += 20.0
                    break  # each cap counts at most once
        return reward


# ═══════════════════════════════════════════════════════════════════════════════
# Utility — pin position sync
# ═══════════════════════════════════════════════════════════════════════════════

def _update_graph_pin_positions(graph: CircuitGraph) -> None:
    """Recompute Pin.abs_x / Pin.abs_y after component positions change.

    Mirrors Phase 1's InitialPlacer._update_pin_positions: pins are spread
    evenly along the bottom edge of their parent component.

    Args:
        graph: CircuitGraph whose Component.x / .y have just been updated.
    """
    for comp in graph.nodes.values():
        n  = len(comp.pins)
        cx = comp.x + comp.footprint.width / 2.0
        for i, pin in enumerate(comp.pins):
            if n == 1:
                pin.abs_x = cx
                pin.abs_y = float(comp.y)
            else:
                pin.abs_x = comp.x + (i + 1) * comp.footprint.width / (n + 1)
                pin.abs_y = float(comp.y)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — GeneticPlacer
# ═══════════════════════════════════════════════════════════════════════════════

class GeneticPlacer:
    """Genetic Algorithm placement optimizer (Strategy pattern).

    Args:
        graph:         Phase 1 CircuitGraph; Component.x/.y are mutated by run().
        pop_size:      Layouts per generation.
        n_generations: Evolution cycles to run.
        mutation_rate: Per-component mutation probability each generation.
        elite_frac:    Fraction of top chromosomes preserved unchanged.
        tournament_k:  Tournament pool size for parent selection.
        seed:          RNG seed for reproducibility.
    """

    def __init__(
        self,
        graph: CircuitGraph,
        pop_size: int       = POP_SIZE,
        n_generations: int  = N_GENERATIONS,
        mutation_rate: float = MUTATION_RATE,
        elite_frac: float   = ELITE_FRAC,
        tournament_k: int   = TOURNAMENT_K,
        seed: int           = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)

        self._graph     = graph
        self._pop_size  = pop_size
        self._n_gen     = n_generations
        self._mut_rate  = mutation_rate
        self._elite_n   = max(1, int(pop_size * elite_frac))
        self._tourn_k   = tournament_k
        self._grid_w    = graph.metadata.width
        self._grid_h    = graph.metadata.height
        self._evaluator = FitnessEvaluator(graph)
        self._comp_list = list(graph.nodes.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _valid_range(self, comp: Component) -> tuple[int, int, int, int]:
        """Return (x_min, x_max, y_min, y_max) for comp's top-left corner."""
        return (
            0,
            max(0, self._grid_w - comp.footprint.width),
            0,
            max(0, self._grid_h - comp.footprint.height),
        )

    def _rand_pos(self, comp: Component) -> tuple[int, int]:
        """Uniformly random in-bounds position for a component."""
        xmin, xmax, ymin, ymax = self._valid_range(comp)
        return (random.randint(xmin, xmax), random.randint(ymin, ymax))

    def _has_overlap_with_others(
        self, chromosome: Chromosome, target_id: str
    ) -> bool:
        """Return True if target_id overlaps any other component in chromosome."""
        target = self._graph.nodes[target_id]
        tpos   = chromosome.positions[target_id]
        for oid, other in self._graph.nodes.items():
            if oid == target_id:
                continue
            if self._evaluator.overlaps_at(target, tpos, other, chromosome.positions[oid]):
                return True
        return False

    # ------------------------------------------------------------------
    # D.1 — Random chromosome (no overlaps)
    # ------------------------------------------------------------------

    def _random_chromosome(self) -> Chromosome:
        """Place all components randomly with no overlaps.

        Tries up to 500 random positions per component; falls back to grid
        origin on exhaustion (repaired later).

        Returns:
            A Chromosome with all components placed on the grid.
        """
        positions: dict[str, tuple[int, int]] = {}
        placed: list[tuple[Component, tuple[int, int]]] = []

        for comp in self._comp_list:
            xmin, xmax, ymin, ymax = self._valid_range(comp)
            placed_ok = False
            for _ in range(500):
                x = random.randint(xmin, xmax)
                y = random.randint(ymin, ymax)
                if all(
                    not self._evaluator.overlaps_at(comp, (x, y), pc, pp)
                    for pc, pp in placed
                ):
                    positions[comp.id] = (x, y)
                    placed.append((comp, (x, y)))
                    placed_ok = True
                    break
            if not placed_ok:
                positions[comp.id] = (xmin, ymin)
                placed.append((comp, (xmin, ymin)))

        return Chromosome(positions=positions)

    # ------------------------------------------------------------------
    # D.2 — Initialize population
    # ------------------------------------------------------------------

    def _initialize_population(self) -> list[Chromosome]:
        """Generate the initial population of pop_size chromosomes.

        The first chromosome uses the Phase 1 seed positions so the GA
        always starts at least as good as the initial placement.

        Returns:
            List of pop_size Chromosomes (unsorted, unevaluated).
        """
        seed_pos = {cid: (comp.x, comp.y) for cid, comp in self._graph.nodes.items()}
        population: list[Chromosome] = [Chromosome(positions=seed_pos)]
        while len(population) < self._pop_size:
            population.append(self._random_chromosome())
        return population

    # ------------------------------------------------------------------
    # D.3 — Evaluate + sort population
    # ------------------------------------------------------------------

    def _evaluate_population(self, population: list[Chromosome]) -> list[Chromosome]:
        """Score every chromosome and return the list sorted best-first.

        Args:
            population: Chromosomes to score.

        Returns:
            Sorted list, ascending by fitness (lower = better).
        """
        for chrom in population:
            self._evaluator.evaluate(chrom)
        return sorted(population, key=lambda c: c.fitness)

    # ------------------------------------------------------------------
    # D.4 — Tournament selection
    # ------------------------------------------------------------------

    def _tournament_select(self, population: list[Chromosome]) -> Chromosome:
        """Return the fittest chromosome from a random pool of tournament_k.

        Args:
            population: Evaluated population to sample from.

        Returns:
            The chromosome with the lowest fitness in the drawn pool.
        """
        pool = random.sample(population, min(self._tourn_k, len(population)))
        return min(pool, key=lambda c: c.fitness)

    # ------------------------------------------------------------------
    # D.5 — Uniform crossover
    # ------------------------------------------------------------------

    def _crossover(
        self, parent_a: Chromosome, parent_b: Chromosome
    ) -> tuple[Chromosome, Chromosome]:
        """Uniform crossover: each component position drawn independently from one parent.

        Args:
            parent_a: First parent Chromosome.
            parent_b: Second parent Chromosome.

        Returns:
            Two child Chromosomes with overlaps repaired.
        """
        pos_c1: dict[str, tuple[int, int]] = {}
        pos_c2: dict[str, tuple[int, int]] = {}
        for cid in parent_a.positions:
            if random.random() < 0.5:
                pos_c1[cid] = parent_a.positions[cid]
                pos_c2[cid] = parent_b.positions[cid]
            else:
                pos_c1[cid] = parent_b.positions[cid]
                pos_c2[cid] = parent_a.positions[cid]
        return (
            self._repair_overlaps(Chromosome(positions=pos_c1)),
            self._repair_overlaps(Chromosome(positions=pos_c2)),
        )

    # ------------------------------------------------------------------
    # D.6 — Mutation
    # ------------------------------------------------------------------

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Randomly perturb component positions with probability mutation_rate.

        Per mutated component:
          - 50 % → jitter within ±3 cells (clamped to grid)
          - 50 % → teleport to a random valid position

        Args:
            chromosome: Source chromosome (not mutated in place; a copy is made).

        Returns:
            New mutated Chromosome with overlaps repaired.
        """
        new_pos = dict(chromosome.positions)
        for comp in self._comp_list:
            if random.random() >= self._mut_rate:
                continue
            xmin, xmax, ymin, ymax = self._valid_range(comp)
            if random.random() < 0.5:
                ox, oy = new_pos[comp.id]
                new_pos[comp.id] = (
                    max(xmin, min(xmax, ox + random.randint(-3, 3))),
                    max(ymin, min(ymax, oy + random.randint(-3, 3))),
                )
            else:
                new_pos[comp.id] = (
                    random.randint(xmin, xmax),
                    random.randint(ymin, ymax),
                )
        return self._repair_overlaps(Chromosome(positions=new_pos))

    # ------------------------------------------------------------------
    # D.7 — Overlap repair
    # ------------------------------------------------------------------

    def _repair_overlaps(self, chromosome: Chromosome) -> Chromosome:
        """Resolve overlapping component pairs by randomly re-placing the offender.

        Iterates over all pairs; the second component of each overlapping pair
        is relocated.  Caps attempts per component at 500 to avoid infinite loops.

        Args:
            chromosome: Chromosome that may contain overlapping footprints.

        Returns:
            The same Chromosome object with best-effort overlap resolution.
        """
        comps = self._comp_list
        for i, a in enumerate(comps):
            for b in comps[i + 1:]:
                if not self._evaluator.overlaps_at(
                    a, chromosome.positions[a.id],
                    b, chromosome.positions[b.id],
                ):
                    continue
                xmin, xmax, ymin, ymax = self._valid_range(b)
                for _ in range(500):
                    chromosome.positions[b.id] = (
                        random.randint(xmin, xmax),
                        random.randint(ymin, ymax),
                    )
                    if not self._has_overlap_with_others(chromosome, b.id):
                        break
        return chromosome

    # ------------------------------------------------------------------
    # D.8 — Main GA loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[Chromosome, list[float]]:
        """Execute the Genetic Algorithm and apply the best layout to the graph.

        Steps:
          1. Initialise and evaluate population.
          2. For each generation: elitism → crossover → mutation → evaluate.
          3. Apply best chromosome positions to graph.nodes.
          4. Update Pin.abs_x / Pin.abs_y.

        Returns:
            (best_chromosome, fitness_history) where fitness_history[i] is
            the best fitness at generation i (index 0 = initial population).
        """
        population = self._evaluate_population(self._initialize_population())
        fitness_history: list[float] = [population[0].fitness]

        for gen in range(1, self._n_gen + 1):
            new_pop: list[Chromosome] = [copy.deepcopy(c) for c in population[:self._elite_n]]

            while len(new_pop) < self._pop_size:
                pa = self._tournament_select(population)
                pb = self._tournament_select(population)
                c1, c2 = self._crossover(pa, pb)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self._pop_size:
                    new_pop.append(self._mutate(c2))

            population = self._evaluate_population(new_pop)
            fitness_history.append(population[0].fitness)

            if gen % 25 == 0:
                hpwl = self._evaluator._hpwl(population[0])
                print(
                    f"  Gen {gen:>4}/{self._n_gen}  |  "
                    f"Best HPWL: {hpwl:>7.2f}  |  "
                    f"Best Fitness: {population[0].fitness:>8.2f}"
                )

        best = population[0]
        for cid, comp in self._graph.nodes.items():
            comp.x, comp.y = best.positions[cid]
        _update_graph_pin_positions(self._graph)
        return best, fitness_history


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_ga_results(
    graph: CircuitGraph,
    fitness_history: list[float],
    seed_fitness: float,
    output_path: Path | None = None,
) -> Path:
    """Render a two-panel dark-mode figure: optimized layout + convergence chart.

    Args:
        graph:          CircuitGraph with GA-optimized component positions.
        fitness_history: Best fitness per generation (index 0 = initial pop).
        seed_fitness:   Phase 1 seed fitness (draws the baseline dashed line).
        output_path:    Destination PNG; defaults to outputs/phase2_output.png.

    Returns:
        Path where the PNG was saved.
    """
    if output_path is None:
        output_path = _OUTPUT_DIR / "phase2_output.png"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 10), facecolor=_BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.08,
                   left=0.04, right=0.97, top=0.93, bottom=0.09)
    ax_layout = fig.add_subplot(gs[0])
    ax_conv   = fig.add_subplot(gs[1])

    gw = graph.metadata.width
    gh = graph.metadata.height

    # ── Left: Optimized layout ─────────────────────────────────────────
    ax_layout.set_facecolor(_BG)
    ax_layout.set_xlim(-0.5, gw + 0.5)
    ax_layout.set_ylim(-0.5, gh + 0.5)
    ax_layout.set_aspect("equal")
    ax_layout.tick_params(colors=_DIM_C, labelsize=7)
    for spine in ax_layout.spines.values():
        spine.set_color(_GRID_C)
    ax_layout.set_title(
        f"Phase 2 — GA Optimized Layout  ({gw} × {gh} {graph.metadata.unit})",
        color=_TEXT_C, fontsize=11, pad=8, fontweight="bold",
    )
    ax_layout.set_xlabel("x (grid units)", color=_DIM_C, fontsize=8)
    ax_layout.set_ylabel("y (grid units)", color=_DIM_C, fontsize=8)

    for x in range(gw + 1):
        ax_layout.axvline(x, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)
    for y in range(gh + 1):
        ax_layout.axhline(y, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)

    drawn: set[str] = set()
    for edge in graph.edges:
        src = graph.nodes[edge.source[0]]
        tgt = graph.nodes[edge.target[0]]
        sx  = src.x + src.footprint.width  / 2.0
        sy  = src.y + src.footprint.height / 2.0
        tx  = tgt.x + tgt.footprint.width  / 2.0
        ty  = tgt.y + tgt.footprint.height / 2.0
        col = _NET_COLORS.get(edge.net_type, "#aaaaaa")
        ax_layout.plot([sx, tx], [sy, ty], color=col, lw=1.5, alpha=0.65,
                        linestyle="--" if edge.net_type == "GROUND" else "-", zorder=2)
        if edge.net_id not in drawn:
            ax_layout.text(
                (sx + tx) / 2, (sy + ty) / 2 + 0.25, edge.net_id,
                color=col, fontsize=5.5, ha="center", va="bottom", alpha=0.9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc=_BG, ec="none", alpha=0.6),
            )
            drawn.add(edge.net_id)

    for comp in graph.nodes.values():
        color = _COMP_COLORS.get(comp.comp_type, "#888888")
        ax_layout.add_patch(mpatches.FancyBboxPatch(
            (comp.x + 0.05, comp.y + 0.05),
            comp.footprint.width - 0.10, comp.footprint.height - 0.10,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="white", alpha=0.82, linewidth=1.5, zorder=4,
        ))
        cx = comp.x + comp.footprint.width  / 2.0
        cy = comp.y + comp.footprint.height / 2.0
        ax_layout.text(cx, cy + 0.2, comp.id, color="white", fontsize=7.5,
                        ha="center", va="center", fontweight="bold", zorder=5)
        ax_layout.text(cx, cy - 0.5, comp.name, color="#ccccdd", fontsize=5.5,
                        ha="center", va="center", zorder=5)

    present = {c.comp_type for c in graph.nodes.values()}
    ax_layout.legend(
        handles=[mpatches.Patch(color=_COMP_COLORS.get(t, "#888"), label=t)
                 for t in sorted(present)],
        loc="upper right", fontsize=7,
        facecolor=_PANEL_BG, edgecolor=_GRID_C, labelcolor=_TEXT_C, framealpha=0.85,
    )

    # ── Right: Fitness convergence ─────────────────────────────────────
    ax_conv.set_facecolor(_BG)
    for spine in ax_conv.spines.values():
        spine.set_color(_GRID_C)
    ax_conv.tick_params(colors=_DIM_C, labelsize=8)

    gens = list(range(len(fitness_history)))
    ax_conv.plot(gens, fitness_history, color=_EMERALD, lw=2.0,
                  label="Best fitness", zorder=3)

    ax_conv.axhline(seed_fitness, color="#FF8C00", lw=1.2, linestyle="--",
                     alpha=0.7, label="Phase 1 seed fitness", zorder=2)

    f_range = max(fitness_history) - min(fitness_history)
    offset  = max(f_range * 0.05, 1.0)
    ax_conv.annotate(
        f"Start\n{fitness_history[0]:.1f}",
        xy=(0, fitness_history[0]),
        xytext=(len(gens) * 0.10, fitness_history[0] + offset),
        color=_TEXT_C, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=_DIM_C, lw=0.8),
    )
    ax_conv.annotate(
        f"Final\n{fitness_history[-1]:.1f}",
        xy=(gens[-1], fitness_history[-1]),
        xytext=(gens[-1] * 0.75, fitness_history[-1] + offset),
        color=_TEXT_C, fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=_DIM_C, lw=0.8),
    )

    ax_conv.set_title("GA Convergence — Fitness Over Generations",
                       color=_TEXT_C, fontsize=11, pad=8, fontweight="bold")
    ax_conv.set_xlabel("Generation", color=_DIM_C, fontsize=9)
    ax_conv.set_ylabel("Best Fitness Score (lower = better)", color=_DIM_C, fontsize=9)
    ax_conv.grid(color=_GRID_C, lw=0.5, alpha=0.5, zorder=1)
    ax_conv.legend(fontsize=8, facecolor=_PANEL_BG, edgecolor=_GRID_C,
                    labelcolor=_TEXT_C, framealpha=0.85)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(
    graph: CircuitGraph,
    *,
    pop_size: int      = POP_SIZE,
    n_generations: int = N_GENERATIONS,
) -> CircuitGraph:
    """Phase 2 pipeline entry-point: CircuitGraph → GA-optimized CircuitGraph.

    Steps:
      1. Compute Phase 1 seed fitness and HPWL.
      2. Run Genetic Algorithm.
      3. Compute optimized HPWL.
      4. Render and save the visualizer canvas.

    Args:
        graph:         CircuitGraph from Phase 1 (positions at seed values).
        pop_size:      Override population size (useful in tests).
        n_generations: Override generation count (useful in tests).

    Returns:
        The same CircuitGraph with Component.x/.y updated to GA-optimized
        positions and Pin.abs_x/abs_y refreshed.  Ready for Phase 3 router.
    """
    hpwl_before = half_perimeter_wire_length(graph)

    # Capture Phase 1 seed fitness (before the GA mutates positions)
    evaluator    = FitnessEvaluator(graph)
    seed_chrom   = Chromosome(positions={cid: (c.x, c.y) for cid, c in graph.nodes.items()})
    seed_fitness = evaluator.evaluate(seed_chrom)

    print(f"\n[Phase 2] Step 1/3  Initializing population ({pop_size} chromosomes) ...")
    placer = GeneticPlacer(graph, pop_size=pop_size, n_generations=n_generations)

    print(f"[Phase 2] Step 2/3  Running {n_generations} generations ...")
    _best, fitness_history = placer.run()

    hpwl_after  = half_perimeter_wire_length(graph)
    improvement = (hpwl_before - hpwl_after) / max(hpwl_before, 1e-9) * 100

    print(f"[Phase 2] Step 3/3  Applying best layout to graph ...")
    print(f"   HPWL before GA : {hpwl_before:.2f} mm")
    print(f"   HPWL after GA  : {hpwl_after:.2f} mm")
    print(f"   Improvement    : {improvement:.1f}%")

    out_path = visualize_ga_results(graph, fitness_history, seed_fitness)
    print(f"   Saved -> {out_path}")

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from phase1_eda_engine import NetlistParser, InitialPlacer

    _sample = Path(__file__).parent / "netlists" / "sample_netlist.json"
    with _sample.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    parser  = NetlistParser()
    netlist = parser.parse(raw)

    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    optimized = run_phase2(graph)
    print("\n[Phase 2] Complete. CircuitGraph is ready for Phase 3.")
    sys.exit(0)
