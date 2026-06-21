"""
Phase 7 — Placement benchmark: random vs GA vs RL.

Runs all three placement strategies on the held-out circuit suite and, for each,
records final HPWL, overlap count, wall-clock runtime, and — by feeding the
placed graph through Phase 3 (router) and Phase 4 (analytics) — routing
completion percentage and wire-crossing count.

Outputs
-------
outputs/benchmark_results.md   — markdown comparison table
outputs/phase7_benchmark.png   — grouped bar chart (random vs GA vs RL)

A result in which the GA still wins is a valid, shippable outcome — the honest
head-to-head comparison is the deliverable, not a guaranteed RL win.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_eda_engine import CircuitGraph, half_perimeter_wire_length
from phase2_genetic_placer import (
    GeneticPlacer,
    run_phase2,
    _update_graph_pin_positions,
)
from phase7_rl_placer import RLPlacer
from train_phase7_rl import build_suite

_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_RESULTS_MD   = _OUTPUT_DIR / "benchmark_results.md"
_CHART_PNG    = _OUTPUT_DIR / "phase7_benchmark.png"

_BG, _PANEL_BG, _GRID_C, _TEXT_C, _DIM_C = (
    "#0f0f1a", "#16162a", "#1e1e3a", "#e0e0ff", "#888899"
)
_PLACER_COLORS = {"random": "#888899", "ga": "#FF8C00", "rl": "#00C97A"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent(fn, *args, **kwargs):
    """Call fn with stdout suppressed (phase functions are very chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _count_overlaps(graph: CircuitGraph) -> int:
    """Number of overlapping component pairs in the current placement."""
    comps = list(graph.nodes.values())
    n = 0
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            if not (
                a.x + a.footprint.width <= b.x
                or b.x + b.footprint.width <= a.x
                or a.y + a.footprint.height <= b.y
                or b.y + b.footprint.height <= a.y
            ):
                n += 1
    return n


def random_placer(graph: CircuitGraph) -> CircuitGraph:
    """Baseline: uniform random, overlap-free placement (Placer interface).

    Reuses the GA's overlap-avoiding random chromosome generator so the
    baseline is a fair, valid random layout rather than a degenerate one.
    """
    placer = GeneticPlacer(graph, pop_size=2, n_generations=1, seed=123)
    chrom  = placer._random_chromosome()
    for cid, comp in graph.nodes.items():
        comp.x, comp.y = chrom.positions[cid]
    _update_graph_pin_positions(graph)
    return graph


def _ga_placer(graph: CircuitGraph) -> CircuitGraph:
    """GA strategy with the project-default population / generations."""
    return run_phase2(graph)


# ---------------------------------------------------------------------------
# Single (placer, circuit) measurement
# ---------------------------------------------------------------------------

def _evaluate(placer_name: str, placer_fn, graph: CircuitGraph) -> dict:
    """Run one placer on one circuit and gather all benchmark metrics.

    Args:
        placer_name: "random" | "ga" | "rl".
        placer_fn:   Callable(CircuitGraph) -> CircuitGraph.
        graph:       Fresh CircuitGraph (a deep copy is made internally).

    Returns:
        Dict with hpwl, overlaps, runtime_s, completion_pct, crossings.
    """
    from phase3_router import run_phase3
    from phase4_analytics import run_phase4

    g = copy.deepcopy(graph)

    t0 = time.perf_counter()
    g  = _silent(placer_fn, g)
    runtime = time.perf_counter() - t0

    hpwl     = half_perimeter_wire_length(g)
    overlaps = _count_overlaps(g)

    # Route + analyse to get completion % and crossings.
    g2, traces, p3 = _silent(run_phase3, g)
    board = _silent(run_phase4, g2, traces, p3)

    return {
        "hpwl":           hpwl,
        "overlaps":       overlaps,
        "runtime_s":      runtime,
        "completion_pct": board.routing_completion_pct,
        "crossings":      board.wire_crossing_count,
    }


# ---------------------------------------------------------------------------
# Markdown + chart
# ---------------------------------------------------------------------------

def _write_markdown(agg: dict, per_circuit: list[dict], hpwl_vs_random: float) -> None:
    """Write the benchmark results markdown table."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 7 — Placement Benchmark (random vs GA vs RL)")
    lines.append("")
    lines.append("Averaged over the held-out evaluation circuits "
                 "(never seen during RL training).")
    lines.append("")
    lines.append("| Placer | Avg HPWL | Avg Overlaps | Avg Runtime (s) | "
                 "Avg Routing Completion | Avg Crossings |")
    lines.append("|--------|---------:|-------------:|----------------:|"
                 "-----------------------:|--------------:|")
    for name in ("random", "ga", "rl"):
        a = agg[name]
        lines.append(
            f"| {name.upper():<6} | {a['hpwl']:.2f} | {a['overlaps']:.2f} | "
            f"{a['runtime_s']:.3f} | {a['completion_pct']:.1f}% | "
            f"{a['crossings']:.2f} |"
        )
    lines.append("")
    lines.append(f"**RL HPWL reduction vs random:** {hpwl_vs_random:.1f}%  "
                 f"(acceptance threshold: ≥ 20%)")
    lines.append("")
    lines.append("## Per-circuit detail")
    lines.append("")
    lines.append("| Circuit | Placer | HPWL | Overlaps | Runtime (s) | "
                 "Completion | Crossings |")
    lines.append("|---------|--------|-----:|---------:|------------:|"
                 "-----------:|----------:|")
    for row in per_circuit:
        lines.append(
            f"| {row['circuit']} | {row['placer'].upper()} | {row['hpwl']:.2f} | "
            f"{row['overlaps']} | {row['runtime_s']:.3f} | "
            f"{row['completion_pct']:.1f}% | {row['crossings']} |"
        )
    lines.append("")
    _RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def _plot_chart(agg: dict) -> None:
    """Render a 2x2 grouped bar chart comparing the three placers."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names   = ["random", "ga", "rl"]
    metrics = [
        ("hpwl",           "Average HPWL (lower better)"),
        ("completion_pct", "Routing Completion % (higher better)"),
        ("crossings",      "Wire Crossings (lower better)"),
        ("runtime_s",      "Runtime seconds (lower better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=_BG)
    fig.suptitle("Phase 7 — Placement Benchmark: random vs GA vs RL",
                 color=_TEXT_C, fontsize=15, fontweight="bold")
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.set_facecolor(_PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID_C)
        ax.tick_params(colors=_DIM_C, labelsize=9)
        vals = [agg[n][key] for n in names]
        bars = ax.bar(names, vals,
                      color=[_PLACER_COLORS[n] for n in names], alpha=0.9)
        ax.set_title(title, color=_TEXT_C, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="y", color=_GRID_C, lw=0.5, alpha=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom",
                    color=_TEXT_C, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(_CHART_PNG, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    """Run the full benchmark and write markdown + chart.

    Returns:
        The aggregate metrics dict keyed by placer name.
    """
    _, eval_graphs, _, eval_names = build_suite()
    rl = RLPlacer()                       # raises if the policy is missing

    placers = {"random": random_placer, "ga": _ga_placer, "rl": rl}

    per_circuit: list[dict] = []
    acc: dict[str, list[dict]] = {n: [] for n in placers}

    for gname, graph in zip(eval_names, eval_graphs):
        print(f"[Bench] Circuit: {gname} "
              f"({len(graph.nodes)} comps, {len(graph.edges)} edges)")
        for pname, pfn in placers.items():
            res = _evaluate(pname, pfn, graph)
            acc[pname].append(res)
            per_circuit.append({"circuit": gname, "placer": pname, **res})
            print(f"   {pname:<6}  HPWL={res['hpwl']:7.2f}  "
                  f"overlaps={res['overlaps']}  "
                  f"completion={res['completion_pct']:.0f}%  "
                  f"crossings={res['crossings']}  "
                  f"time={res['runtime_s']:.3f}s")

    agg = {
        n: {k: float(np.mean([r[k] for r in acc[n]]))
            for k in ("hpwl", "overlaps", "runtime_s", "completion_pct", "crossings")}
        for n in placers
    }

    rand_h, rl_h = agg["random"]["hpwl"], agg["rl"]["hpwl"]
    hpwl_vs_random = (rand_h - rl_h) / rand_h * 100 if rand_h > 0 else 0.0

    _write_markdown(agg, per_circuit, hpwl_vs_random)
    _plot_chart(agg)

    print(f"\n[Bench] RL HPWL reduction vs random: {hpwl_vs_random:.1f}%")
    print(f"[Bench] Wrote {_RESULTS_MD}")
    print(f"[Bench] Wrote {_CHART_PNG}")
    return agg


if __name__ == "__main__":
    run_benchmark()
