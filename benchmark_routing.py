"""
Phase 8 — Routing benchmark: single-layer (Phase 3) vs multi-layer (Phase 8).

Both routers run on the SAME placement so the comparison isolates the routing
strategy.  Records, per router: routing completion %, total crossings,
same-layer crossings, via count, total trace length, and wall-clock runtime.

Outputs
-------
outputs/benchmark_routing.md   — markdown comparison table
outputs/phase8_benchmark.png   — grouped bar chart (single vs multi)
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_eda_engine import NetlistParser, InitialPlacer, CircuitGraph
from phase2_genetic_placer import run_phase2
from phase3_router import run_phase3
from phase8_multilayer_router import run_phase8, compute_layer_crossings

_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_RESULTS_MD   = _OUTPUT_DIR / "benchmark_routing.md"
_CHART_PNG    = _OUTPUT_DIR / "phase8_benchmark.png"

_BG, _PANEL_BG, _GRID_C, _TEXT_C, _DIM_C = (
    "#0f0f1a", "#16162a", "#1e1e3a", "#e0e0ff", "#888899"
)
_COLORS = {"single": "#FF8C00", "multi": "#00C97A"}


def _silent(fn, *args, **kwargs):
    """Call fn with stdout suppressed (routers are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _completion(p3: dict) -> float:
    """Routing completion % from a metrics dict."""
    routed = p3.get("total_routed", 0)
    failed = p3.get("total_failed", 0)
    total  = routed + failed
    return (routed / total * 100.0) if total else 100.0


def _evaluate(name: str, route_fn, graph: CircuitGraph) -> dict:
    """Run one router on a copy of the placement and gather metrics."""
    g = copy.deepcopy(graph)
    t0 = time.perf_counter()
    _g, traces, p3 = _silent(route_fn, g)
    runtime = time.perf_counter() - t0

    same_layer, _ = compute_layer_crossings(traces) if name == "multi" else (
        p3.get("crossing_count", 0), {}
    )
    return {
        "completion_pct":       _completion(p3),
        "total_crossings":      p3.get("crossing_count", 0),
        "same_layer_crossings": p3.get("same_layer_crossings", same_layer),
        "via_count":            p3.get("via_count", 0),
        "total_length":         p3.get("total_length", 0),
        "runtime_s":            runtime,
    }


def _write_markdown(results: dict, circuit: str = "") -> None:
    """Write the routing benchmark markdown table."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 8 — Routing Benchmark (single-layer vs multi-layer)",
        "",
        f"Both routers run on the **same placement** "
        f"(GA-optimised circuit: `{circuit}`).",
        "",
        "| Router | Completion | Total Crossings | Same-Layer Crossings | "
        "Vias | Trace Length | Runtime (s) |",
        "|--------|-----------:|----------------:|---------------------:|"
        "-----:|-------------:|------------:|",
    ]
    for name in ("single", "multi"):
        r = results[name]
        label = "SINGLE (P3)" if name == "single" else "MULTI (P8)"
        lines.append(
            f"| {label} | {r['completion_pct']:.1f}% | {r['total_crossings']} | "
            f"{r['same_layer_crossings']} | {r['via_count']} | "
            f"{r['total_length']} | {r['runtime_s']:.3f} |"
        )
    lines.append("")
    _RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def _plot_chart(results: dict) -> None:
    """Render a 2x2 grouped bar chart comparing the two routers."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = ["single", "multi"]
    metrics = [
        ("completion_pct",       "Routing Completion % (higher better)"),
        ("same_layer_crossings", "Same-Layer Crossings (lower better)"),
        ("via_count",            "Via Count"),
        ("total_length",         "Total Trace Length (cells)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=_BG)
    fig.suptitle("Phase 8 — Routing Benchmark: single-layer vs multi-layer",
                 color=_TEXT_C, fontsize=14, fontweight="bold")
    labels = {"single": "single", "multi": "multi"}
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.set_facecolor(_PANEL_BG)
        for sp in ax.spines.values():
            sp.set_color(_GRID_C)
        ax.tick_params(colors=_DIM_C, labelsize=9)
        vals = [results[n][key] for n in names]
        bars = ax.bar([labels[n] for n in names], vals,
                      color=[_COLORS[n] for n in names], alpha=0.9)
        ax.set_title(title, color=_TEXT_C, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="y", color=_GRID_C, lw=0.5, alpha=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom",
                    color=_TEXT_C, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_CHART_PNG, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


def _densest_netlist() -> Path:
    """Pick the netlist with the most components (best stress for the router).

    Falls back to the sample netlist.  A dense circuit is where the single-layer
    router actually struggles, so the single-vs-multi contrast is meaningful.
    """
    candidates = [_PROJECT_ROOT / "netlists" / "sample_netlist.json"]
    candidates += sorted((_PROJECT_ROOT / "netlists" / "generated").glob("*.json"))
    best, best_n = candidates[0], -1
    for p in candidates:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            n = len(raw.get("netlist", raw).get("components", []))
        except Exception:
            continue
        if n > best_n:
            best, best_n = p, n
    return best


def run_benchmark() -> dict:
    """Run single vs multi routing on one GA-placed circuit; write md + chart."""
    path = _densest_netlist()
    print(f"[Bench] Circuit: {path.stem}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)
    graph = _silent(run_phase2, graph)        # one shared GA placement

    results = {
        "single": _evaluate("single", run_phase3, graph),
        "multi":  _evaluate("multi",  run_phase8, graph),
    }
    for name in ("single", "multi"):
        r = results[name]
        print(f"[Bench] {name:<6} completion={r['completion_pct']:.0f}% "
              f"crossings={r['total_crossings']} "
              f"same_layer={r['same_layer_crossings']} "
              f"vias={r['via_count']} length={r['total_length']} "
              f"time={r['runtime_s']:.3f}s")

    _write_markdown(results, path.stem)
    _plot_chart(results)
    print(f"[Bench] Wrote {_RESULTS_MD}")
    print(f"[Bench] Wrote {_CHART_PNG}")
    return results


if __name__ == "__main__":
    run_benchmark()
