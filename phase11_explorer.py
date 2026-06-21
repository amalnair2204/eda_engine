"""
Phase 11 — Design-Space Exploration (multi-objective Pareto)

An ORCHESTRATION layer over the existing phases.  Instead of returning a single
layout, Phase 11 sweeps a configurable grid of place + route strategies for the
SAME netlist, scores every candidate on multiple objectives, computes the
Pareto-optimal (non-dominated) set, and recommends a sensible default trade-off.

It does NOT modify Phases 0-10.  Each candidate runs the real pipeline on its
own deep copy of the CircuitGraph (so run_phase3's grid-expansion mutation can
never leak back to the caller), then reads metrics from the Phase 4 analytics
engine read-only.

Sections
--------
1. Config + objective definitions
2. Candidate generation (place + route + Phase 4 metrics + runtime)
3. Non-dominated sorting (Pareto front)
4. Recommendation
5. Visualization (scatter PNG) + results table (markdown)
6. run_phase11() pipeline entry-point
7. CLI entry-point
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_eda_engine import CircuitGraph

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_PARETO_PNG   = _OUTPUT_DIR / "phase11_pareto.png"
_RESULTS_MD   = _OUTPUT_DIR / "phase11_results.md"

# ---------------------------------------------------------------------------
# Colour palette — mirrors the project dark-mode aesthetic
# ---------------------------------------------------------------------------
_BG, _PANEL_BG, _GRID_C, _TEXT_C, _DIM_C = (
    "#0f0f1a", "#16162a", "#1e1e3a", "#e0e0ff", "#888899"
)
# One colour per (placer+router) strategy combination.
_STRATEGY_COLORS = {
    "ga+single":  "#1E90FF",
    "ga+multi":   "#00C97A",
    "rl+single":  "#FF8C00",
    "rl+multi":   "#DA70D6",
}
_PARETO_RING = "#FFD700"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Config + objective definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Objective:
    """One scalar objective to minimise.

    Attributes:
        name:  Human label (also the candidate["objectives"] key).
        lower_is_better: Always True here (every objective is minimised); kept
                         explicit so the set is trivially extensible to maximised
                         objectives later by flipping this flag.
    """

    name: str
    lower_is_better: bool = True


# The objective set is explicit and easy to extend: append an Objective and make
# sure _candidate_objectives() fills the matching key.
OBJECTIVES: list[Objective] = [
    Objective("hpwl"),
    Objective("crossings"),
    Objective("trace_length"),
    Objective("runtime_s"),
]

# Below this routing completion a candidate is invalid (treated as dominated).
_COMPLETE_PCT = 100.0


def _default_config() -> dict:
    """Return the default exploration config (modest grid; override per call)."""
    return {
        "placers":        ["ga", "rl"],        # rl auto-skips if no trained model
        "routers":        ["single", "multi"],
        "ga_generations": [40],                # one candidate per value (ga only)
        "ga_pop":         30,
        "rl_model_path":  None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Candidate generation
# ═══════════════════════════════════════════════════════════════════════════════

def _strategy_key(placer: str, router: str) -> str:
    """Combined strategy label used for colouring + display."""
    return f"{placer}+{router}"


def _candidate_objectives(board, runtime_s: float) -> dict:
    """Extract the minimised-objective values from a Phase 4 BoardMetrics.

    Args:
        board:     BoardMetrics from AnalyticsEngine.compute().
        runtime_s: Wall-clock seconds for this candidate's place + route.

    Returns:
        Dict keyed by every Objective.name (all values "lower is better").
    """
    return {
        "hpwl":         float(board.hpwl_mm),
        "crossings":    int(board.wire_crossing_count),
        "trace_length": float(board.total_trace_length_mm),
        "runtime_s":    float(runtime_s),
    }


def _run_one_candidate(
    base_graph: CircuitGraph,
    placer: str,
    router: str,
    ga_generations: int | None,
    cfg: dict,
) -> dict | None:
    """Run the full place + route pipeline once and collect metrics + runtime.

    The candidate gets its OWN deep copy of the graph so run_phase3's grid
    expansion (which mutates metadata) cannot affect the caller or sibling
    candidates.

    Args:
        base_graph:     Seed CircuitGraph (read-only; deep-copied here).
        placer:         "ga" or "rl".
        router:         "single" or "multi".
        ga_generations: Generation count for GA candidates (None for RL).
        cfg:            Exploration config.

    Returns:
        A candidate dict, or None if the run failed (e.g. RL model missing).
    """
    from phase4_analytics import AnalyticsEngine

    graph = copy.deepcopy(base_graph)
    label = _strategy_key(placer, router)
    if ga_generations is not None:
        label = f"{label}+g{ga_generations}"

    t0 = time.perf_counter()
    try:
        # ── Placement strategy ──────────────────────────────────────────
        if placer == "ga":
            from phase2_genetic_placer import run_phase2
            graph = run_phase2(
                graph,
                pop_size=cfg.get("ga_pop", 30),
                n_generations=ga_generations if ga_generations is not None else 40,
            )
        elif placer == "rl":
            from phase7_rl_placer import RLPlacer
            model_path = cfg.get("rl_model_path")
            graph = RLPlacer(model_path).place(graph)
        else:
            raise ValueError(f"Unknown placer '{placer}'")

        # ── Routing strategy ────────────────────────────────────────────
        if router == "single":
            from phase3_router import run_phase3
            graph, traces, rmetrics = run_phase3(graph)
        elif router == "multi":
            from phase8_multilayer_router import run_phase8
            graph, traces, rmetrics = run_phase8(graph)
        else:
            raise ValueError(f"Unknown router '{router}'")

        # ── Phase 4 metrics (read-only; no file writes) ─────────────────
        board = AnalyticsEngine(graph, traces, rmetrics).compute()
    except Exception as exc:   # noqa: BLE001 — explorer must survive a bad combo
        print(f"[Phase 11] Skipping candidate '{label}': {exc}")
        return None
    runtime_s = time.perf_counter() - t0

    objectives  = _candidate_objectives(board, runtime_s)
    completion  = float(board.routing_completion_pct)
    return {
        "id":             label,
        "placer":         placer,
        "router":         router,
        "ga_generations": ga_generations,
        "strategy":       _strategy_key(placer, router),
        "completion":     completion,
        "runtime_s":      round(runtime_s, 3),
        "metrics": {
            "hpwl_mm":         round(float(board.hpwl_mm), 2),
            "crossings":       int(board.wire_crossing_count),
            "trace_length_mm": round(float(board.total_trace_length_mm), 1),
            "via_count":       int(getattr(board, "via_count", 0)),
            "completion_pct":  completion,
        },
        "objectives":     objectives,
        "pareto":         False,   # set later by compute_pareto()
    }


def generate_candidates(base_graph: CircuitGraph, config: dict | None = None) -> list[dict]:
    """Sweep the configured place/route grid and return all candidate dicts.

    Args:
        base_graph: Seed CircuitGraph (deep-copied per candidate).
        config:     Optional override of _default_config().

    Returns:
        List of candidate dicts (failed combos are dropped).
    """
    cfg = {**_default_config(), **(config or {})}
    ga_gens = list(cfg.get("ga_generations") or [40])

    candidates: list[dict] = []
    for placer in cfg["placers"]:
        for router in cfg["routers"]:
            # GA explores its generation settings; RL has a single setting.
            settings = ga_gens if placer == "ga" else [None]
            for gen in settings:
                cand = _run_one_candidate(base_graph, placer, router, gen, cfg)
                if cand is not None:
                    candidates.append(cand)
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Non-dominated sorting (Pareto front)
# ═══════════════════════════════════════════════════════════════════════════════

def dominates(a: tuple, b: tuple) -> bool:
    """Return True if point *a* dominates point *b* (all objectives minimised).

    a dominates b iff a is <= b in every objective and < b in at least one.

    Args:
        a, b: Equal-length objective tuples (lower is better).

    Returns:
        True when a Pareto-dominates b.
    """
    no_worse = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return no_worse and strictly_better


def pareto_front_indices(points: list[tuple]) -> list[int]:
    """Indices of the non-dominated (Pareto-optimal) points (all minimised).

    A point is on the front if no other point dominates it.

    Args:
        points: List of equal-length objective tuples.

    Returns:
        Sorted list of indices that are Pareto-optimal.
    """
    n = len(points)
    front: list[int] = []
    for i in range(n):
        if not any(j != i and dominates(points[j], points[i]) for j in range(n)):
            front.append(i)
    return front


def objective_vector(candidate: dict) -> tuple:
    """Objective tuple for a candidate, ordered by OBJECTIVES (all minimised)."""
    return tuple(candidate["objectives"][o.name] for o in OBJECTIVES)


def compute_pareto(candidates: list[dict]) -> list[dict]:
    """Mark each candidate's "pareto" flag and return the Pareto set.

    Candidates with completion < 100% are invalid and can never be on the
    front (they are treated as dominated regardless of their objective values).

    Args:
        candidates: Candidate dicts from generate_candidates() (mutated in place
                    — each gets its "pareto" flag set).

    Returns:
        List of candidates on the Pareto front (a subset of valid candidates).
    """
    for c in candidates:
        c["pareto"] = False

    valid = [c for c in candidates if c["completion"] >= _COMPLETE_PCT]
    if not valid:
        return []

    points = [objective_vector(c) for c in valid]
    for idx in pareto_front_indices(points):
        valid[idx]["pareto"] = True
    return [c for c in valid if c["pareto"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Recommendation
# ═══════════════════════════════════════════════════════════════════════════════

def recommend(pareto_set: list[dict]) -> dict | None:
    """Pick a sensible default from the Pareto set.

    Heuristic: among fully-routed (completion == 100%) Pareto candidates, choose
    the one with the fewest crossings, breaking ties by lowest HPWL, then by
    lowest runtime.  Returns a copy with a one-line "rationale".

    Args:
        pareto_set: Candidates on the Pareto front.

    Returns:
        The recommended candidate dict (with "rationale"), or None if empty.
    """
    complete = [c for c in pareto_set if c["completion"] >= _COMPLETE_PCT]
    pool = complete or pareto_set
    if not pool:
        return None

    best = min(pool, key=lambda c: (
        c["objectives"]["crossings"],
        c["objectives"]["hpwl"],
        c["objectives"]["runtime_s"],
    ))
    rec = dict(best)
    rec["rationale"] = (
        f"Pareto-optimal {best['strategy']} layout — fully routed "
        f"({best['completion']:.0f}%) with the fewest crossings "
        f"({best['objectives']['crossings']}) then lowest HPWL "
        f"({best['objectives']['hpwl']:.1f} mm)."
    )
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Visualization + results table
# ═══════════════════════════════════════════════════════════════════════════════

def render_pareto_scatter(
    candidates: list[dict], output_path: Path | None = None
) -> Path:
    """Render a 2D Pareto scatter (HPWL vs crossings) and save as PNG.

    Points are coloured by strategy; Pareto-optimal candidates are ringed in
    gold and connected (sorted by HPWL) to trace the front.

    Args:
        candidates:  All candidate dicts (with "pareto" flags set).
        output_path: Destination PNG (defaults to outputs/phase11_pareto.png).

    Returns:
        Path where the PNG was saved.
    """
    if output_path is None:
        output_path = _PARETO_PNG
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=_BG)
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(_GRID_C)
    ax.tick_params(colors=_DIM_C, labelsize=8)
    ax.grid(color=_GRID_C, lw=0.5, alpha=0.5, zorder=0)
    ax.set_title("Phase 11 — Design-Space Pareto Front (HPWL vs Crossings)",
                 color=_TEXT_C, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("HPWL (mm) — lower better", color=_DIM_C, fontsize=9)
    ax.set_ylabel("Wire crossings — lower better", color=_DIM_C, fontsize=9)

    seen_strategies: set[str] = set()
    for c in candidates:
        col = _STRATEGY_COLORS.get(c["strategy"], "#aaaaaa")
        x, y = c["objectives"]["hpwl"], c["objectives"]["crossings"]
        invalid = c["completion"] < _COMPLETE_PCT
        label = c["strategy"] if c["strategy"] not in seen_strategies else None
        seen_strategies.add(c["strategy"])
        ax.scatter(
            x, y, s=130, color=col, alpha=0.45 if invalid else 0.9,
            marker="x" if invalid else "o",
            edgecolors="white", linewidths=0.6, zorder=3, label=label,
        )
        ax.annotate(c["id"], (x, y), color=_DIM_C, fontsize=6,
                    xytext=(4, 4), textcoords="offset points", zorder=4)

    # Highlight + connect the Pareto front (sorted by HPWL).
    front = sorted([c for c in candidates if c["pareto"]],
                   key=lambda c: c["objectives"]["hpwl"])
    if front:
        fx = [c["objectives"]["hpwl"] for c in front]
        fy = [c["objectives"]["crossings"] for c in front]
        ax.plot(fx, fy, color=_PARETO_RING, lw=1.4, alpha=0.7,
                linestyle="--", zorder=2)
        ax.scatter(fx, fy, s=260, facecolors="none", edgecolors=_PARETO_RING,
                   linewidths=2.0, zorder=5, label="Pareto-optimal")

    ax.legend(fontsize=8, facecolor=_PANEL_BG, edgecolor=_GRID_C,
              labelcolor=_TEXT_C, framealpha=0.9, loc="best")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return output_path


def write_results_table(
    candidates: list[dict],
    recommendation: dict | None,
    output_path: Path | None = None,
) -> Path:
    """Write a markdown results table of every candidate to outputs/.

    Args:
        candidates:     All candidate dicts (with "pareto" flags set).
        recommendation: The recommended candidate (or None).
        output_path:    Destination markdown (defaults to phase11_results.md).

    Returns:
        Path where the markdown was saved.
    """
    if output_path is None:
        output_path = _RESULTS_MD
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rec_id = recommendation["id"] if recommendation else None
    lines: list[str] = [
        "# Phase 11 — Design-Space Exploration Results",
        "",
        f"Candidates evaluated: **{len(candidates)}**  |  "
        f"Objectives minimised: {', '.join(o.name for o in OBJECTIVES)}",
        "",
        "| Candidate | Placer | Router | HPWL (mm) | Crossings | "
        "Trace len (mm) | Vias | Completion % | Runtime (s) | Pareto | Recommended |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in candidates:
        m = c["metrics"]
        lines.append(
            f"| {c['id']} | {c['placer']} | {c['router']} | "
            f"{m['hpwl_mm']:.2f} | {m['crossings']} | {m['trace_length_mm']:.1f} | "
            f"{m['via_count']} | {m['completion_pct']:.1f} | {c['runtime_s']:.2f} | "
            f"{'✅' if c['pareto'] else '—'} | "
            f"{'⭐' if c['id'] == rec_id else '—'} |"
        )

    lines += ["", "## Recommendation", ""]
    if recommendation:
        lines.append(f"**{recommendation['id']}** — {recommendation['rationale']}")
    else:
        lines.append("_No fully-routed candidate found; no recommendation._")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase11(circuit_graph: CircuitGraph, config: dict | None = None) -> dict:
    """Phase 11 pipeline: explore the design space and recommend a trade-off.

    Sweeps a grid of place/route strategies for the SAME netlist, scores each on
    multiple objectives, computes the Pareto-optimal set, renders a scatter +
    results table, and recommends a default.  Phases 0-10 are untouched; the
    input graph is never mutated (every candidate uses its own deep copy).

    Args:
        circuit_graph: Seed CircuitGraph (from Phase 1, positions at seed values
                       or already placed — re-placement starts fresh either way).
        config:        Optional dict overriding _default_config():
                       placers, routers, ga_generations, ga_pop, rl_model_path.

    Returns:
        Dict:
        {
          "candidates":     [candidate dicts],
          "pareto":         [candidates on the front],
          "recommendation": candidate dict with "rationale" (or None),
          "objectives":     [objective names],
          "pareto_png":     Path,
          "results_md":     Path,
        }
    """
    print("\n[Phase 11] Exploring design space ...")
    candidates = generate_candidates(circuit_graph, config)
    if not candidates:
        raise RuntimeError(
            "Phase 11 produced no candidates — check placer/router config "
            "(e.g. RL model availability)."
        )

    pareto_set = compute_pareto(candidates)
    recommendation = recommend(pareto_set)

    png_path = render_pareto_scatter(candidates)
    md_path  = write_results_table(candidates, recommendation)

    print(f"[Phase 11] {len(candidates)} candidates, "
          f"{len(pareto_set)} Pareto-optimal.")
    if recommendation:
        print(f"[Phase 11] Recommended: {recommendation['id']} — "
              f"{recommendation['rationale']}")
    print(f"[Phase 11] Saved -> {png_path}")
    print(f"[Phase 11] Saved -> {md_path}")

    return {
        "candidates":     candidates,
        "pareto":         pareto_set,
        "recommendation": recommendation,
        "objectives":     [o.name for o in OBJECTIVES],
        "pareto_png":     png_path,
        "results_md":     md_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys

    from phase1_eda_engine import NetlistParser, InitialPlacer

    _sample = _PROJECT_ROOT / "netlists" / "sample_netlist.json"
    with _sample.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    parser  = NetlistParser()
    netlist = parser.parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    result = run_phase11(graph)
    print("\n[Phase 11] Complete. Pareto results in outputs/phase11_results.md")
    sys.exit(0)
