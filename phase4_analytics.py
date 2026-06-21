"""
Phase 4 — Analytics Engine

Sections
--------
1. Physical constants
2. TraceMetrics dataclass
3. BoardMetrics dataclass
4. AnalyticsEngine  — Observer: computes all EEE metrics from Phase 3 output
5. ReportGenerator  — prints terminal report, saves JSON, renders 4-panel chart
6. run_phase4()     — pipeline entry-point
7. CLI entry-point
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from phase1_eda_engine import CircuitGraph, half_perimeter_wire_length
from phase3_router import RoutedTrace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Physical constants (SI units internally; display converts as needed)
# ---------------------------------------------------------------------------
TRACE_WIDTH_MM      = 0.2       # default trace width (mm)
COPPER_THICKNESS_MM = 0.035     # standard 1 oz copper PCB (mm)
PCB_PERMITTIVITY    = 4.5       # FR4 relative permittivity
VACUUM_PERMITTIVITY = 8.854e-12 # F/m
CELL_SIZE_MM        = 1.0       # 1 grid cell = 1 mm
COPPER_RESISTIVITY  = 1.72e-8   # Ω·m

_SPEED_OF_LIGHT = 3e8           # m/s

# ---------------------------------------------------------------------------
# Colour palette — dark-mode aesthetic
# ---------------------------------------------------------------------------
_BG       = "#0f0f1a"
_PANEL_BG = "#16162a"
_GRID_C   = "#1e1e3a"
_TEXT_C   = "#e0e0ff"
_DIM_C    = "#888899"

_NET_TYPE_COLORS: dict[str, str] = {
    "POWER":  "#FFD700",
    "GROUND": "#888888",
    "SIGNAL": "#00C97A",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — TraceMetrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceMetrics:
    """Electrical properties for one routed trace segment.

    Attributes:
        net_id:                    Net this trace belongs to.
        net_type:                  POWER | SIGNAL | GROUND.
        length_cells:              Path length in grid cells.
        length_mm:                 Physical length in millimetres.
        resistance_ohms:           DC trace resistance.
        parasitic_capacitance_pf:  Parasitic capacitance (picofarads).
        estimated_delay_ps:        Signal propagation delay (picoseconds).
    """

    net_id: str
    net_type: str
    length_cells: int
    length_mm: float
    resistance_ohms: float
    parasitic_capacitance_pf: float
    estimated_delay_ps: float


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — BoardMetrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoardMetrics:
    """Top-level analytics report for a fully routed PCB design.

    Fully JSON-serialisable via dataclasses.asdict().
    This is the Phase 4 → Phase 6 hand-off object.
    """

    design_name: str
    # Placement metrics
    hpwl_mm: float
    component_count: int
    net_count: int
    # Routing metrics
    total_traces_routed: int
    total_traces_failed: int
    total_trace_length_mm: float
    longest_trace_mm: float
    shortest_trace_mm: float
    wire_crossing_count: int
    routing_completion_pct: float
    # Electrical metrics
    total_resistance_ohms: float
    total_capacitance_pf: float
    max_signal_delay_ps: float
    # Per-trace breakdown
    trace_metrics: list[TraceMetrics]
    # EEE rule violations
    violations: list[str]
    # Multi-layer routing (Phase 8 — additive; 0 / single-layer for Phase 3)
    via_count: int = 0
    per_layer_crossings: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — AnalyticsEngine  (Observer pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """Computes EEE metrics by observing Phase 3 routing output.

    Applies real electrical engineering formulas to produce resistance,
    parasitic capacitance, and signal delay for each trace, then runs
    the EEE design-rule checker (DRC) to flag violations.
    """

    def __init__(
        self,
        graph: CircuitGraph,
        traces: list[RoutedTrace],
        phase3_metrics: dict,
    ) -> None:
        """Attach to Phase 3 output.

        Args:
            graph:          CircuitGraph with final component positions.
            traces:         List of RoutedTrace objects from Phase 3.
            phase3_metrics: Metrics dict from run_phase3().
        """
        self._graph   = graph
        self._traces  = traces
        self._p3      = phase3_metrics

        # Pre-compute net membership (net_id -> set of comp_ids)
        self._net_comps: dict[str, set[str]] = {}
        for edge in graph.edges:
            nid = edge.net_id
            if nid not in self._net_comps:
                self._net_comps[nid] = set()
            self._net_comps[nid].add(edge.source[0])
            self._net_comps[nid].add(edge.target[0])

    # ------------------------------------------------------------------
    # Physical calculation methods
    # ------------------------------------------------------------------

    def _calc_trace_length_mm(self, trace: RoutedTrace) -> float:
        """Convert path length from grid cells to millimetres.

        Args:
            trace: RoutedTrace with path length in cells.

        Returns:
            Physical length in mm (cells × CELL_SIZE_MM).
        """
        return trace.length * CELL_SIZE_MM

    def _calc_resistance(self, length_mm: float) -> float:
        """Compute DC trace resistance using R = (rho × L) / A.

        Args:
            length_mm: Trace length in millimetres.

        Returns:
            Resistance in Ohms.
        """
        L = length_mm / 1000.0                            # metres
        A = (TRACE_WIDTH_MM / 1000.0) * (COPPER_THICKNESS_MM / 1000.0)  # m²
        return (COPPER_RESISTIVITY * L) / A

    def _calc_parasitic_capacitance(self, length_mm: float) -> float:
        """Estimate parasitic capacitance via parallel-plate microstrip model.

        Formula: C = (eps_0 × eps_r × W × L) / d

        Args:
            length_mm: Trace length in millimetres.

        Returns:
            Parasitic capacitance in picofarads (pF).
        """
        W = TRACE_WIDTH_MM      / 1000.0   # metres
        L = length_mm           / 1000.0   # metres
        d = COPPER_THICKNESS_MM / 1000.0   # metres (approximate dielectric gap)
        C_farads = (VACUUM_PERMITTIVITY * PCB_PERMITTIVITY * W * L) / d
        return C_farads * 1e12              # convert to pF

    def _calc_signal_delay(self, length_mm: float) -> float:
        """Estimate signal propagation delay through FR4: t = L * sqrt(eps_r) / c.

        Args:
            length_mm: Trace length in millimetres.

        Returns:
            Propagation delay in picoseconds (ps).
        """
        L = length_mm / 1000.0             # metres
        t_seconds = L * math.sqrt(PCB_PERMITTIVITY) / _SPEED_OF_LIGHT
        return t_seconds * 1e12            # convert to ps

    # ------------------------------------------------------------------
    # EEE rule checker
    # ------------------------------------------------------------------

    def _check_eee_violations(self) -> list[str]:
        """Run EEE design-rule checks and return all violation strings.

        Checks:
          1. Wire crossings (DRC fail).
          2. Long signal traces > 20 mm (signal integrity warning).
          3. MCU/IC without decoupling cap within 3 cells.
          4. Unrouted nets (DRC fail).
          5. Power trace potentially too thin (3+ components on short trace).

        Returns:
            List of violation/warning strings (empty = no violations).
        """
        violations: list[str] = []
        graph   = self._graph
        traces  = self._traces
        p3      = self._p3

        # 1. Wire crossings
        n_cross = p3.get("crossing_count", 0)
        if n_cross > 0:
            violations.append(
                f"DRC FAIL: {n_cross} wire crossing(s) detected -- "
                f"reroute affected nets"
            )

        # 2. Long signal traces
        for tm in self._trace_metrics_list():
            if tm.net_type == "SIGNAL" and tm.length_mm > 20.0:
                violations.append(
                    f"SIGNAL WARNING: Net {tm.net_id} trace length "
                    f"{tm.length_mm:.1f} mm exceeds 20 mm -- "
                    f"risk of signal integrity issues"
                )

        # 3. Missing decoupling cap per MCU/IC
        for cid, comp in graph.nodes.items():
            if comp.comp_type not in ("MCU", "IC"):
                continue
            cx = comp.x + comp.footprint.width  / 2.0
            cy = comp.y + comp.footprint.height / 2.0
            has_cap = False
            for oid, other in graph.nodes.items():
                if other.comp_type != "CAPACITOR":
                    continue
                ox = other.x + other.footprint.width  / 2.0
                oy = other.y + other.footprint.height / 2.0
                if math.hypot(cx - ox, cy - oy) <= 3.0:
                    has_cap = True
                    break
            if not has_cap:
                violations.append(
                    f"EEE WARNING: {cid} has no decoupling capacitor "
                    f"within 3 cells"
                )

        # 4. Unrouted nets
        n_fail = p3.get("total_failed", 0)
        if n_fail > 0:
            violations.append(
                f"DRC FAIL: {n_fail} net segment(s) unrouted -- "
                f"board is not complete"
            )

        # 5. Power trace too thin (short trace serving 5+ components)
        power_traces = [t for t in traces if t.net_type == "POWER"]
        for t in power_traces:
            nid = t.net_id
            n_comps = len(self._net_comps.get(nid, set()))
            if t.length < 5 and n_comps >= 5:
                violations.append(
                    f"POWER WARNING: Net {nid} may need wider trace "
                    f"for current capacity"
                )

        return violations

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _trace_metrics_list(self) -> list[TraceMetrics]:
        """Compute TraceMetrics for every RoutedTrace.

        Returns:
            List of TraceMetrics in the same order as self._traces.
        """
        result = []
        for t in self._traces:
            lmm = self._calc_trace_length_mm(t)
            result.append(TraceMetrics(
                net_id=t.net_id,
                net_type=t.net_type,
                length_cells=t.length,
                length_mm=lmm,
                resistance_ohms=self._calc_resistance(lmm),
                parasitic_capacitance_pf=self._calc_parasitic_capacitance(lmm),
                estimated_delay_ps=self._calc_signal_delay(lmm),
            ))
        return result

    # ------------------------------------------------------------------
    # Multi-layer metrics (backward-compatible with single-layer Phase 3)
    # ------------------------------------------------------------------

    def _layer_metrics(self) -> tuple[int, dict[int, int]]:
        """Compute (via_count, per_layer_crossings) from the traces.

        Traces lacking layer info (Phase 3 RoutedTrace) are treated as all on
        layer 0, so single-layer output yields {0: <crossings>} and 0 vias.
        A crossing is an interior cell on a layer occupied by 2+ nets.

        Returns:
            (unique via count, {layer: crossing-cell count}).
        """
        from collections import defaultdict

        cell_nets: dict[tuple, set] = defaultdict(set)
        via_cells: set[tuple[int, int]] = set()
        for t in self._traces:
            layers = getattr(t, "layers", None)
            for i, cell in enumerate(t.path[1:-1], start=1):
                layer = layers[i] if layers is not None else 0
                cell_nets[(cell[0], cell[1], layer)].add(t.net_id)
            for v in getattr(t, "vias", []):
                via_cells.add(tuple(v))

        per_layer: dict[int, int] = defaultdict(int)
        for (_, _, layer), nets in cell_nets.items():
            if len(nets) > 1:
                per_layer[layer] += 1
        # Prefer the router's own counts when present (authoritative).
        via_count = self._p3.get("via_count", len(via_cells))
        if "per_layer_crossings" in self._p3 and self._p3["per_layer_crossings"]:
            per_layer = dict(self._p3["per_layer_crossings"])
        return via_count, dict(per_layer)

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute(self) -> BoardMetrics:
        """Compute and return the complete BoardMetrics report.

        Returns:
            BoardMetrics with all placement, routing, electrical metrics,
            and EEE rule-check results.
        """
        tm_list = self._trace_metrics_list()

        # Aggregate electrical totals
        total_r   = sum(tm.resistance_ohms          for tm in tm_list)
        total_c   = sum(tm.parasitic_capacitance_pf for tm in tm_list)
        max_delay = max((tm.estimated_delay_ps       for tm in tm_list), default=0.0)

        lengths_mm = [tm.length_mm for tm in tm_list]

        total_routed = self._p3.get("total_routed", len(self._traces))
        total_failed = self._p3.get("total_failed", 0)
        total_segs   = total_routed + total_failed
        pct = (total_routed / total_segs * 100.0) if total_segs > 0 else 100.0

        violations = self._check_eee_violations()

        net_ids = {e.net_id for e in self._graph.edges}

        via_count, per_layer_crossings = self._layer_metrics()

        return BoardMetrics(
            design_name=self._graph.metadata.name,
            hpwl_mm=half_perimeter_wire_length(self._graph) * CELL_SIZE_MM,
            component_count=len(self._graph.nodes),
            net_count=len(net_ids),
            total_traces_routed=total_routed,
            total_traces_failed=total_failed,
            total_trace_length_mm=sum(lengths_mm) if lengths_mm else 0.0,
            longest_trace_mm=max(lengths_mm) if lengths_mm else 0.0,
            shortest_trace_mm=min(lengths_mm) if lengths_mm else 0.0,
            wire_crossing_count=self._p3.get("crossing_count", 0),
            routing_completion_pct=round(pct, 1),
            total_resistance_ohms=total_r,
            total_capacitance_pf=total_c,
            max_signal_delay_ps=max_delay,
            trace_metrics=tm_list,
            violations=violations,
            via_count=via_count,
            per_layer_crossings=per_layer_crossings,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — ReportGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates the analytics report in three forms: terminal, JSON, PNG.

    Observer pattern: reads BoardMetrics, never modifies it.
    """

    def __init__(self, metrics: BoardMetrics) -> None:
        """Attach to a completed BoardMetrics object.

        Args:
            metrics: Fully computed BoardMetrics from AnalyticsEngine.compute().
        """
        self._m = metrics

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted ASCII analytics report to stdout."""
        m = self._m
        sep = "=" * 60

        # Find longest / shortest net for labels
        longest  = max(m.trace_metrics, key=lambda t: t.length_mm, default=None)
        shortest = min(m.trace_metrics, key=lambda t: t.length_mm, default=None)
        longest_label  = f"  ({longest.net_id})"  if longest  else ""
        shortest_label = f"  ({shortest.net_id})" if shortest else ""

        print(sep)
        print("  EDA Engine -- Phase 4 Analytics Report")
        print(sep)
        print()
        print(f"  Design        : {m.design_name}")
        print(f"  Components    : {m.component_count}")
        print(f"  Nets          : {m.net_count}")
        print()
        print("-- Placement Metrics " + "-" * 39)
        print(f"  HPWL                  : {m.hpwl_mm:.2f} mm")
        print()
        print("-- Routing Metrics " + "-" * 41)
        print(f"  Traces routed         : {m.total_traces_routed} / "
              f"{m.total_traces_routed + m.total_traces_failed}"
              f"  ({m.routing_completion_pct:.1f}%)")
        print(f"  Total trace length    : {m.total_trace_length_mm:.1f} mm")
        print(f"  Longest trace         : {m.longest_trace_mm:.1f} mm{longest_label}")
        print(f"  Shortest trace        : {m.shortest_trace_mm:.1f} mm{shortest_label}")
        print(f"  Wire crossings        : {m.wire_crossing_count}")
        print()
        print("-- Electrical Metrics " + "-" * 38)
        print(f"  Total resistance      : {m.total_resistance_ohms:.6f} ohm")
        print(f"  Total capacitance     : {m.total_capacitance_pf:.4f} pF")
        print(f"  Max signal delay      : {m.max_signal_delay_ps:.2f} ps")
        print()
        print("-- Per-Trace Breakdown " + "-" * 37)
        header = f"  {'NET':<14} {'LENGTH':>8}  {'R(ohm)':>10}  {'C(pF)':>8}  {'DELAY(ps)':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for tm in m.trace_metrics:
            print(
                f"  {tm.net_id:<14} {tm.length_mm:>6.1f} mm"
                f"  {tm.resistance_ohms:>10.6f}"
                f"  {tm.parasitic_capacitance_pf:>8.4f}"
                f"  {tm.estimated_delay_ps:>10.2f}"
            )
        print()
        print("-- EEE Rule Check " + "-" * 42)
        if not m.violations:
            print("  [OK] No wire crossings")
            print("  [OK] All nets routed")
            print("  [OK] Signal lengths within limit")
            print("  [OK] Decoupling caps in range")
            print("  [OK] Power traces adequate")
        else:
            for v in m.violations:
                print(f"  [!!] {v}")
        print(sep)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def save_json_report(self, path: Path | None = None) -> Path:
        """Serialise BoardMetrics to JSON at the given path.

        Args:
            path: Destination file; defaults to outputs/phase4_report.json.

        Returns:
            Path where the JSON was written.
        """
        if path is None:
            path = _OUTPUT_DIR / "phase4_report.json"
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        report_dict = dataclasses.asdict(self._m)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2)
        return path

    # ------------------------------------------------------------------
    # 4-panel visualisation
    # ------------------------------------------------------------------

    def visualize_analytics(self, output_path: Path | None = None) -> Path:
        """Render a four-panel dark-mode analytics figure and save as PNG.

        Panels:
          1 (top-left)     — Trace length bar chart coloured by net type.
          2 (top-right)    — Normalised R / C / delay grouped bar chart.
          3 (bottom-left)  — Routing completion donut chart.
          4 (bottom-right) — EEE DRC rule check text panel.

        Args:
            output_path: Destination PNG; defaults to outputs/phase4_output.png.

        Returns:
            Path where the PNG was saved.
        """
        if output_path is None:
            output_path = _OUTPUT_DIR / "phase4_output.png"
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        m   = self._m
        tms = m.trace_metrics

        fig, axes = plt.subplots(
            2, 2, figsize=(20, 14), facecolor=_BG,
            gridspec_kw=dict(hspace=0.38, wspace=0.28,
                             left=0.07, right=0.97, top=0.92, bottom=0.07),
        )
        ax1 = axes[0][0]   # trace lengths
        ax2 = axes[0][1]   # electrical properties
        ax3 = axes[1][0]   # donut
        ax4 = axes[1][1]   # DRC text

        # Shared dark styling
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor(_PANEL_BG)
            for spine in ax.spines.values():
                spine.set_color(_GRID_C)
            ax.tick_params(colors=_DIM_C, labelsize=8)

        # ── Panel 1: Trace length bars ─────────────────────────────────
        if tms:
            labels = [f"{t.net_id}\n({t.net_type})" for t in tms]
            lengths = [t.length_mm for t in tms]
            colors  = [_NET_TYPE_COLORS.get(t.net_type, "#aaaaaa") for t in tms]
            x = range(len(tms))
            bars = ax1.bar(x, lengths, color=colors, alpha=0.85, width=0.6, zorder=2)
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(labels, rotation=30, ha="right",
                                 fontsize=7, color=_TEXT_C)
            ax1.set_ylabel("Length (mm)", color=_DIM_C, fontsize=9)
            ax1.grid(axis="y", color=_GRID_C, lw=0.5, alpha=0.5)
            for bar, lmm in zip(bars, lengths):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{lmm:.1f}",
                    ha="center", va="bottom", fontsize=7, color=_TEXT_C,
                )
            legend_patches = [
                mpatches.Patch(color=col, label=nt)
                for nt, col in _NET_TYPE_COLORS.items()
            ]
            ax1.legend(handles=legend_patches, fontsize=7,
                       facecolor=_BG, edgecolor=_GRID_C,
                       labelcolor=_TEXT_C, framealpha=0.85)
        ax1.set_title("Trace Lengths by Net", color=_TEXT_C,
                       fontsize=11, fontweight="bold", pad=8)

        # ── Panel 2: Electrical properties (normalised) ─────────────────
        if tms:
            r_vals = np.array([t.resistance_ohms          for t in tms])
            c_vals = np.array([t.parasitic_capacitance_pf for t in tms])
            d_vals = np.array([t.estimated_delay_ps        for t in tms])

            def _norm(arr: np.ndarray) -> np.ndarray:
                m_ = arr.max()
                return arr / m_ if m_ > 0 else arr

            r_n, c_n, d_n = _norm(r_vals), _norm(c_vals), _norm(d_vals)
            x    = np.arange(len(tms))
            w    = 0.25
            labs = [t.net_id for t in tms]

            ax2.bar(x - w,   r_n, w, label="R (norm)", color="#FF8C00", alpha=0.85, zorder=2)
            ax2.bar(x,       c_n, w, label="C (norm)", color="#00CED1", alpha=0.85, zorder=2)
            ax2.bar(x + w,   d_n, w, label="Delay (norm)", color="#DA70D6", alpha=0.85, zorder=2)
            ax2.set_xticks(list(x))
            ax2.set_xticklabels(labs, rotation=30, ha="right",
                                 fontsize=8, color=_TEXT_C)
            ax2.set_ylabel("Normalised value (0–1)", color=_DIM_C, fontsize=9)
            ax2.grid(axis="y", color=_GRID_C, lw=0.5, alpha=0.5, zorder=1)
            ax2.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID_C,
                       labelcolor=_TEXT_C, framealpha=0.85)
        ax2.set_title("Electrical Properties per Trace",
                       color=_TEXT_C, fontsize=11, fontweight="bold", pad=8)

        # ── Panel 3: Routing completion donut ──────────────────────────
        routed = m.total_traces_routed
        failed = m.total_traces_failed
        total  = routed + failed
        if total == 0:
            routed, failed, total = 1, 0, 1
        ax3.pie(
            [routed, max(failed, 0)],
            colors=["#00C97A", "#FF4444"],
            wedgeprops=dict(width=0.45, edgecolor=_BG, linewidth=2),
            startangle=90,
        )
        pct_str = f"{m.routing_completion_pct:.1f}%"
        ax3.text(0, 0, pct_str, ha="center", va="center",
                 fontsize=20, fontweight="bold", color=_TEXT_C)
        ax3.text(0, -0.55, f"{routed}/{total} segments", ha="center",
                 fontsize=10, color=_DIM_C)
        ax3.set_title("Routing Completion", color=_TEXT_C,
                       fontsize=11, fontweight="bold", pad=8)
        ax3.legend(
            handles=[
                mpatches.Patch(color="#00C97A", label="Routed"),
                mpatches.Patch(color="#FF4444", label="Failed"),
            ],
            fontsize=8, facecolor=_BG, edgecolor=_GRID_C,
            labelcolor=_TEXT_C, framealpha=0.85, loc="lower center",
        )

        # ── Panel 4: EEE DRC report ────────────────────────────────────
        ax4.axis("off")
        ax4.set_title("EEE DRC Report", color=_TEXT_C,
                       fontsize=11, fontweight="bold", pad=8)

        # Passing checks (only the ones we actually verified)
        passing: list[str] = []
        if m.wire_crossing_count == 0:
            passing.append("No wire crossings")
        if m.total_traces_failed == 0:
            passing.append("All net segments routed")
        if all(t.length_mm <= 20.0 for t in tms if t.net_type == "SIGNAL"):
            passing.append("Signal lengths within 20 mm limit")

        lines: list[tuple[str, str]] = []  # (text, color)
        for v in m.violations:
            lines.append((f"[!!] {v}", "#FF6B6B"))
        for p in passing:
            lines.append((f"[OK] {p}", "#69FF69"))

        if not lines:
            lines = [("[OK] No violations detected", "#69FF69")]

        y = 0.95
        for text, color in lines:
            ax4.text(0.05, y, text, transform=ax4.transAxes,
                     color=color, fontsize=8.5, va="top",
                     fontfamily="monospace", linespacing=1.5)
            y -= 0.10

        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase4(
    graph: CircuitGraph,
    traces: list[RoutedTrace],
    phase3_metrics: dict,
) -> BoardMetrics:
    """Phase 4 pipeline: routed graph + traces -> BoardMetrics analytics report.

    Steps:
      1. Compute per-trace electrical properties.
      2. Run EEE rule checker.
      3. Save JSON report and PNG visualisation.

    Args:
        graph:          CircuitGraph with GA-optimised, routed component positions.
        traces:         List of RoutedTrace objects from run_phase3().
        phase3_metrics: Metrics dict from run_phase3().

    Returns:
        Complete BoardMetrics object (the Phase 4 -> Phase 6 hand-off value).
    """
    print("\n[Phase 4] Step 1/3  Computing trace electrical properties ...")
    engine  = AnalyticsEngine(graph, traces, phase3_metrics)
    metrics = engine.compute()
    print(f"   {len(metrics.trace_metrics)} traces analysed")

    print("\n[Phase 4] Step 2/3  Running EEE rule checker ...")
    n_v = len(metrics.violations)
    if n_v == 0:
        print("   0 violations found -- design passes all EEE checks")
    else:
        for v in metrics.violations:
            print(f"   [!!] {v}")

    print("\n[Phase 4] Step 3/3  Generating report ...")
    reporter = ReportGenerator(metrics)

    json_path = reporter.save_json_report()
    print(f"   Saved -> {json_path}")

    png_path = reporter.visualize_analytics()
    print(f"   Saved -> {png_path}")

    reporter.print_report()

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phase1_eda_engine import NetlistParser, InitialPlacer
    from phase2_genetic_placer import run_phase2
    from phase3_router import run_phase3

    _sample = Path(__file__).parent / "netlists" / "sample_netlist.json"
    import json as _json
    with _sample.open(encoding="utf-8") as fh:
        raw = _json.load(fh)

    parser  = NetlistParser()
    netlist = parser.parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    graph = run_phase2(graph)
    graph, traces, phase3_metrics = run_phase3(graph)
    metrics = run_phase4(graph, traces, phase3_metrics)

    print("\n[Phase 4] Complete. BoardMetrics ready for Phase 6.")
    sys.exit(0)
