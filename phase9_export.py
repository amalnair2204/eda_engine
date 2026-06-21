"""
Phase 9 — Manufacturing Export (Gerber / Drill / BOM / KiCad)

Terminal stage (NOT a swappable strategy): converts a placed + routed board
into industry-standard fabrication files.  Consumes the routed output of EITHER
the single-layer Phase 3 router (RoutedTrace, no layer info -> treated as layer
0) OR the multi-layer Phase 8 router (LayeredTrace, per-cell layers + vias).

Produces, under outputs/manufacturing/:
  - <board>-F_Cu.gbr, <board>-B_Cu.gbr, ...   one Gerber per copper layer
  - <board>-Edge_Cuts.gbr                     board outline
  - <board>.drl                               Excellon drill (vias + TH pads)
  - bom.csv                                   grouped bill of materials
  - <board>.net                               KiCad S-expression netlist
  - <board>_gerbers.zip                       fab-ready bundle (Gerbers + drill)
and outputs/phase9_preview.png                copper + outline preview.

Gerbers and the drill file are generated with gerbonara (RS-274X / Excellon are
never hand-rolled).  Units are millimetres; one grid cell = GRID_PITCH_MM mm.
The grid origin is bottom-left (y=0 at the bottom), matching Gerber, so no Y
flip is required.

Sections
--------
1. Config + coordinate transform
2. Geometry extraction (segments per layer, pads, vias)
3. Gerber + Excellon writers (gerbonara)
4. BOM CSV + refdes derivation
5. KiCad netlist
6. Preview render + zip bundle
7. run_phase9() pipeline entry-point
8. CLI entry-point
"""

from __future__ import annotations

import csv
import os
import re
import zipfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gerbonara as gn
from gerbonara import graphic_objects as go, apertures as ap
from gerbonara import MM

from dotenv import load_dotenv

from phase1_eda_engine import CircuitGraph, Component

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_MFG_DIR      = _OUTPUT_DIR / "manufacturing"

# ---------------------------------------------------------------------------
# Copper layer naming (KiCad convention)
# ---------------------------------------------------------------------------
def _layer_name(idx: int, n_layers: int) -> str:
    """Map a layer index to a KiCad copper layer name."""
    if idx == 0:
        return "F_Cu"
    if idx == n_layers - 1:
        return "B_Cu"
    return f"In{idx}_Cu"


_REFDES_PREFIX = {
    "RESISTOR": "R", "CAPACITOR": "C", "IC": "U", "MCU": "U",
    "LED": "D", "DIODE": "D", "CONNECTOR": "J", "POWER": "J",
}
# A real reference designator is a short 1-2 letter prefix + number (R1, C10,
# U3).  Deliberately excludes part names like "ESP32" (3-letter prefix).
_REFDES_RE = re.compile(r"^[A-Z]{1,2}\d+$")

# Preview palette (mirrors the project aesthetic)
_BG, _GRID_C, _TEXT_C, _DIM_C = "#0f0f1a", "#1e1e3a", "#e0e0ff", "#888899"
_LAYER_COLORS = ["#FF4444", "#1E90FF", "#00C97A", "#FFD700", "#DA70D6"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Config + coordinate transform
# ═══════════════════════════════════════════════════════════════════════════════

def _f(key: str, default: float) -> float:
    """Read a float physical parameter from the environment."""
    return float(os.getenv(key, str(default)))


class ExportConfig:
    """Physical fabrication parameters (read from .env)."""

    def __init__(self) -> None:
        self.pitch     = _f("GRID_PITCH_MM", 2.54)
        self.trace_w   = _f("TRACE_WIDTH_MM", 0.4)
        self.pad_dia   = _f("PAD_DIAMETER_MM", 1.6)
        self.via_drill = _f("VIA_DRILL_MM", 0.6)
        self.pad_drill = _f("PAD_DRILL_MM", 0.8)
        self.margin_mm = self.pitch  # board-outline margin = one cell

    def mm(self, coord: float) -> float:
        """Convert a grid coordinate to millimetres."""
        return coord * self.pitch


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Geometry extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _trace_layers(trace) -> list[int]:
    """Per-cell layer list for a trace (single-layer traces -> all layer 0)."""
    layers = getattr(trace, "layers", None)
    if layers is not None:
        return list(layers)
    return [0] * len(trace.path)


def _collect_layers(routed_paths: list) -> list[int]:
    """Sorted list of distinct copper layers referenced by the routed paths."""
    present = {0}
    for t in routed_paths:
        present.update(_trace_layers(t))
    return sorted(present)


def _segments_by_layer(routed_paths: list) -> dict[int, list[tuple]]:
    """Group routed planar segments by layer as ((x0,y0),(x1,y1)) cell pairs."""
    seg: dict[int, list[tuple]] = defaultdict(list)
    for t in routed_paths:
        layers = _trace_layers(t)
        for i in range(len(t.path) - 1):
            l0, l1 = layers[i], layers[i + 1]
            if l0 != l1:
                continue   # via transition — no planar copper segment
            seg[l0].append((t.path[i], t.path[i + 1]))
    return seg


def _all_vias(routed_paths: list) -> list[tuple[int, int]]:
    """Unique via (x, y) cell coordinates across all routed paths."""
    vias: set[tuple[int, int]] = set()
    for t in routed_paths:
        for v in getattr(t, "vias", []):
            vias.add(tuple(v))
    return sorted(vias)


def _all_pins(graph: CircuitGraph) -> list[tuple[float, float]]:
    """Absolute (x, y) cell positions of every component pin (through-holes)."""
    pins: list[tuple[float, float]] = []
    for comp in graph.nodes.values():
        for pin in comp.pins:
            pins.append((pin.abs_x, pin.abs_y))
    return pins


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Gerber + Excellon writers (gerbonara)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_copper_gerbers(
    graph: CircuitGraph, routed_paths: list, cfg: ExportConfig,
    board: str, out_dir: Path,
) -> dict[int, Path]:
    """Write one copper Gerber per routed layer (traces + pads + vias).

    Returns:
        Dict mapping layer index -> written Gerber path.
    """
    layers   = _collect_layers(routed_paths)
    n_layers = len(layers)
    seg_by_l = _segments_by_layer(routed_paths)
    pins     = _all_pins(graph)
    vias     = _all_vias(routed_paths)

    trace_ap = ap.CircleAperture(cfg.trace_w, unit=MM)
    pad_ap   = ap.CircleAperture(cfg.pad_dia, unit=MM)

    written: dict[int, Path] = {}
    for layer in layers:
        objs: list = []
        # Routed traces on this layer
        for (x0, y0), (x1, y1) in seg_by_l.get(layer, []):
            objs.append(go.Line(cfg.mm(x0), cfg.mm(y0),
                                cfg.mm(x1), cfg.mm(y1),
                                aperture=trace_ap, unit=MM))
        # Pads (through-hole) appear on every copper layer
        for px, py in pins:
            objs.append(go.Flash(cfg.mm(px), cfg.mm(py), aperture=pad_ap, unit=MM))
        # Via pads on every copper layer
        for vx, vy in vias:
            objs.append(go.Flash(cfg.mm(vx), cfg.mm(vy), aperture=pad_ap, unit=MM))

        gf = gn.GerberFile(objects=objs)
        name = _layer_name(layer, max(n_layers, layer + 1))
        path = out_dir / f"{board}-{name}.gbr"
        gf.save(str(path))
        written[layer] = path
    return written


def _write_outline_gerber(
    graph: CircuitGraph, cfg: ExportConfig, board: str, out_dir: Path,
) -> Path:
    """Write the Edge_Cuts board-outline Gerber (grid bounds + margin)."""
    gw = graph.metadata.width
    gh = graph.metadata.height
    m  = cfg.margin_mm
    x0, y0 = -m, -m
    x1, y1 = cfg.mm(gw) + m, cfg.mm(gh) + m
    edge_ap = ap.CircleAperture(0.1, unit=MM)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    objs = [
        go.Line(corners[i][0], corners[i][1],
                corners[i + 1][0], corners[i + 1][1],
                aperture=edge_ap, unit=MM)
        for i in range(len(corners) - 1)
    ]
    gf = gn.GerberFile(objects=objs)
    path = out_dir / f"{board}-Edge_Cuts.gbr"
    gf.save(str(path))
    return path


def _write_drill(
    graph: CircuitGraph, routed_paths: list, cfg: ExportConfig,
    board: str, out_dir: Path,
) -> tuple[Path, int]:
    """Write the Excellon drill file (one hit per via + per through-hole pad).

    Returns:
        (drill path, total hit count).
    """
    via_tool = ap.ExcellonTool(cfg.via_drill, plated=True, unit=MM)
    pad_tool = ap.ExcellonTool(cfg.pad_drill, plated=True, unit=MM)

    objs: list = []
    for px, py in _all_pins(graph):
        objs.append(go.Flash(cfg.mm(px), cfg.mm(py), aperture=pad_tool, unit=MM))
    for vx, vy in _all_vias(routed_paths):
        objs.append(go.Flash(cfg.mm(vx), cfg.mm(vy), aperture=via_tool, unit=MM))

    ef = gn.ExcellonFile(objects=objs)
    path = out_dir / f"{board}.drl"
    ef.save(str(path))
    return path, len(objs)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — BOM CSV + refdes derivation
# ═══════════════════════════════════════════════════════════════════════════════

def _component_value(comp: Component) -> str:
    """Best-effort human value string from a component's properties."""
    p = comp.properties or {}
    for key in ("value", "resistance_ohm", "capacitance_uf",
                "forward_voltage_v", "package"):
        if key in p:
            return str(p[key])
    return ""


def derive_refdes(graph: CircuitGraph) -> "OrderedDict[str, str]":
    """Map each component id to a reference designator (deterministic).

    Uses Component.name if it already looks like a refdes (e.g. "R1"); otherwise
    derives a type prefix and numbers sequentially per prefix.

    Returns:
        OrderedDict comp_id -> refdes, ordered by comp_id.
    """
    refmap: "OrderedDict[str, str]" = OrderedDict()
    counters: dict[str, int] = defaultdict(int)
    used: set[str] = set()

    for cid in sorted(graph.nodes.keys()):
        comp = graph.nodes[cid]
        name = (comp.name or "").strip()
        if _REFDES_RE.match(name) and name not in used:
            refmap[cid] = name
            used.add(name)
            continue
        prefix = _REFDES_PREFIX.get(comp.comp_type.upper(), "X")
        counters[prefix] += 1
        ref = f"{prefix}{counters[prefix]}"
        while ref in used:
            counters[prefix] += 1
            ref = f"{prefix}{counters[prefix]}"
        refmap[cid] = ref
        used.add(ref)
    return refmap


def _write_bom(
    graph: CircuitGraph, refmap: dict, out_dir: Path,
) -> tuple[Path, int]:
    """Write a grouped BOM CSV.  Returns (path, total quantity)."""
    groups: "OrderedDict[tuple, list[str]]" = OrderedDict()
    meta: dict[tuple, tuple] = {}
    for cid in sorted(graph.nodes.keys()):
        comp = graph.nodes[cid]
        value = _component_value(comp)
        footprint = f"{comp.footprint.width}x{comp.footprint.height}"
        key = (comp.comp_type, value, footprint)
        groups.setdefault(key, []).append(refmap[cid])
        meta[key] = (comp.comp_type, value, footprint)

    path = out_dir / "bom.csv"
    total_qty = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Reference(s)", "Quantity", "Type", "Value", "Footprint"])
        for key, refs in groups.items():
            ctype, value, footprint = meta[key]
            refs_sorted = sorted(refs)
            w.writerow([", ".join(refs_sorted), len(refs_sorted),
                        ctype, value, footprint])
            total_qty += len(refs_sorted)
    return path, total_qty


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — KiCad netlist
# ═══════════════════════════════════════════════════════════════════════════════

def _esc(s: str) -> str:
    """Escape a string for a KiCad S-expression quoted token."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _write_kicad_netlist(
    graph: CircuitGraph, refmap: dict, board: str, out_dir: Path,
) -> Path:
    """Write a valid KiCad S-expression netlist (.net)."""
    # Net membership from pins (complete), net_type from edges where available.
    net_type: dict[str, str] = {e.net_id: e.net_type for e in graph.edges}
    nets: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for cid in sorted(graph.nodes.keys()):
        comp = graph.nodes[cid]
        for pin in comp.pins:
            if not pin.net:
                continue
            nets.setdefault(pin.net, []).append((refmap[cid], pin.id))
    for e in graph.edges:                      # ensure edge nets appear too
        nets.setdefault(e.net_id, [])

    lines: list[str] = []
    lines.append('(export (version "E")')
    lines.append('  (design')
    lines.append(f'    (source "{_esc(board)}")')
    lines.append('    (tool "EDA Engine Phase 9"))')
    # Components
    lines.append("  (components")
    for cid in sorted(graph.nodes.keys()):
        comp = graph.nodes[cid]
        ref = refmap[cid]
        value = _component_value(comp) or comp.name or comp.comp_type
        fp = f"EDA:{comp.comp_type}_{comp.footprint.width}x{comp.footprint.height}"
        lines.append(f'    (comp (ref "{_esc(ref)}")')
        lines.append(f'      (value "{_esc(value)}")')
        lines.append(f'      (footprint "{_esc(fp)}"))')
    lines.append("  )")
    # Nets
    lines.append("  (nets")
    for code, (net_name, nodes) in enumerate(nets.items(), start=1):
        lines.append(f'    (net (code "{code}") (name "{_esc(net_name)}")')
        for ref, pin_id in nodes:
            lines.append(f'      (node (ref "{_esc(ref)}") (pin "{_esc(pin_id)}")))')
        lines.append("    )")
    lines.append("  )")
    lines.append(")")

    path = out_dir / f"{board}.net"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Preview render + zip bundle
# ═══════════════════════════════════════════════════════════════════════════════

def _render_preview(
    graph: CircuitGraph, routed_paths: list, cfg: ExportConfig,
    out_path: Path,
) -> Path:
    """Render copper layers + outline to a preview PNG (matplotlib).

    gerbonara produces the authoritative RS-274X / Excellon fab files; this
    PNG is a faithful raster of the same exported geometry for quick eyeballing.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers   = _collect_layers(routed_paths)
    seg_by_l = _segments_by_layer(routed_paths)

    fig, ax = plt.subplots(figsize=(13, 11), facecolor=_BG)
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(_GRID_C)
    ax.tick_params(colors=_DIM_C, labelsize=7)
    ax.set_aspect("equal")
    ax.set_title("Phase 9 — Manufacturing Preview (copper + outline)",
                 color=_TEXT_C, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("mm", color=_DIM_C); ax.set_ylabel("mm", color=_DIM_C)

    # Board outline
    gw, gh, m = graph.metadata.width, graph.metadata.height, cfg.margin_mm
    ax.add_patch(mpatches.Rectangle(
        (-m, -m), cfg.mm(gw) + 2 * m, cfg.mm(gh) + 2 * m,
        fill=False, edgecolor="#FFD700", lw=1.5, zorder=1))

    # Traces per layer
    for layer in layers:
        col = _LAYER_COLORS[layer % len(_LAYER_COLORS)]
        for (x0, y0), (x1, y1) in seg_by_l.get(layer, []):
            ax.plot([cfg.mm(x0), cfg.mm(x1)], [cfg.mm(y0), cfg.mm(y1)],
                    color=col, lw=cfg.trace_w * 3, alpha=0.85, zorder=3,
                    solid_capstyle="round")
    # Pads + vias
    for px, py in _all_pins(graph):
        ax.add_patch(mpatches.Circle((cfg.mm(px), cfg.mm(py)),
                     cfg.pad_dia / 2, color="#cccccc", zorder=4))
    for vx, vy in _all_vias(routed_paths):
        ax.add_patch(mpatches.Circle((cfg.mm(vx), cfg.mm(vy)),
                     cfg.pad_dia / 2, color="#ffffff",
                     ec="#000", lw=0.6, zorder=5))

    handles = [mpatches.Patch(color=_LAYER_COLORS[L % len(_LAYER_COLORS)],
                              label=_layer_name(L, len(layers)))
               for L in layers]
    handles.append(mpatches.Patch(color="#FFD700", label="Edge_Cuts"))
    ax.legend(handles=handles, fontsize=8, facecolor="#16162a",
              edgecolor=_GRID_C, labelcolor=_TEXT_C)
    ax.autoscale_view()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return out_path


def _bundle_zip(
    copper: dict[int, Path], outline: Path, drill: Path,
    board: str, out_dir: Path,
) -> Path:
    """Bundle all Gerbers + drill into a fab-ready zip."""
    path = out_dir / f"{board}_gerbers.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in copper.values():
            zf.write(p, p.name)
        zf.write(outline, outline.name)
        zf.write(drill, drill.name)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def _count_unrouted_nets(graph: CircuitGraph, routed_paths: list) -> int:
    """Number of nets with fewer routed segments than required (chain routing).

    Expected segments for a net = (distinct components on the net) - 1.
    """
    net_comps: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        net_comps[e.net_id].add(e.source[0])
        net_comps[e.net_id].add(e.target[0])
    routed_segs: Counter = Counter(t.net_id for t in routed_paths)

    unrouted = 0
    for nid, comps in net_comps.items():
        expected = max(0, len(comps) - 1)
        if routed_segs.get(nid, 0) < expected:
            unrouted += 1
    return unrouted


def run_phase9(graph: CircuitGraph, routed_paths: list) -> dict:
    """Phase 9 pipeline: placed + routed board -> manufacturing files.

    Args:
        graph:        Placed CircuitGraph (read-only here).
        routed_paths: Routed traces from run_phase3 (RoutedTrace) or run_phase8
                      (LayeredTrace).  Single-layer paths are treated as layer 0.

    Returns:
        Dict of written file paths plus completion info:
        {
          "copper_gerbers": {layer: path}, "outline_gerber": path,
          "drill": path, "drill_hits": int, "bom": path, "kicad_netlist": path,
          "preview": path, "zip": path, "board": str, "layers": [int],
          "unrouted_nets": int, "completion_pct": float,
        }
    """
    _MFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ExportConfig()
    board = re.sub(r"[^A-Za-z0-9_.-]", "_", graph.metadata.name or "board")

    # DRC gate: completion (still export if < 100%, but report the shortfall).
    unrouted = _count_unrouted_nets(graph, routed_paths)
    total_nets = len({e.net_id for e in graph.edges})
    completion = (100.0 if total_nets == 0
                  else round((total_nets - unrouted) / total_nets * 100.0, 1))
    if unrouted > 0:
        print(f"[Phase 9] WARNING: {unrouted} net(s) not fully routed "
              f"({completion:.1f}% complete) — exporting anyway.")

    print(f"[Phase 9] Exporting '{board}' to {_MFG_DIR} ...")
    copper  = _write_copper_gerbers(graph, routed_paths, cfg, board, _MFG_DIR)
    outline = _write_outline_gerber(graph, cfg, board, _MFG_DIR)
    drill, hits = _write_drill(graph, routed_paths, cfg, board, _MFG_DIR)

    refmap = derive_refdes(graph)
    bom, total_qty = _write_bom(graph, refmap, _MFG_DIR)
    netlist = _write_kicad_netlist(graph, refmap, board, _MFG_DIR)

    preview = _render_preview(graph, routed_paths, cfg,
                              _OUTPUT_DIR / "phase9_preview.png")
    zip_path = _bundle_zip(copper, outline, drill, board, _MFG_DIR)

    print(f"[Phase 9] Copper layers : {sorted(copper)}  "
          f"({len(copper)} Gerber(s))")
    print(f"[Phase 9] Drill hits    : {hits}")
    print(f"[Phase 9] BOM rows total qty : {total_qty} (components: {len(graph.nodes)})")
    print(f"[Phase 9] Bundle        : {zip_path}")

    return {
        "board":          board,
        "layers":         _collect_layers(routed_paths),
        "copper_gerbers": copper,
        "outline_gerber": outline,
        "drill":          drill,
        "drill_hits":     hits,
        "bom":            bom,
        "bom_total_qty":  total_qty,
        "kicad_netlist":  netlist,
        "preview":        preview,
        "zip":            zip_path,
        "unrouted_nets":  unrouted,
        "completion_pct": completion,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys

    from phase1_eda_engine import NetlistParser, InitialPlacer
    from phase2_genetic_placer import run_phase2
    from phase8_multilayer_router import run_phase8

    _sample = _PROJECT_ROOT / "netlists" / "sample_netlist.json"
    with _sample.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)
    graph = run_phase2(graph)
    graph, traces, _metrics = run_phase8(graph)

    result = run_phase9(graph, traces)
    print("\n[Phase 9] Manufacturing export complete:")
    for k, v in result.items():
        print(f"   {k}: {v}")
    sys.exit(0)
