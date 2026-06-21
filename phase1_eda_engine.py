"""
Phase 1 — Netlist Parser, Circuit Graph Builder, and 2D Grid Visualizer

Sections
--------
1. Data models   — Pin, Component, Net, GridMetadata, Netlist, GraphEdge, CircuitGraph
2. NetlistParser — Factory: JSON dict → typed Netlist object
3. CircuitGraph  — Adapter: Netlist → adjacency graph via star-expansion per net
4. HPWL          — Half-perimeter wire-length fitness metric
5. InitialPlacer — Strategy: assigns non-overlapping seed positions on the grid
6. Visualizer    — Dark-mode matplotlib canvas → outputs/phase1_output.png
7. Pipeline      — run_phase1(), run_phase0_to_phase1(), CLI entry-point

Phase handoff contract (CLAUDE.md):
  in  ← Phase 0: raw JSON dict with root key "netlist"
  out → Phase 2: CircuitGraph with Component.x / Component.y set at seed positions
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib
matplotlib.use("Agg")            # headless — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_SAMPLE_JSON  = _PROJECT_ROOT / "netlists" / "sample_netlist.json"

# ---------------------------------------------------------------------------
# Colour palette (dark-mode)
# ---------------------------------------------------------------------------
_BG       = "#0f0f1a"
_PANEL_BG = "#16162a"
_GRID_C   = "#1e1e3a"
_TEXT_C   = "#e0e0ff"
_DIM_C    = "#888899"

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
_PIN_COLORS: dict[str, str] = {
    "OUTPUT":        "#FFD700",
    "INPUT":         "#FF6B6B",
    "PASSIVE":       "#aaaaaa",
    "POWER":         "#FF4444",
    "BIDIRECTIONAL": "#DA70D6",
}

_VALID_PIN_TYPES  = set(_PIN_COLORS)
_VALID_NET_TYPES  = set(_NET_COLORS)
_VALID_COMP_TYPES = set(_COMP_COLORS)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pin:
    """A single electrical pin on a Component.

    Attributes:
        id:       Pin identifier within the parent component (e.g. "GPIO2").
        pin_type: Directionality — one of OUTPUT | INPUT | PASSIVE | POWER | BIDIRECTIONAL.
        net:      ID of the Net this pin connects to.
        abs_x:    Absolute x-coordinate on the grid (set after placement).
        abs_y:    Absolute y-coordinate on the grid (set after placement).
    """
    id: str
    pin_type: str
    net: str
    abs_x: float = 0.0
    abs_y: float = 0.0


@dataclass
class Footprint:
    """Physical bounding box of a component on the grid (in grid units)."""
    width: int
    height: int


@dataclass
class Component:
    """A circuit component with placement information.

    Only Component.x and Component.y are mutable after parsing.
    All other fields are set by NetlistParser and treated as immutable.
    """
    id: str
    comp_type: str
    name: str
    pins: list[Pin]
    footprint: Footprint
    x: int
    y: int
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Net:
    """An electrical net connecting one or more component pins.

    Attributes:
        id:             Net identifier (e.g. "LED_DRIVE").
        net_type:       POWER | SIGNAL | GROUND.
        connected_pins: List of (component_id, pin_id) tuples.
    """
    id: str
    net_type: str
    connected_pins: list[tuple[str, str]]


@dataclass
class GridMetadata:
    """Dimensions and unit of the routing grid."""
    width: int
    height: int
    unit: str
    name: str = ""


@dataclass
class Netlist:
    """Top-level container produced by NetlistParser."""
    metadata: GridMetadata
    components: list[Component]
    nets: list[Net]


@dataclass
class GraphEdge:
    """A directed edge in the CircuitGraph representing a net connection.

    Produced by star-expansion: one hub component connects to every other
    component on the same net.  The edge carries net metadata for the router.
    """
    net_id: str
    net_type: str
    source: tuple[str, str]   # (component_id, pin_id)
    target: tuple[str, str]   # (component_id, pin_id)
    weight: float = 1.0


@dataclass
class CircuitGraph:
    """Adapter over a Netlist that exposes a graph interface.

    Attributes:
        nodes:     Dict mapping component_id → Component.
        edges:     All GraphEdge objects (star-expanded, one per hub–spoke pair).
        adjacency: Dict mapping component_id → set of adjacent component_ids.
        metadata:  Grid dimensions (passed through for downstream phases).
    """
    nodes: dict[str, Component]
    edges: list[GraphEdge]
    adjacency: dict[str, set[str]]
    metadata: GridMetadata

    @classmethod
    def from_netlist(cls, netlist: Netlist) -> "CircuitGraph":
        """Build a CircuitGraph from a Netlist using star-expansion per net.

        For each net with N connected components, one component is chosen as
        the hub and N-1 edges are created (hub → each spoke).  This is the
        minimum spanning tree approximation used by EDA tools.

        Args:
            netlist: Parsed Netlist from Phase 1.

        Returns:
            A fully populated CircuitGraph ready for Phase 2.
        """
        nodes: dict[str, Component] = {c.id: c for c in netlist.components}
        adjacency: dict[str, set[str]] = {c.id: set() for c in netlist.components}
        edges: list[GraphEdge] = []

        for net in netlist.nets:
            # Build comp_id → pin_id map for this net (first pin wins per comp)
            comp_pin: dict[str, str] = {}
            for cid, pid in net.connected_pins:
                if cid not in comp_pin and cid in nodes:
                    comp_pin[cid] = pid

            comp_ids = list(comp_pin.keys())
            if len(comp_ids) < 2:
                continue

            hub = comp_ids[0]
            for spoke in comp_ids[1:]:
                edge = GraphEdge(
                    net_id=net.id,
                    net_type=net.net_type,
                    source=(hub, comp_pin[hub]),
                    target=(spoke, comp_pin[spoke]),
                    weight=1.0,
                )
                edges.append(edge)
                adjacency[hub].add(spoke)
                adjacency[spoke].add(hub)

        return cls(
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            metadata=netlist.metadata,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — NetlistParser  (Factory pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class NetlistParser:
    """Factory that converts a raw JSON dict (Phase 0 output) into a typed Netlist.

    Accepts either a dict or a JSON string.  The input must have a root key
    "netlist" containing "metadata", "components", and "nets".
    """

    def parse(self, data: dict | str) -> Netlist:
        """Parse raw netlist data into a typed Netlist object.

        Args:
            data: Either a dict (direct from run_phase0) or a JSON string.

        Returns:
            A fully-populated Netlist with typed Python objects.

        Raises:
            ValueError: If the root structure is missing required keys.
        """
        if isinstance(data, str):
            data = json.loads(data)

        raw = data.get("netlist", data)   # tolerate both {netlist:{...}} and {...}

        if "components" not in raw or "nets" not in raw:
            raise ValueError("Netlist JSON must have 'components' and 'nets' keys")

        meta_raw = raw.get("metadata", {})
        grid_raw = meta_raw.get("grid", {})
        metadata = GridMetadata(
            width=int(grid_raw.get("width",  24)),
            height=int(grid_raw.get("height", 20)),
            unit=grid_raw.get("unit", "mm"),
            name=meta_raw.get("name", "unnamed"),
        )

        components = [self._parse_component(c) for c in raw["components"]]
        nets       = [self._parse_net(n)       for n in raw["nets"]]

        return Netlist(metadata=metadata, components=components, nets=nets)

    def _parse_component(self, raw: dict) -> Component:
        """Parse a single component dict into a typed Component.

        Args:
            raw: Dict from the JSON "components" array.

        Returns:
            A Component with a list of typed Pin objects.
        """
        fp_raw = raw.get("footprint", {})
        footprint = Footprint(
            width=int(fp_raw.get("width", 1)),
            height=int(fp_raw.get("height", 1)),
        )
        pins = [self._parse_pin(p) for p in raw.get("pins", [])]
        return Component(
            id=raw["id"],
            comp_type=raw.get("type", "IC"),
            name=raw.get("name", raw["id"]),
            pins=pins,
            footprint=footprint,
            x=int(raw.get("x", 0)),
            y=int(raw.get("y", 0)),
            properties=raw.get("properties", {}),
        )

    def _parse_pin(self, raw: dict) -> Pin:
        """Parse a single pin dict into a typed Pin.

        Args:
            raw: Dict from a component's "pins" array.

        Returns:
            A Pin object (abs_x / abs_y default to 0.0; updated after placement).
        """
        return Pin(
            id=raw["id"],
            pin_type=raw.get("type", "PASSIVE"),
            net=raw.get("net", ""),
        )

    def _parse_net(self, raw: dict) -> Net:
        """Parse a single net dict into a typed Net.

        Args:
            raw: Dict from the JSON "nets" array.

        Returns:
            A Net whose connected_pins is a list of (comp_id, pin_id) tuples.
        """
        connected = [
            (entry["component_id"], entry["pin_id"])
            for entry in raw.get("connected_pins", [])
        ]
        return Net(
            id=raw["id"],
            net_type=raw.get("type", "SIGNAL"),
            connected_pins=connected,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — HPWL  (used by Phase 2 GA fitness function)
# ═══════════════════════════════════════════════════════════════════════════════

def half_perimeter_wire_length(graph: CircuitGraph) -> float:
    """Compute the total Half-Perimeter Wire Length over all nets.

    For each net, finds the bounding box of all connected component centres
    and sums its half perimeter: (max_x − min_x) + (max_y − min_y).
    Lower HPWL = better placement.

    Args:
        graph: CircuitGraph with current Component.x / Component.y values.

    Returns:
        Total HPWL as a float (0.0 if the graph has no edges).
    """
    # Collect component centres per net from the edge list
    net_xs: dict[str, list[float]] = defaultdict(list)
    net_ys: dict[str, list[float]] = defaultdict(list)

    for edge in graph.edges:
        for cid in (edge.source[0], edge.target[0]):
            comp = graph.nodes[cid]
            cx = comp.x + comp.footprint.width  / 2.0
            cy = comp.y + comp.footprint.height / 2.0
            net_xs[edge.net_id].append(cx)
            net_ys[edge.net_id].append(cy)

    total = 0.0
    for net_id in net_xs:
        xs = net_xs[net_id]
        ys = net_ys[net_id]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4b — Placer interface  (Strategy pattern contract)
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Placer(Protocol):
    """Strategy interface shared by all swappable placement algorithms.

    A Placer takes a CircuitGraph, optimises component positions, and returns
    the SAME CircuitGraph with only Component.x / Component.y mutated (and
    Pin.abs_x / Pin.abs_y refreshed).  Nothing else on the graph may change.

    Both Phase 2 (``run_phase2``, Genetic Algorithm) and Phase 7
    (``run_phase7`` / ``RLPlacer``, Reinforcement Learning) satisfy this
    callable signature, so they are interchangeable at the pipeline level.
    """

    def __call__(self, graph: "CircuitGraph") -> "CircuitGraph":
        """Place components and return the graph with positions optimised."""
        ...


@runtime_checkable
class Router(Protocol):
    """Strategy interface shared by all swappable routing algorithms.

    A Router reads Component.x / Component.y (and pin positions) from a
    CircuitGraph and returns ``(graph, traces, metrics)`` WITHOUT mutating the
    graph's topology — only producing trace paths as separate output.

    Both Phase 3 (``run_phase3``, single-layer Lee's maze router) and Phase 8
    (``run_phase8``, multi-layer router with vias) satisfy this signature, so
    they are interchangeable at the pipeline level.
    """

    def __call__(self, graph: "CircuitGraph") -> tuple["CircuitGraph", list, dict]:
        """Route all nets and return (graph, traces, metrics dict)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — InitialPlacer  (Strategy pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class InitialPlacer:
    """Assigns valid seed positions to all components on the grid.

    Strategy:
    - If all positions from the JSON are already valid (in-bounds, non-overlapping),
      keep them and only update pin absolute coordinates.
    - Otherwise, fall back to a row-based deterministic placement that respects
      EEE rules (MCU/IC away from edges, passives anywhere inside grid).

    After placement, updates Pin.abs_x / Pin.abs_y for every pin on every component.
    """

    # EEE: MCU/IC must stay inside this inner zone
    _MCU_X_MIN, _MCU_X_MAX = 3, None   # None → computed from grid width
    _MCU_Y_MIN, _MCU_Y_MAX = 3, None

    def __init__(self, metadata: GridMetadata) -> None:
        """Initialise with the grid metadata.

        Args:
            metadata: GridMetadata carrying width, height, unit.
        """
        self.meta = metadata

    def place(self, netlist: Netlist) -> None:
        """Place all components, then update pin absolute positions.

        Args:
            netlist: Netlist whose Component.x / .y values may be modified.
        """
        if self._all_valid(netlist):
            print("[Phase 1] Placer: JSON positions are valid - keeping seed positions.")
        else:
            print("[Phase 1] Placer: Invalid JSON positions - running row-based fallback.")
            self._assign_positions(netlist)

        self._update_pin_positions(netlist)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _all_valid(self, netlist: Netlist) -> bool:
        """Return True if every component is in-bounds and no two overlap."""
        comps = netlist.components
        for comp in comps:
            if not self._in_bounds(comp):
                return False
        for i, a in enumerate(comps):
            for b in comps[i + 1:]:
                if self._overlaps(a, b):
                    return False
        return True

    def _in_bounds(self, comp: Component) -> bool:
        """Return True if the component footprint lies entirely within the grid."""
        return (
            comp.x >= 0
            and comp.y >= 0
            and comp.x + comp.footprint.width  <= self.meta.width
            and comp.y + comp.footprint.height <= self.meta.height
        )

    def _overlaps(self, a: Component, b: Component) -> bool:
        """Return True if two component footprints share any grid cell."""
        return not (
            a.x + a.footprint.width  <= b.x
            or b.x + b.footprint.width  <= a.x
            or a.y + a.footprint.height <= b.y
            or b.y + b.footprint.height <= a.y
        )

    # ------------------------------------------------------------------
    # Fallback row-based placer
    # ------------------------------------------------------------------

    def _assign_positions(self, netlist: Netlist) -> None:
        """Row-based placement: MCU/IC first in centre zone, passives after.

        Guarantees no overlaps and respects EEE edge-clearance rules.
        """
        grid_w = self.meta.width
        priority = {"MCU", "IC"}

        # Sort: high-frequency components first
        ordered = (
            sorted([c for c in netlist.components if c.comp_type in priority],
                   key=lambda c: c.id)
            + sorted([c for c in netlist.components if c.comp_type not in priority],
                     key=lambda c: c.id)
        )

        cur_x, cur_y = 4, 4
        row_h = 0

        for comp in ordered:
            fw, fh = comp.footprint.width, comp.footprint.height
            # Wrap to next row if we'd exceed the right edge
            if cur_x + fw > grid_w - 2:
                cur_x = 4
                cur_y += row_h + 2
                row_h = 0
            comp.x = cur_x
            comp.y = cur_y
            cur_x += fw + 2
            row_h = max(row_h, fh)

    # ------------------------------------------------------------------
    # Pin position update
    # ------------------------------------------------------------------

    def _update_pin_positions(self, netlist: Netlist) -> None:
        """Set Pin.abs_x / Pin.abs_y using perimeter-based pin placement.

        Looks up each component in the component library to determine which
        side each pin belongs to, then distributes pins evenly along that side.
        Unrecognised pins fall back to the bottom edge.
        """
        from component_library import lookup as lib_lookup

        for comp in netlist.components:
            comp_def = lib_lookup(comp.name, pin_count=len(comp.pins))
            # Build name → side mapping from library (case-insensitive)
            name_to_side: dict[str, str] = {
                pd.name.lower(): pd.side for pd in comp_def.pin_defs
            }

            # Group pins by their assigned side
            sides: dict[str, list] = {"left": [], "right": [], "top": [], "bottom": []}
            for pin in comp.pins:
                side = name_to_side.get(pin.id.lower(), "bottom")
                sides[side].append(pin)

            fw = comp.footprint.width
            fh = comp.footprint.height

            def _space(n: int, length: float) -> list[float]:
                """Evenly spaced positions for n pins along a line of given length."""
                if n == 0:
                    return []
                if n == 1:
                    return [length / 2.0]
                return [length * (i + 1) / (n + 1) for i in range(n)]

            # Left edge: x = comp.x, y varies along height
            for pin, off in zip(sides["left"], _space(len(sides["left"]), fh)):
                pin.abs_x = float(comp.x)
                pin.abs_y = comp.y + off

            # Right edge: x = comp.x + fw, y varies along height
            for pin, off in zip(sides["right"], _space(len(sides["right"]), fh)):
                pin.abs_x = float(comp.x + fw)
                pin.abs_y = comp.y + off

            # Top edge: y = comp.y + fh, x varies along width
            for pin, off in zip(sides["top"], _space(len(sides["top"]), fw)):
                pin.abs_x = comp.x + off
                pin.abs_y = float(comp.y + fh)

            # Bottom edge (default for unrecognised pins): y = comp.y
            for pin, off in zip(sides["bottom"], _space(len(sides["bottom"]), fw)):
                pin.abs_x = comp.x + off
                pin.abs_y = float(comp.y)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(graph: CircuitGraph, output_path: Path | None = None) -> Path:
    """Render a dark-mode EDA canvas and save it as a PNG.

    Layout: main circuit canvas (left, 75%) + analytics panel (right, 25%).

    Canvas shows:
    - Dark grid (24 × 20 cells)
    - Coloured component boxes with ID + name labels
    - Pin dots distributed along component bottom edges
    - Rat's-nest connections coloured by net type
    - Net ID labels at edge midpoints

    Analytics panel shows:
    - Design name, grid info
    - Component counts by type
    - Net counts by type
    - HPWL metric

    Args:
        graph:       CircuitGraph with placed components.
        output_path: Destination PNG path.  Defaults to outputs/phase1_output.png.

    Returns:
        The Path where the PNG was saved.
    """
    if output_path is None:
        output_path = _OUTPUT_DIR / "phase1_output.png"

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 11), facecolor=_BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.03,
                   left=0.04, right=0.98, top=0.95, bottom=0.06)

    ax_main = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    gw = graph.metadata.width
    gh = graph.metadata.height

    # ── Main canvas setup ──────────────────────────────────────────────
    ax_main.set_facecolor(_BG)
    ax_main.set_xlim(-0.5, gw + 0.5)
    ax_main.set_ylim(-0.5, gh + 0.5)
    ax_main.set_aspect("equal")
    ax_main.tick_params(colors=_DIM_C, labelsize=7)
    for spine in ax_main.spines.values():
        spine.set_color(_GRID_C)
    ax_main.set_title(
        f"Phase 1 — EDA Grid  ({gw} × {gh} {graph.metadata.unit})",
        color=_TEXT_C, fontsize=11, pad=8, fontweight="bold",
    )
    ax_main.set_xlabel("x (grid units)", color=_DIM_C, fontsize=8)
    ax_main.set_ylabel("y (grid units)", color=_DIM_C, fontsize=8)

    # Grid lines
    for x in range(gw + 1):
        ax_main.axvline(x, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)
    for y in range(gh + 1):
        ax_main.axhline(y, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)

    # ── Rat's-nest connections ─────────────────────────────────────────
    drawn_labels: set[str] = set()
    for edge in graph.edges:
        src  = graph.nodes[edge.source[0]]
        tgt  = graph.nodes[edge.target[0]]
        sx   = src.x + src.footprint.width  / 2.0
        sy   = src.y + src.footprint.height / 2.0
        tx   = tgt.x + tgt.footprint.width  / 2.0
        ty   = tgt.y + tgt.footprint.height / 2.0
        col  = _NET_COLORS.get(edge.net_type, "#aaaaaa")

        ax_main.plot(
            [sx, tx], [sy, ty],
            color=col, lw=1.5, alpha=0.65, zorder=2,
            linestyle="--" if edge.net_type == "GROUND" else "-",
        )
        # Label each unique net once, at the midpoint
        label_key = edge.net_id
        if label_key not in drawn_labels:
            mx, my = (sx + tx) / 2, (sy + ty) / 2
            ax_main.text(
                mx, my + 0.25, edge.net_id,
                color=col, fontsize=5.5, ha="center", va="bottom",
                alpha=0.9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc=_BG, ec="none", alpha=0.6),
            )
            drawn_labels.add(label_key)

    # ── Component boxes ────────────────────────────────────────────────
    for comp in graph.nodes.values():
        color = _COMP_COLORS.get(comp.comp_type, "#888888")
        rect  = mpatches.FancyBboxPatch(
            (comp.x + 0.05, comp.y + 0.05),
            comp.footprint.width  - 0.10,
            comp.footprint.height - 0.10,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="white",
            alpha=0.82,
            linewidth=1.5,
            zorder=4,
        )
        ax_main.add_patch(rect)

        # ID + name labels
        cx = comp.x + comp.footprint.width  / 2.0
        cy = comp.y + comp.footprint.height / 2.0
        ax_main.text(
            cx, cy + 0.2, comp.id,
            color="white", fontsize=7.5, ha="center", va="center",
            fontweight="bold", zorder=5,
        )
        ax_main.text(
            cx, cy - 0.5, comp.name,
            color="#ccccdd", fontsize=5.5, ha="center", va="center",
            zorder=5,
        )

        # Pin squares on component perimeter
        for pin in comp.pins:
            pin_col = _PIN_COLORS.get(pin.pin_type, "#ffffff")
            sq = mpatches.Rectangle(
                (pin.abs_x - 0.15, pin.abs_y - 0.15), 0.3, 0.3,
                facecolor="white", edgecolor=pin_col, linewidth=0.8, zorder=6,
            )
            ax_main.add_patch(sq)
            # Pin name label (4pt, truncated to 6 chars)
            label = pin.id[:6]
            ax_main.text(
                pin.abs_x, pin.abs_y + 0.22, label,
                color=pin_col, fontsize=3.5, ha="center", va="bottom",
                zorder=7, alpha=0.85,
            )

    # ── Legend (component types present) ──────────────────────────────
    present_types = {c.comp_type for c in graph.nodes.values()}
    legend_patches = [
        mpatches.Patch(color=_COMP_COLORS.get(t, "#888"), label=t)
        for t in sorted(present_types)
    ]
    ax_main.legend(
        handles=legend_patches,
        loc="upper right", fontsize=7,
        facecolor=_PANEL_BG, edgecolor=_GRID_C,
        labelcolor=_TEXT_C, framealpha=0.85,
    )

    # ═══ Analytics panel ══════════════════════════════════════════════
    ax_info.set_facecolor(_PANEL_BG)
    ax_info.axis("off")
    ax_info.set_title("Analytics", color=_TEXT_C, fontsize=9, pad=6)

    hpwl = half_perimeter_wire_length(graph)
    comp_counts = Counter(c.comp_type for c in graph.nodes.values())

    # Net type counts from edge list
    net_type_map: dict[str, str] = {}
    for e in graph.edges:
        net_type_map[e.net_id] = e.net_type
    net_counts = Counter(net_type_map.values())

    def _bar(label: str, val: int, total: int) -> str:
        filled = int(round(8 * val / max(total, 1)))
        return f"  {label:<11} {'█' * filled}{'░' * (8 - filled)} {val}"

    lines = [
        "━" * 24,
        f"  {graph.metadata.name}",
        "━" * 24,
        "",
        f"  Grid  {gw} × {gh} {graph.metadata.unit}",
        f"  Nodes  {len(graph.nodes)}",
        f"  Edges  {len(graph.edges)}",
        "",
        "  COMPONENTS",
        "  ─────────────────────",
    ]
    total_comps = len(graph.nodes)
    for ctype, cnt in sorted(comp_counts.items()):
        lines.append(_bar(ctype, cnt, total_comps))

    lines += [
        "",
        "  NETS",
        "  ─────────────────────",
    ]
    total_nets = len(net_type_map)
    for ntype, cnt in sorted(net_counts.items()):
        col = _NET_COLORS.get(ntype, "#aaa")
        lines.append(f"  {ntype:<11} {cnt:>2}")

    lines += [
        "",
        "  METRICS",
        "  ─────────────────────",
        f"  HPWL     {hpwl:>8.2f}",
        f"  (lower = better)",
        "",
        "━" * 24,
        "  Phase 0 → Phase 1",
        "  ✓ Parser   OK",
        "  ✓ Graph    OK",
        "  ✓ Placer   OK",
        "━" * 24,
    ]

    ax_info.text(
        0.05, 0.97, "\n".join(lines),
        transform=ax_info.transAxes,
        fontfamily="monospace",
        color=_TEXT_C,
        fontsize=7.2,
        va="top",
        ha="left",
        linespacing=1.55,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"[Phase 1] Canvas saved -> {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(netlist_dict: dict | str) -> CircuitGraph:
    """Phase 1 pipeline entry-point: JSON dict → CircuitGraph.

    Steps:
      1. Parse the raw dict into a typed Netlist.
      2. Run InitialPlacer (validates or reassigns component positions).
      3. Build CircuitGraph via star-expansion.
      4. Compute and print HPWL.
      5. Render and save the visualizer canvas.

    Args:
        netlist_dict: Raw dict from run_phase0(), or a JSON string.

    Returns:
        CircuitGraph ready for Phase 2 Genetic Algorithm.
    """
    print("[Phase 1] Parsing netlist ...")
    parser  = NetlistParser()
    netlist = parser.parse(netlist_dict)
    print(f"[Phase 1] Parsed  {len(netlist.components)} components, "
          f"{len(netlist.nets)} nets.")

    print("[Phase 1] Running InitialPlacer ...")
    placer = InitialPlacer(netlist.metadata)
    placer.place(netlist)

    print("[Phase 1] Building CircuitGraph ...")
    graph = CircuitGraph.from_netlist(netlist)
    print(f"[Phase 1] Graph   {len(graph.nodes)} nodes, {len(graph.edges)} edges.")
    for cid, neighbours in sorted(graph.adjacency.items()):
        print(f"          {cid} -- {{{', '.join(sorted(neighbours))}}}")

    hpwl = half_perimeter_wire_length(graph)
    print(f"[Phase 1] HPWL (seed) = {hpwl:.4f}  (Phase 2 GA will minimise this)")

    print("[Phase 1] Rendering visualizer ...")
    visualize(graph)

    return graph


def run_phase0_to_phase1(prompt: str) -> CircuitGraph:
    """Bridge: run Phase 0 (Grok API) then immediately run Phase 1.

    This is the first test of the full Phase 0 → Phase 1 pipeline.

    Args:
        prompt: Plain-English circuit description for the Grok API.

    Returns:
        CircuitGraph produced from the live Grok API response.
    """
    from phase0_groq_translator import run_phase0
    print(f"[Phase 0->1] Calling Grok API with prompt: {prompt!r}")
    netlist_dict = run_phase0(prompt)
    return run_phase1(netlist_dict)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(
        description="Phase 1 — EDA Netlist Parser, Graph Builder, and Visualizer"
    )
    cli.add_argument(
        "--netlist",
        type=Path,
        default=_SAMPLE_JSON,
        help="Path to a JSON netlist file (default: netlists/sample_netlist.json)",
    )
    cli.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="If provided, call Phase 0 (Grok API) first with this prompt.",
    )
    args = cli.parse_args()

    if args.prompt:
        graph = run_phase0_to_phase1(args.prompt)
    else:
        with args.netlist.open(encoding="utf-8") as fh:
            data = json.load(fh)
        graph = run_phase1(data)

    print("\n[Phase 1] Complete. CircuitGraph is ready for Phase 2.")
    sys.exit(0)
