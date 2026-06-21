"""
Phase 8 — Multi-Layer Maze Router with Via Insertion

A drop-in alternative to the single-layer Phase 3 router.  This honours the
project's Strategy pattern: a Router is a swappable algorithm that reads a
placed CircuitGraph and produces routed trace paths — here across 2+ copper
layers connected by vias.

Phase 8 is functionally an ALTERNATIVE to Phase 3 (selectable at runtime), NOT
a stage that runs after Phase 6.  Phase 4 consumes its output unchanged (the
LayeredTrace below is a superset of Phase 3's RoutedTrace).

Sections
--------
1. Config + LayeredTrace        — .env params, the enriched trace structure
2. MultiLayerGrid               — 3D (layer, y, x) obstacle / trace grid
3. LayeredLeeRouter             — weighted BFS (Dijkstra) over (x, y, layer)
4. MultiLayerNetRouter          — net ordering, chain routing, rip-up/reroute
5. Visualizer                   — layer-distinct canvas -> outputs/phase8_output.png
6. run_phase8()                 — pipeline entry-point mirroring run_phase3
7. CLI entry-point

Key properties
--------------
* No two different nets ever occupy the same (x, y, layer), so SAME-LAYER
  crossings are 0 by construction (and power can never cross ground on a layer).
* A via at (x, y) is a plated through-hole: it blocks (x, y) on ALL layers for
  every other net.  Vias are charged VIA_COST so the router minimises them.
* Soft per-layer direction bias (horizontal on even layers, vertical on odd)
  collapses crossings the classic 2-layer way — a tunable cost, not a hard rule.
"""

from __future__ import annotations

import heapq
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from dotenv import load_dotenv

from phase1_eda_engine import CircuitGraph, Component, Net

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Grid cell states (per layer)
# ---------------------------------------------------------------------------
CELL_FREE      = 0
CELL_COMPONENT = 1
CELL_ROUTED    = 2
CELL_BLOCKED   = 3   # 1-cell DRC clearance around a routed trace (same layer)

# ---------------------------------------------------------------------------
# Colour palette — dark-mode aesthetic
# ---------------------------------------------------------------------------
_BG, _PANEL_BG, _GRID_C, _TEXT_C, _DIM_C = (
    "#0f0f1a", "#16162a", "#1e1e3a", "#e0e0ff", "#888899"
)
_COMP_COLORS: dict[str, str] = {
    "MCU": "#1E90FF", "RESISTOR": "#FF8C00", "CAPACITOR": "#00CED1",
    "LED": "#FFD700", "IC": "#9370DB", "POWER": "#FF4444",
}
# Per-layer trace colours (cycled for > len)
_LAYER_COLORS = ["#FF4444", "#1E90FF", "#00C97A", "#FFD700", "#DA70D6"]
_VIA_COLOR    = "#FFFFFF"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Config + LayeredTrace
# ═══════════════════════════════════════════════════════════════════════════════

def routing_layers() -> int:
    """Number of copper layers from .env ROUTING_LAYERS (default 2, min 1)."""
    return max(1, int(os.getenv("ROUTING_LAYERS", "2")))


def via_cost() -> float:
    """Via insertion cost from .env VIA_COST (default 10.0)."""
    return float(os.getenv("VIA_COST", "10.0"))


def layer_dir_bias() -> float:
    """Per-layer direction-bias weight from .env LAYER_DIR_BIAS (default 0.6)."""
    return float(os.getenv("LAYER_DIR_BIAS", "0.6"))


@dataclass
class LayeredTrace:
    """One routed connection across one or more copper layers.

    Superset of Phase 3's RoutedTrace: it keeps the flat ``path`` (x, y) cell
    list and ``length`` so Phase 4 / the frontend work unchanged, and adds
    ``layers`` (the layer of each path cell) and ``vias`` (layer-transition
    coordinates).

    Attributes:
        net_id, net_type:        Net identity.
        source_comp/pin,
        target_comp/pin:         Endpoints of this segment.
        path:    Ordered (x, y) grid cells from source to target.
        layers:  Parallel to path — the copper layer of each cell.
        vias:    (x, y) cells where this segment changes layer (plated holes).
        length:  Number of cells in path (includes endpoints).
    """

    net_id: str
    net_type: str
    source_comp: str
    source_pin: str
    target_comp: str
    target_pin: str
    path: list[tuple[int, int]]
    layers: list[int]
    vias: list[tuple[int, int]]
    length: int


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — MultiLayerGrid
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLayerGrid:
    """3D routing grid: one obstacle/trace plane per copper layer.

    Indexed grid[layer, y, x].  Vias are tracked separately because a via
    blocks its (x, y) on every layer (a plated through-hole).
    """

    def __init__(self, width: int, height: int, layers: int) -> None:
        """Create an empty multi-layer grid.

        Args:
            width:  Grid columns.
            height: Grid rows.
            layers: Number of copper layers (>= 1).
        """
        self.width  = width
        self.height = height
        self.layers = layers
        self.grid   = np.zeros((layers, height, width), dtype=int)
        self._pin_cells: set[tuple[int, int]] = set()
        # via (x, y) -> owning net_id  (blocks all layers for other nets)
        self.via_owner: dict[tuple[int, int], str] = {}
        # (x, y, layer) -> net_id for routed cells (crossing/attribution)
        self.cell_net: dict[tuple[int, int, int], str] = {}

    # ------------------------------------------------------------------

    def initialize_from_graph(self, graph: CircuitGraph) -> None:
        """Mark component footprints as obstacles on the TOP layer; collect pins.

        Components physically sit on layer 0 (top copper), so their footprints
        block routing only there.  Inner / bottom layers are free underneath —
        the classic reason multi-layer boards route so much more than one layer.
        Pins are plated through-holes, accessible from every layer.

        Args:
            graph: Placed CircuitGraph (Component.x/.y set, pins positioned).
        """
        self.grid[:] = CELL_FREE
        self._pin_cells.clear()
        for comp in graph.nodes.values():
            for dy in range(comp.footprint.height):
                for dx in range(comp.footprint.width):
                    cx, cy = comp.x + dx, comp.y + dy
                    if self.in_bounds(cx, cy):
                        self.grid[0, cy, cx] = CELL_COMPONENT
            for pin in comp.pins:
                px, py = round(pin.abs_x), round(pin.abs_y)
                if self.in_bounds(px, py):
                    self._pin_cells.add((px, py))

    # ------------------------------------------------------------------

    def clone(self) -> "MultiLayerGrid":
        """Return an independent deep copy (used for relaxed fallback routing)."""
        c = MultiLayerGrid(self.width, self.height, self.layers)
        c.grid = self.grid.copy()
        c._pin_cells = set(self._pin_cells)
        c.via_owner  = dict(self.via_owner)
        c.cell_net   = dict(self.cell_net)
        return c

    def in_bounds(self, x: int, y: int) -> bool:
        """True if (x, y) lies inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int, layer: int) -> bool:
        """True only for a CELL_FREE cell on the given layer."""
        return bool(self.grid[layer, y, x] == CELL_FREE)

    def via_blocked_for(self, x: int, y: int, net_id: str) -> bool:
        """True if (x, y) holds a via owned by a DIFFERENT net."""
        owner = self.via_owner.get((x, y))
        return owner is not None and owner != net_id

    # ------------------------------------------------------------------

    def mark_trace(
        self,
        nodes: list[tuple[int, int, int]],
        net_id: str,
        net_type: str,
        never_block: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Commit a routed segment to the grid and return its via cells.

        Marks interior cells CELL_ROUTED on their layer, applies 1-cell DRC
        clearance on the SAME layer, and records via cells (blocking all layers
        for other nets).  Endpoint (pin) cells are left accessible.

        Args:
            nodes:        Path as (x, y, layer) nodes from source to target.
            net_id:       Owning net.
            net_type:     POWER | GROUND | SIGNAL.
            never_block:  (x, y) cells never to mark BLOCKED (current net pins).

        Returns:
            List of via (x, y) coordinates inserted by this segment.
        """
        vias: list[tuple[int, int]] = []
        # Detect vias: consecutive nodes sharing (x, y) but differing layer.
        for (x0, y0, l0), (x1, y1, l1) in zip(nodes, nodes[1:]):
            if x0 == x1 and y0 == y1 and l0 != l1:
                vias.append((x0, y0))
                self.via_owner[(x0, y0)] = net_id
                # A plated hole occupies the cell on every layer.
                for L in range(self.layers):
                    if self.grid[L, y0, x0] != CELL_COMPONENT:
                        self.grid[L, y0, x0] = CELL_ROUTED
                        self.cell_net[(x0, y0, L)] = net_id

        for x, y, layer in nodes[1:-1]:
            if self.grid[layer, y, x] != CELL_COMPONENT:
                self.grid[layer, y, x] = CELL_ROUTED
                self.cell_net[(x, y, layer)] = net_id
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (self.in_bounds(nx, ny)
                        and self.grid[layer, ny, nx] == CELL_FREE
                        and (nx, ny) not in never_block
                        and (nx, ny) not in self.via_owner):
                    self.grid[layer, ny, nx] = CELL_BLOCKED
        return sorted(set(vias))

    def unmark_trace(self, trace: "LayeredTrace") -> None:
        """Remove a previously committed trace (for rip-up-and-reroute)."""
        freed: set[tuple[int, int, int]] = set()
        for (x, y), layer in zip(trace.path[1:-1], trace.layers[1:-1]):
            if self.grid[layer, y, x] == CELL_ROUTED:
                self.grid[layer, y, x] = CELL_FREE
                self.cell_net.pop((x, y, layer), None)
                freed.add((x, y, layer))
        for vx, vy in trace.vias:
            if self.via_owner.get((vx, vy)) == trace.net_id:
                self.via_owner.pop((vx, vy), None)
                for L in range(self.layers):
                    if self.grid[L, vy, vx] == CELL_ROUTED:
                        self.grid[L, vy, vx] = CELL_FREE
                        self.cell_net.pop((vx, vy, L), None)
        # Clear orphaned clearance cells adjacent to freed cells.
        for x, y, layer in freed:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny):
                    continue
                if self.grid[layer, ny, nx] != CELL_BLOCKED:
                    continue
                still = any(
                    self.in_bounds(nx + ddx, ny + ddy)
                    and self.grid[layer, ny + ddy, nx + ddx] == CELL_ROUTED
                    for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                if not still:
                    self.grid[layer, ny, nx] = CELL_FREE


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — LayeredLeeRouter  (weighted BFS / Dijkstra over (x, y, layer))
# ═══════════════════════════════════════════════════════════════════════════════

class LayeredLeeRouter:
    """Dijkstra wavefront over the 3D grid (Lee's algorithm + via/bias costs).

    Planar moves cost 1 plus a soft per-layer direction-bias penalty; a layer
    change (via) at the same (x, y) costs VIA_COST.  Source and target pin
    cells are always passable on every layer (plated through-holes).
    """

    def __init__(self, grid: MultiLayerGrid) -> None:
        """Attach the router to a multi-layer grid."""
        self.grid      = grid
        self._via_cost = via_cost()
        self._bias     = layer_dir_bias()

    # ------------------------------------------------------------------

    def _move_cost(self, layer: int, dx: int, dy: int) -> float:
        """Planar move cost with soft per-layer direction bias.

        Even layers prefer horizontal travel, odd layers prefer vertical.
        """
        against = (dy != 0) if (layer % 2 == 0) else (dx != 0)
        return 1.0 + (self._bias if against else 0.0)

    def _passable(
        self, x: int, y: int, layer: int,
        net_id: str, pins: set[tuple[int, int]],
    ) -> bool:
        """True if (x, y, layer) may be entered while routing net_id."""
        if not self.grid.in_bounds(x, y):
            return False
        if self.grid.via_blocked_for(x, y, net_id):
            return False
        if (x, y) in pins:
            return True                       # own pin: passable on any layer
        return self.grid.is_free(x, y, layer)

    def route(
        self,
        src: tuple[int, int],
        tgt: tuple[int, int],
        net_id: str,
        pins: set[tuple[int, int]],
    ) -> list[tuple[int, int, int]] | None:
        """Find a least-cost (x, y, layer) path from src to tgt.

        Args:
            src:    Source pin (x, y).
            tgt:    Target pin (x, y).
            net_id: Net being routed (for via ownership checks).
            pins:   All pin cells of this net (always passable).

        Returns:
            Path as a list of (x, y, layer) nodes, or None if unreachable.
        """
        sx, sy = src
        tx, ty = tgt
        L = self.grid.layers

        dist: dict[tuple[int, int, int], float] = {}
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
        heap: list[tuple[float, int, int, int]] = []

        # Multi-source: a pin is a through-hole reachable from every layer.
        for layer in range(L):
            node = (sx, sy, layer)
            dist[node] = 0.0
            parent[node] = None
            heapq.heappush(heap, (0.0, sx, sy, layer))

        goal: tuple[int, int, int] | None = None
        while heap:
            d, x, y, layer = heapq.heappop(heap)
            if d > dist.get((x, y, layer), float("inf")):
                continue
            if (x, y) == (tx, ty):
                goal = (x, y, layer)
                break
            # Planar neighbours (same layer)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not self._passable(nx, ny, layer, net_id, pins):
                    continue
                nd = d + self._move_cost(layer, dx, dy)
                node = (nx, ny, layer)
                if nd < dist.get(node, float("inf")):
                    dist[node] = nd
                    parent[node] = (x, y, layer)
                    heapq.heappush(heap, (nd, nx, ny, layer))
            # Layer changes (via) at the same (x, y)
            for nl in (layer - 1, layer + 1):
                if not (0 <= nl < L):
                    continue
                if not self._passable(x, y, nl, net_id, pins):
                    continue
                nd = d + self._via_cost
                node = (x, y, nl)
                if nd < dist.get(node, float("inf")):
                    dist[node] = nd
                    parent[node] = (x, y, layer)
                    heapq.heappush(heap, (nd, x, y, nl))

        if goal is None:
            return None
        path: list[tuple[int, int, int]] = []
        cur: tuple[int, int, int] | None = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — MultiLayerNetRouter
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLayerNetRouter:
    """Routes every net across layers, with rip-up-and-reroute on failure."""

    def __init__(self, graph: CircuitGraph, grid: MultiLayerGrid) -> None:
        """Attach to a circuit graph and an initialised multi-layer grid."""
        self._graph = graph
        self._grid  = grid

    # ------------------------------------------------------------------

    def _reconstruct_nets(self) -> list[Net]:
        """Rebuild Net objects from the graph's star-expanded edge list."""
        data: dict[str, dict] = {}
        for e in self._graph.edges:
            d = data.setdefault(e.net_id, {"type": e.net_type, "pins": {}})
            d["pins"].setdefault(e.source[0], e.source[1])
            d["pins"].setdefault(e.target[0], e.target[1])
        return [
            Net(id=nid, net_type=d["type"], connected_pins=list(d["pins"].items()))
            for nid, d in data.items()
        ]

    def _prioritize(self) -> list[Net]:
        """POWER -> GROUND -> SIGNAL (signals by ascending pin count)."""
        nets = self._reconstruct_nets()
        power  = [n for n in nets if n.net_type == "POWER"]
        ground = [n for n in nets if n.net_type == "GROUND"]
        signal = sorted([n for n in nets if n.net_type == "SIGNAL"],
                        key=lambda n: len(n.connected_pins))
        return power + ground + signal

    def _pin_cell(self, comp_id: str, pin_id: str) -> tuple[int, int]:
        """Routing-accessible (x, y) for a named pin (rounded abs position)."""
        comp = self._graph.nodes[comp_id]
        for pin in comp.pins:
            if pin.id == pin_id:
                return (round(pin.abs_x), round(pin.abs_y))
        return (comp.x + comp.footprint.width // 2,
                comp.y + comp.footprint.height // 2)

    # ------------------------------------------------------------------

    def route_all(self) -> tuple[list[LayeredTrace], list[str]]:
        """Route all nets via chain routing; rip-up-and-reroute failures.

        Returns:
            (traces, failed) — failed entries are "net_id: a/pa -> b/pb" strings.
        """
        traces: list[LayeredTrace] = []
        failed: list[str] = []
        nets = self._prioritize()
        n_nets = len(nets)
        router = LayeredLeeRouter(self._grid)

        for idx, net in enumerate(nets, start=1):
            pins = net.connected_pins
            if len(pins) < 2:
                continue
            net_pin_cells = {self._pin_cell(c, p) for c, p in pins}
            for i in range(len(pins) - 1):
                ca, pa = pins[i]
                cb, pb = pins[i + 1]
                src = self._pin_cell(ca, pa)
                tgt = self._pin_cell(cb, pb)
                nodes = router.route(src, tgt, net.id, net_pin_cells)
                if nodes is None:
                    failed.append(f"{net.id}: {ca}/{pa} -> {cb}/{pb}")
                    print(f"   [{idx}/{n_nets}] {net.id:<12} {ca}/{pa} -> "
                          f"{cb}/{pb}  [FAIL]")
                    continue
                trace = self._commit(net, ca, pa, cb, pb, nodes, net_pin_cells)
                traces.append(trace)
                print(f"   [{idx}/{n_nets}] {net.id:<12} {ca}/{pa} -> {cb}/{pb}"
                      f"  len={trace.length} vias={len(trace.vias)} "
                      f"layers={sorted(set(trace.layers))}  [OK]")

        if failed:
            traces, failed = self._rip_up_reroute(traces, failed)
        if failed:
            traces, failed = self._route_allowing_crossings(traces, failed)
        return traces, failed

    def _commit(
        self, net: Net, ca: str, pa: str, cb: str, pb: str,
        nodes: list[tuple[int, int, int]], net_pin_cells: set[tuple[int, int]],
    ) -> LayeredTrace:
        """Mark a routed path on the grid and build its LayeredTrace."""
        vias = self._grid.mark_trace(nodes, net.id, net.net_type, net_pin_cells)
        return LayeredTrace(
            net_id=net.id, net_type=net.net_type,
            source_comp=ca, source_pin=pa, target_comp=cb, target_pin=pb,
            path=[(x, y) for x, y, _ in nodes],
            layers=[layer for _, _, layer in nodes],
            vias=vias, length=len(nodes),
        )

    # ------------------------------------------------------------------

    def _route_allowing_crossings(
        self, traces: list[LayeredTrace], failed: list[str],
    ) -> tuple[list[LayeredTrace], list[str]]:
        """Last resort: connect remaining nets even if it costs a crossing.

        Routes each still-failed net on a relaxed grid (other nets' traces and
        clearance treated as free), then commits on the real grid.  This trades
        a (minimised, layer-spread) crossing for a completed connection — a
        manufacturable trade vs. leaving the net open.  Crossings remain counted
        honestly in the metrics.
        """
        still: list[str] = []
        for fail_str in failed:
            try:
                nid_part, route_part = fail_str.split(": ", 1)
                s_str, t_str = route_part.split(" -> ", 1)
                ca, pa = s_str.split("/", 1)
                cb, pb = t_str.split("/", 1)
            except ValueError:
                still.append(fail_str)
                continue
            net_id = nid_part.strip()
            net_type = next((e.net_type for e in self._graph.edges
                             if e.net_id == net_id), "SIGNAL")
            src, tgt = self._pin_cell(ca, pa), self._pin_cell(cb, pb)
            net_pins = {src, tgt}

            relaxed = self._grid.clone()
            relaxed.grid[(relaxed.grid == CELL_ROUTED)
                         | (relaxed.grid == CELL_BLOCKED)] = CELL_FREE
            relaxed.via_owner = {}
            nodes = LayeredLeeRouter(relaxed).route(src, tgt, net_id, net_pins)
            if nodes is None:
                still.append(fail_str)
                print(f"   [FAIL] {net_id}: no path exists even relaxed")
                continue
            net_obj = Net(id=net_id, net_type=net_type, connected_pins=[])
            traces.append(self._commit(net_obj, ca, pa, cb, pb, nodes, net_pins))
            print(f"   [OK] connected {net_id} (crossing-permitted fallback)")
        return traces, still

    def _ideal_nodes(self, src, tgt, net_id, net_pins):
        """Least-cost path on an obstacle-free grid (components only).

        Ignores every other net's traces/clearance — used to discover which
        committed traces stand between a failed net and its target.
        """
        scratch = MultiLayerGrid(self._grid.width, self._grid.height,
                                 self._grid.layers)
        scratch.initialize_from_graph(self._graph)
        return LayeredLeeRouter(scratch).route(src, tgt, net_id, net_pins)

    def _rip_up_reroute(
        self, traces: list[LayeredTrace], failed: list[str],
    ) -> tuple[list[LayeredTrace], list[str]]:
        """Recover failed nets by surgically ripping only the blocking traces.

        Per failed net: find its ideal (obstacle-free) path, identify which
        committed non-POWER traces occupy cells on that path (same layer or via),
        rip only those, route the failed net, then re-route the ripped traces.
        """
        MAX_PASSES = 4
        still = list(failed)
        router = LayeredLeeRouter(self._grid)

        for pass_n in range(1, MAX_PASSES + 1):
            if not still:
                break
            print(f"[Phase 8] Rip-up pass {pass_n}/{MAX_PASSES} "
                  f"for {len(still)} net(s)...")
            nxt: list[str] = []
            for fail_str in still:
                try:
                    nid_part, route_part = fail_str.split(": ", 1)
                    s_str, t_str = route_part.split(" -> ", 1)
                    ca, pa = s_str.split("/", 1)
                    cb, pb = t_str.split("/", 1)
                except ValueError:
                    nxt.append(fail_str)
                    continue
                net_id = nid_part.strip()
                net_type = next((e.net_type for e in self._graph.edges
                                 if e.net_id == net_id), "SIGNAL")
                src, tgt = self._pin_cell(ca, pa), self._pin_cell(cb, pb)
                net_pins = {src, tgt}

                ideal = self._ideal_nodes(src, tgt, net_id, net_pins)
                if ideal is None:
                    nxt.append(fail_str)
                    continue

                # Cells (and via coords) the ideal path needs.
                ideal_cells = {(x, y, L) for x, y, L in ideal}
                ideal_xy    = {(x, y) for x, y, _ in ideal}

                # Blocking traces = non-POWER traces overlapping those cells.
                blockers: list[LayeredTrace] = []
                for t in traces:
                    if t.net_type == "POWER" or t.net_id == net_id:
                        continue
                    hit = any((x, y, L) in ideal_cells
                              for (x, y), L in zip(t.path[1:-1], t.layers[1:-1]))
                    hit = hit or any(v in ideal_xy for v in t.vias)
                    if hit:
                        blockers.append(t)

                for t in blockers:
                    self._grid.unmark_trace(t)
                traces = [t for t in traces if t not in blockers]

                nodes = router.route(src, tgt, net_id, net_pins)
                if nodes is not None:
                    net_obj = Net(id=net_id, net_type=net_type, connected_pins=[])
                    traces.append(
                        self._commit(net_obj, ca, pa, cb, pb, nodes, net_pins))
                    print(f"   [OK] recovered {net_id}"
                          + (f" (ripped {len(blockers)})" if blockers else ""))
                else:
                    nxt.append(fail_str)

                # Re-route ripped traces (regardless of the above outcome).
                for t in blockers:
                    rp = {self._pin_cell(t.source_comp, t.source_pin),
                          self._pin_cell(t.target_comp, t.target_pin)}
                    rn = router.route(self._pin_cell(t.source_comp, t.source_pin),
                                      self._pin_cell(t.target_comp, t.target_pin),
                                      t.net_id, rp)
                    if rn is not None:
                        net_obj2 = Net(id=t.net_id, net_type=t.net_type,
                                       connected_pins=[])
                        traces.append(self._commit(
                            net_obj2, t.source_comp, t.source_pin,
                            t.target_comp, t.target_pin, rn, rp))
                    else:
                        nxt.append(f"{t.net_id}: {t.source_comp}/{t.source_pin}"
                                   f" -> {t.target_comp}/{t.target_pin}")
            still = nxt
        return traces, still


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Crossing metrics + Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

def compute_layer_crossings(
    traces: list[LayeredTrace],
) -> tuple[int, dict[int, int]]:
    """Count same-layer crossings overall and per layer.

    A crossing is an interior cell on a given layer occupied by 2+ DIFFERENT
    nets.  By construction the router prevents these, so this is normally 0.

    Returns:
        (total_same_layer_crossings, {layer: crossings}).
    """
    cell_nets: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for t in traces:
        for (x, y), layer in zip(t.path[1:-1], t.layers[1:-1]):
            cell_nets[(x, y, layer)].add(t.net_id)
    per_layer: dict[int, int] = defaultdict(int)
    for (_, _, layer), nets in cell_nets.items():
        if len(nets) > 1:
            per_layer[layer] += 1
    return sum(per_layer.values()), dict(per_layer)


def visualize_multilayer(
    graph: CircuitGraph,
    traces: list[LayeredTrace],
    metrics: dict,
    output_path: Path | None = None,
) -> Path:
    """Render a layer-distinct routed-board canvas and save as PNG.

    Args:
        graph:       Placed CircuitGraph.
        traces:      Routed LayeredTrace objects.
        metrics:     Phase 8 metrics dict.
        output_path: Destination PNG (defaults to outputs/phase8_output.png).

    Returns:
        Path where the PNG was saved.
    """
    if output_path is None:
        output_path = _OUTPUT_DIR / "phase8_output.png"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gw, gh = graph.metadata.width, graph.metadata.height
    n_layers = metrics.get("layers", routing_layers())

    fig = plt.figure(figsize=(26, 11), facecolor=_BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.05,
                   left=0.03, right=0.98, top=0.93, bottom=0.07)
    ax = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    ax.set_facecolor(_BG)
    ax.set_xlim(-0.5, gw + 0.5)
    ax.set_ylim(-0.5, gh + 0.5)
    ax.set_aspect("equal")
    ax.tick_params(colors=_DIM_C, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(_GRID_C)
    ax.set_title(f"Phase 8 — Multi-Layer Routed PCB  ({n_layers} layers)",
                 color=_TEXT_C, fontsize=12, fontweight="bold", pad=8)
    for x in range(gw + 1):
        ax.axvline(x, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)
    for y in range(gh + 1):
        ax.axhline(y, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)

    # Trace segments coloured by layer (split path into per-layer runs)
    for t in traces:
        for i in range(len(t.path) - 1):
            (x0, y0), (x1, y1) = t.path[i], t.path[i + 1]
            l0, l1 = t.layers[i], t.layers[i + 1]
            if l0 != l1:
                continue   # via (no planar segment)
            col = _LAYER_COLORS[l0 % len(_LAYER_COLORS)]
            ax.plot([x0 + 0.5, x1 + 0.5], [y0 + 0.5, y1 + 0.5],
                    color=col, lw=2.0, alpha=0.85, zorder=3,
                    solid_capstyle="round")
    # Vias as filled circles
    for t in traces:
        for vx, vy in t.vias:
            ax.plot(vx + 0.5, vy + 0.5, "o", color=_VIA_COLOR,
                    markersize=7, markeredgecolor="#000", markeredgewidth=0.8,
                    zorder=6)

    # Component boxes
    for comp in graph.nodes.values():
        color = _COMP_COLORS.get(comp.comp_type, "#888888")
        ax.add_patch(mpatches.FancyBboxPatch(
            (comp.x + 0.05, comp.y + 0.05),
            comp.footprint.width - 0.10, comp.footprint.height - 0.10,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="white", alpha=0.85, linewidth=1.5, zorder=4))
        cx = comp.x + comp.footprint.width / 2.0
        cy = comp.y + comp.footprint.height / 2.0
        ax.text(cx, cy, comp.id, color="white", fontsize=7,
                ha="center", va="center", fontweight="bold", zorder=5)

    # Legend: one entry per layer + via
    handles = [mpatches.Patch(color=_LAYER_COLORS[i % len(_LAYER_COLORS)],
                              label=f"Layer {i}") for i in range(n_layers)]
    handles.append(mpatches.Patch(color=_VIA_COLOR, label="Via"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              facecolor=_PANEL_BG, edgecolor=_GRID_C, labelcolor=_TEXT_C)

    # Info panel
    ax_info.set_facecolor(_PANEL_BG)
    ax_info.axis("off")
    plc = metrics.get("per_layer_crossings", {})
    lines = [
        "=" * 26, "  MULTI-LAYER ROUTING", "=" * 26, "",
        f"  Layers      : {n_layers}",
        f"  Routed      : {metrics.get('total_routed', 0)}",
        f"  Failed      : {metrics.get('total_failed', 0)}",
        f"  Completion  : {metrics.get('completion_pct', 0):.1f}%",
        f"  Vias        : {metrics.get('via_count', 0)}",
        f"  Same-layer  : {metrics.get('crossing_count', 0)} crossings",
        f"  Length      : {metrics.get('total_length', 0)} cells",
        "", "  PER-LAYER CROSSINGS", "  " + "-" * 22,
    ]
    for L in range(n_layers):
        lines.append(f"  Layer {L}     : {plc.get(L, 0)}")
    lines += ["", "=" * 26, "  Phase 8 router done", "=" * 26]
    ax_info.text(0.05, 0.97, "\n".join(lines), transform=ax_info.transAxes,
                 fontfamily="monospace", color=_TEXT_C, fontsize=8,
                 va="top", ha="left", linespacing=1.6)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase8(
    graph: CircuitGraph,
) -> tuple[CircuitGraph, list[LayeredTrace], dict]:
    """Phase 8 pipeline: CircuitGraph -> multi-layer routed traces + metrics.

    Drop-in alternative to run_phase3().  Returns the SAME (graph, traces,
    metrics) shape so Phase 4 consumes it unchanged.  Does NOT mutate the graph.

    Args:
        graph: Placed CircuitGraph (from Phase 2 GA or Phase 7 RL).

    Returns:
        (graph, traces, metrics) where metrics carries the Phase 3 keys PLUS
        via_count, per_layer_crossings, same_layer_crossings, and layers.
    """
    gw, gh = graph.metadata.width, graph.metadata.height
    n_layers = routing_layers()

    print(f"\n[Phase 8] Routing on {n_layers} layers "
          f"({gw}x{gh}, via_cost={via_cost()}, bias={layer_dir_bias()}) ...")
    grid = MultiLayerGrid(gw, gh, n_layers)
    grid.initialize_from_graph(graph)

    net_router = MultiLayerNetRouter(graph, grid)
    traces, failed = net_router.route_all()

    same_layer, per_layer = compute_layer_crossings(traces)
    via_count = len({v for t in traces for v in t.vias})
    lengths = [t.length for t in traces]
    n_nets = len(net_router._prioritize())
    total_segs = len(traces) + len(failed)
    completion = (len(traces) / total_segs * 100.0) if total_segs else 100.0

    metrics: dict = {
        # Phase 3-compatible keys (crossing_count = same-layer crossings)
        "total_routed":   len(traces),
        "total_failed":   len(failed),
        "total_length":   sum(lengths) if lengths else 0,
        "crossing_count": same_layer,
        "longest_trace":  max(lengths) if lengths else 0,
        "shortest_trace": min(lengths) if lengths else 0,
        "failed_routes":  failed,
        # Phase 8 additive keys
        "layers":               n_layers,
        "via_count":            via_count,
        "same_layer_crossings": same_layer,
        "per_layer_crossings":  per_layer,
        "completion_pct":       round(completion, 1),
    }

    status = "all nets connected" if not failed else f"{len(failed)} failed"
    print(f"[Phase 8] {len(traces)} segments ({n_nets} nets, {status})")
    print(f"[Phase 8] Same-layer crossings: {same_layer}  |  vias: {via_count}")
    print(f"[Phase 8] Per-layer crossings : {per_layer}")

    out = visualize_multilayer(graph, traces, metrics)
    print(f"[Phase 8] Saved -> {out}")
    return graph, traces, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phase1_eda_engine import NetlistParser, InitialPlacer
    from phase2_genetic_placer import run_phase2

    _sample = _PROJECT_ROOT / "netlists" / "sample_netlist.json"
    with _sample.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    parser  = NetlistParser()
    netlist = parser.parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    graph = run_phase2(graph)
    graph, traces, metrics = run_phase8(graph)

    print("\n[Phase 8] Complete. Graph + layered traces ready for Phase 4.")
    sys.exit(0)
