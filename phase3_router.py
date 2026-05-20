"""
Phase 3 — Lee's Algorithm Maze Router

Sections
--------
1. Constants & colour palette
2. RoutedTrace dataclass
3. RoutingGrid   — 2D obstacle + trace grid
4. LeeRouter     — BFS wavefront expansion + backtrace
5. NetRouter     — net prioritisation + chain routing
6. Visualizer    — three-panel dark-mode canvas
7. run_phase3()  — pipeline entry-point
8. CLI entry-point
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import numpy as np

from phase1_eda_engine import CircuitGraph, Component, Net, GridMetadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Grid cell state constants
# ---------------------------------------------------------------------------
CELL_FREE      = 0
CELL_COMPONENT = 1
CELL_ROUTED    = 2
CELL_BLOCKED   = 3
MIN_TRACE_SEP  = 1   # 1-cell DRC clearance between traces

# ---------------------------------------------------------------------------
# Colour palette — dark-mode aesthetic
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
_TRACE_COLORS: dict[str, str] = {
    "POWER":  "#FFD700",
    "GROUND": "#888888",
    "SIGNAL": "#00C97A",
}
_TRACE_WIDTHS: dict[str, float] = {
    "POWER":  2.5,
    "GROUND": 2.0,
    "SIGNAL": 1.5,
}
_HEATMAP_COLORS = ["#0f0f1a", "#1E90FF", "#00C97A", "#FF4444"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — RoutedTrace
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoutedTrace:
    """One successfully routed electrical connection (a copper trace segment).

    Attributes:
        net_id:      Net this trace belongs to.
        net_type:    POWER | SIGNAL | GROUND.
        source_comp: Component ID of source pin.
        source_pin:  Pin ID of source.
        target_comp: Component ID of target pin.
        target_pin:  Pin ID of target.
        path:        Ordered (x, y) grid cells from source to target.
        length:      Number of cells in path (includes endpoints).
    """

    net_id: str
    net_type: str
    source_comp: str
    source_pin: str
    target_comp: str
    target_pin: str
    path: list[tuple[int, int]]
    length: int


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — RoutingGrid
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingGrid:
    """2D integer grid tracking obstacle and trace states for the maze router.

    Grid is indexed as grid[y, x] (numpy row-major).
    y = 0 is the bottom row (origin at lower-left, matching matplotlib default).

    Cell states:
        CELL_FREE (0)      — routable empty space
        CELL_COMPONENT (1) — occupied by a component footprint
        CELL_ROUTED (2)    — occupied by a placed copper trace
        CELL_BLOCKED (3)   — DRC clearance zone around a routed trace
    """

    def __init__(self, width: int, height: int) -> None:
        """Create an empty routing grid.

        Args:
            width:  Number of grid columns.
            height: Number of grid rows.
        """
        self._width     = width
        self._height    = height
        self.grid       = np.zeros((height, width), dtype=int)
        self._pin_cells: set[tuple[int, int]] = set()
        # Maps (x, y) -> net_id for cells that are CELL_ROUTED
        self._cell_net: dict[tuple[int, int], str] = {}
        # Maps (x, y) -> net_type for cells that are CELL_ROUTED
        self._cell_net_type: dict[tuple[int, int], str] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Grid width in cells."""
        return self._width

    @property
    def height(self) -> int:
        """Grid height in cells."""
        return self._height

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_from_graph(self, graph: CircuitGraph) -> None:
        """Mark component footprints as CELL_COMPONENT and collect all pin cells.

        Pin cells are never blocked by DRC clearance so that future chain
        routing segments can still access them.

        Args:
            graph: CircuitGraph with Component.x/.y set by Phase 2.
        """
        self.grid[:] = CELL_FREE
        self._pin_cells.clear()

        for comp in graph.nodes.values():
            for dy in range(comp.footprint.height):
                for dx in range(comp.footprint.width):
                    cx, cy = comp.x + dx, comp.y + dy
                    if self.in_bounds(cx, cy):
                        self.grid[cy, cx] = CELL_COMPONENT

            for pin in comp.pins:
                px, py = round(pin.abs_x), round(pin.abs_y)
                if self.in_bounds(px, py):
                    self._pin_cells.add((px, py))

    # ------------------------------------------------------------------
    # Trace marking
    # ------------------------------------------------------------------

    def mark_trace(
        self,
        path: list[tuple[int, int]],
        extra_free: set[tuple[int, int]] | None = None,
        net_id: str | None = None,
        net_type: str | None = None,
    ) -> None:
        """Mark intermediate trace cells CELL_ROUTED and apply 1-cell DRC clearance.

        Skips the first and last cells (pin endpoint cells inside components).
        Never marks any cell in _pin_cells or extra_free as CELL_BLOCKED so
        that subsequent chain segments can still reach their start pins.
        Does not mark already-ROUTED cells as BLOCKED (preserves earlier traces).

        Args:
            path:       Full path from source to target (LeeRouter output).
            extra_free: Additional cells never to block (current net's pin cells).
            net_id:     Net identifier for cell ownership tracking.
            net_type:   POWER | SIGNAL | GROUND for cell type tracking.
        """
        if len(path) <= 2:
            return

        never_block = self._pin_cells | (extra_free or set())

        for x, y in path[1:-1]:
            if self.grid[y, x] != CELL_COMPONENT:
                self.grid[y, x] = CELL_ROUTED
                if net_id is not None:
                    self._cell_net[(x, y)] = net_id
                if net_type is not None:
                    self._cell_net_type[(x, y)] = net_type

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (
                    self.in_bounds(nx, ny)
                    and self.grid[ny, nx] == CELL_FREE
                    and (nx, ny) not in never_block
                ):
                    self.grid[ny, nx] = CELL_BLOCKED

    def unmark_trace(self, path: list[tuple[int, int]]) -> None:
        """Restore intermediate path cells from CELL_ROUTED back to CELL_FREE.

        Also clears adjacent BLOCKED (DRC clearance) cells so that rerouting
        can find alternative paths through the freed area.

        Args:
            path: The path of the trace to remove (from RoutedTrace.path).
        """
        freed: set[tuple[int, int]] = set()
        for x, y in path[1:-1]:
            if self.grid[y, x] == CELL_ROUTED:
                self.grid[y, x] = CELL_FREE
                self._cell_net.pop((x, y), None)
                self._cell_net_type.pop((x, y), None)
                freed.add((x, y))
        # Clear adjacent BLOCKED cells that were DRC clearance for this trace.
        # We clear any BLOCKED cell adjacent to a freed cell that is not also
        # adjacent to a still-ROUTED cell (to avoid releasing clearance of other traces).
        for x, y in freed:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny):
                    continue
                if self.grid[ny, nx] != CELL_BLOCKED:
                    continue
                # Only clear if no adjacent ROUTED cell remains
                still_needed = any(
                    self.in_bounds(nx + ddx, ny + ddy)
                    and self.grid[ny + ddy, nx + ddx] == CELL_ROUTED
                    for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                if not still_needed:
                    self.grid[ny, nx] = CELL_FREE

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_free(self, x: int, y: int) -> bool:
        """Return True only for CELL_FREE cells."""
        return bool(self.grid[y, x] == CELL_FREE)

    def in_bounds(self, x: int, y: int) -> bool:
        """Return True if (x, y) lies within grid dimensions."""
        return 0 <= x < self._width and 0 <= y < self._height

    def get_free_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return 4-connected free neighbors (Manhattan movement only).

        Args:
            x: Column index.
            y: Row index.

        Returns:
            List of (x, y) pairs for CELL_FREE neighbouring cells.
        """
        result = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.is_free(nx, ny):
                result.append((nx, ny))
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clone(self) -> "RoutingGrid":
        """Return an independent deep copy of this grid.

        Used by route_with_detour to explore relaxed DRC without mutating
        the authoritative grid.

        Returns:
            New RoutingGrid with identical cell states and pin-cell registry.
        """
        c = RoutingGrid(self._width, self._height)
        c.grid = self.grid.copy()
        c._pin_cells      = set(self._pin_cells)
        c._cell_net       = dict(self._cell_net)
        c._cell_net_type  = dict(self._cell_net_type)
        return c

    def component_cell_count(self) -> int:
        """Return the number of cells currently marked CELL_COMPONENT."""
        return int(np.sum(self.grid == CELL_COMPONENT))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — LeeRouter (BFS maze router)
# ═══════════════════════════════════════════════════════════════════════════════

class LeeRouter:
    """Lee's Algorithm BFS maze router (Strategy pattern — swappable with A*).

    Pure BFS wavefront expansion; no heuristic is applied.
    An A* upgrade stub is included as a comment below.
    """

    def __init__(self, routing_grid: RoutingGrid) -> None:
        """Attach the router to a routing grid.

        Args:
            routing_grid: Grid representing the current board state.
        """
        self.grid = routing_grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self, sx: int, sy: int, tx: int, ty: int
    ) -> list[tuple[int, int]] | None:
        """Route from (sx, sy) to (tx, ty) using BFS wavefront expansion.

        Phase 1 — Expansion: BFS from source, tracking each cell's parent.
        Phase 2 — Backtrace: follow parent pointers from target to source.

        Source and target are always treated as passable regardless of their
        cell state (they may be CELL_COMPONENT pin locations inside footprints).

        Args:
            sx, sy: Source grid coordinates.
            tx, ty: Target grid coordinates.

        Returns:
            Ordered path [(sx,sy), ..., (tx,ty)], or None if unreachable.
        """
        if sx == tx and sy == ty:
            return [(sx, sy)]

        parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        found = False

        while queue and not found:
            x, y = queue.popleft()
            for nx, ny in self._passable_neighbors(x, y, sx, sy, tx, ty):
                if (nx, ny) not in parent:
                    parent[(nx, ny)] = (x, y)
                    if nx == tx and ny == ty:
                        found = True
                        break
                    queue.append((nx, ny))

        if (tx, ty) not in parent:
            return None

        # Backtrace — follow parent pointers from target back to source
        path: list[tuple[int, int]] = []
        curr: tuple[int, int] | None = (tx, ty)
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        return list(reversed(path))

    def route_with_detour(
        self,
        sx: int, sy: int,
        tx: int, ty: int,
        max_attempts: int = 3,
    ) -> list[tuple[int, int]] | None:
        """Try normal routing; fall back to progressively relaxed detour on failure.

        Three escalating attempts:
          1. Normal: respect all DRC rules.
          2. DRC-relaxed: BLOCKED cells treated as FREE (ignore clearance).
          3. Full-fallback: BLOCKED + ROUTED cells treated as FREE (allow crossing).

        The authoritative grid is never mutated during detour attempts.

        Args:
            sx, sy:       Source coordinates.
            tx, ty:       Target coordinates.
            max_attempts: Reserved for API compatibility (always 3 levels).

        Returns:
            Shortest valid (or best-effort) path, or None if truly unreachable.
        """
        path = self.route(sx, sy, tx, ty)
        if path is not None:
            return path

        # Level 2 — relax BLOCKED cells (ignore DRC clearance)
        r2 = self.grid.clone()
        r2.grid[r2.grid == CELL_BLOCKED] = CELL_FREE
        path = LeeRouter(r2).route(sx, sy, tx, ty)
        if path is not None:
            return path

        # Level 3 — relax BLOCKED + ROUTED (allow crossing, absolute fallback)
        r3 = self.grid.clone()
        r3.grid[r3.grid == CELL_BLOCKED] = CELL_FREE
        r3.grid[r3.grid == CELL_ROUTED]  = CELL_FREE
        return LeeRouter(r3).route(sx, sy, tx, ty)

    # ------------------------------------------------------------------
    # A* upgrade stub — uncomment and implement to replace BFS
    # ------------------------------------------------------------------
    # def _astar(self, sx: int, sy: int, tx: int, ty: int):
    #     """A* heuristic router using Manhattan distance.
    #     Provides better performance on large grids at the cost of complexity.
    #     Not yet activated — switch route() to call this instead of BFS."""
    #     raise NotImplementedError("A* not yet implemented; using Lee BFS.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passable_neighbors(
        self,
        x: int, y: int,
        sx: int, sy: int,
        tx: int, ty: int,
    ) -> list[tuple[int, int]]:
        """Return 4-connected passable neighbours for BFS expansion.

        Source and target are always passable; all other cells must be CELL_FREE.

        Args:
            x, y:   Current cell being expanded.
            sx, sy: Source cell (always passable).
            tx, ty: Target cell (always passable).

        Returns:
            List of reachable (x, y) neighbour cells.
        """
        result = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if not self.grid.in_bounds(nx, ny):
                continue
            if (nx, ny) == (sx, sy) or (nx, ny) == (tx, ty):
                result.append((nx, ny))
            elif self.grid.is_free(nx, ny):
                result.append((nx, ny))
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — NetRouter
# ═══════════════════════════════════════════════════════════════════════════════

class NetRouter:
    """Orchestrates routing of all nets in EEE-correct priority order.

    Routing order: POWER -> GROUND -> SIGNAL (fewest pins first).
    Uses chain routing (Steiner-tree approximation) for multi-pin nets.
    """

    def __init__(self, graph: CircuitGraph, routing_grid: RoutingGrid) -> None:
        """Attach the router to a circuit graph and routing grid.

        Args:
            graph:        CircuitGraph with optimised component positions.
            routing_grid: Initialised RoutingGrid with component obstacles marked.
        """
        self._graph = graph
        self._grid  = routing_grid

    # ------------------------------------------------------------------
    # Net reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_nets(self) -> list[Net]:
        """Rebuild Net objects from the graph's star-expanded edge list.

        CircuitGraph stores GraphEdge objects, not Net objects.  This method
        groups edges by net_id to recover the full (comp_id, pin_id) membership.

        Returns:
            List of Net objects with complete connected_pins lists.
        """
        net_data: dict[str, dict] = {}
        for edge in self._graph.edges:
            nid = edge.net_id
            if nid not in net_data:
                net_data[nid] = {"type": edge.net_type, "pins": {}}
            if edge.source[0] not in net_data[nid]["pins"]:
                net_data[nid]["pins"][edge.source[0]] = edge.source[1]
            if edge.target[0] not in net_data[nid]["pins"]:
                net_data[nid]["pins"][edge.target[0]] = edge.target[1]

        return [
            Net(
                id=nid,
                net_type=data["type"],
                connected_pins=list(data["pins"].items()),
            )
            for nid, data in net_data.items()
        ]

    # ------------------------------------------------------------------
    # Net prioritisation
    # ------------------------------------------------------------------

    def _prioritize_nets(self) -> list[Net]:
        """Return nets in EEE routing order: POWER -> GROUND -> SIGNAL (ascending pin count).

        Wide power traces must be placed first so signal traces route around them.

        Returns:
            Ordered list of Net objects for the maze router to process.
        """
        nets   = self._reconstruct_nets()
        power  = [n for n in nets if n.net_type == "POWER"]
        ground = [n for n in nets if n.net_type == "GROUND"]
        signal = sorted(
            [n for n in nets if n.net_type == "SIGNAL"],
            key=lambda n: len(n.connected_pins),
        )
        return power + ground + signal

    # ------------------------------------------------------------------
    # Pin cell lookup
    # ------------------------------------------------------------------

    def _get_pin_cell(self, comp_id: str, pin_id: str) -> tuple[int, int]:
        """Return the routing-accessible grid cell (x, y) for a named pin.

        Primary: round(pin.abs_x), round(pin.abs_y).
        Fallback: if all four neighbours of the pin cell are CELL_COMPONENT
        (the component is hemmed in by adjacent components), return the nearest
        accessible cell on the component's footprint boundary instead.

        Args:
            comp_id: Component identifier in graph.nodes.
            pin_id:  Pin identifier within that component.

        Returns:
            Best (x, y) the router can reach to/from for this pin.
        """
        comp = self._graph.nodes[comp_id]
        for pin in comp.pins:
            if pin.id == pin_id:
                px, py = round(pin.abs_x), round(pin.abs_y)
                # If any neighbour is non-COMPONENT the router can exit normally
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = px + dx, py + dy
                    if (self._grid.in_bounds(nx, ny)
                            and self._grid.grid[ny, nx] != CELL_COMPONENT):
                        return (px, py)
                # All neighbours are COMPONENT — find nearest boundary exit
                return self._find_boundary_exit(comp, px, py)
        return (
            comp.x + comp.footprint.width  // 2,
            comp.y + comp.footprint.height // 2,
        )

    def _find_boundary_exit(
        self, comp: Component, pin_x: int, pin_y: int
    ) -> tuple[int, int]:
        """Return the nearest non-COMPONENT cell adjacent to a component's footprint.

        Used when the nominal pin cell is completely surrounded by component
        footprints (over-packed placement).  Searches every cell on the outer
        border of the footprint and returns the closest accessible exit.

        Args:
            comp:          The component whose footprint we scan.
            pin_x, pin_y: Reference pin position for distance ranking.

        Returns:
            (x, y) of the nearest accessible boundary exit, or the original
            pin position if no exit is found (edge case — grid too small).
        """
        candidates: set[tuple[int, int]] = set()
        for dy in range(comp.footprint.height):
            for dx in range(comp.footprint.width):
                cx, cy = comp.x + dx, comp.y + dy
                for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ncx, ncy = cx + ddx, cy + ddy
                    if (self._grid.in_bounds(ncx, ncy)
                            and self._grid.grid[ncy, ncx] != CELL_COMPONENT):
                        candidates.add((ncx, ncy))
        if candidates:
            return min(candidates,
                       key=lambda c: abs(c[0] - pin_x) + abs(c[1] - pin_y))
        return (pin_x, pin_y)

    # ------------------------------------------------------------------
    # Main routing loop
    # ------------------------------------------------------------------

    def route_all(self) -> tuple[list[RoutedTrace], list[str]]:
        """Route all nets using chain (Steiner) routing in priority order.

        For each net with N pins, routes: pin[0]->pin[1], pin[1]->pin[2], ...
        Successful segments are marked on the grid and recorded as RoutedTrace
        objects.  Failed segments are logged as descriptive strings.

        Returns:
            (routed_traces, failed_routes) — failed_routes are strings of the
            form "net_id: comp_a/pin_a -> comp_b/pin_b".
        """
        routed_traces: list[RoutedTrace] = []
        failed_routes: list[str]         = []
        ordered_nets  = self._prioritize_nets()
        n_nets        = len(ordered_nets)

        for net_idx, net in enumerate(ordered_nets, start=1):
            pins = net.connected_pins  # list[(comp_id, pin_id)]
            if len(pins) < 2:
                continue

            net_pin_cells = {
                self._get_pin_cell(cid, pid) for cid, pid in pins
            }
            router = LeeRouter(self._grid)

            for i in range(len(pins) - 1):
                comp_a, pin_a = pins[i]
                comp_b, pin_b = pins[i + 1]
                sx, sy = self._get_pin_cell(comp_a, pin_a)
                tx, ty = self._get_pin_cell(comp_b, pin_b)

                path = router.route_with_detour(sx, sy, tx, ty)

                if path is not None:
                    trace = RoutedTrace(
                        net_id=net.id,
                        net_type=net.net_type,
                        source_comp=comp_a,
                        source_pin=pin_a,
                        target_comp=comp_b,
                        target_pin=pin_b,
                        path=path,
                        length=len(path),
                    )
                    routed_traces.append(trace)
                    self._grid.mark_trace(
                        path,
                        extra_free=net_pin_cells,
                        net_id=net.id,
                        net_type=net.net_type,
                    )
                    print(
                        f"   [{net_idx}/{n_nets}] {net.id:<12} : "
                        f"{comp_a}/{pin_a} --> {comp_b}/{pin_b}  "
                        f"length={len(path)}  [OK]"
                    )
                else:
                    fail_str = f"{net.id}: {comp_a}/{pin_a} -> {comp_b}/{pin_b}"
                    failed_routes.append(fail_str)
                    print(
                        f"   [{net_idx}/{n_nets}] {net.id:<12} : "
                        f"{comp_a}/{pin_a} --> {comp_b}/{pin_b}  [FAIL]"
                    )

        # Post-routing: attempt rip-up-and-reroute for failed nets
        if failed_routes:
            routed_traces, failed_routes = self._rip_up_and_reroute_failed(
                failed_routes, routed_traces
            )

        # Post-routing: eliminate crossings
        routed_traces = self._eliminate_crossings(routed_traces)

        return routed_traces, failed_routes

    # ------------------------------------------------------------------
    # Rip-up-and-reroute
    # ------------------------------------------------------------------

    def _rip_up_and_reroute_failed(
        self,
        failed_routes: list[str],
        routed_traces: list[RoutedTrace],
    ) -> tuple[list[RoutedTrace], list[str]]:
        """Try to recover unrouted nets by ripping up blocking lower-priority traces.

        Runs up to 3 full passes.  In each pass every failed route attempts:
        1. Relax grid (no BLOCKED cells) to find an ideal path.
        2. Identify which SIGNAL/GROUND ROUTED cells the ideal path needs.
        3. Rip those blocking traces.
        4. Route the failed net.
        5. Re-route the ripped traces.

        Args:
            failed_routes: Failure strings from route_all().
            routed_traces: Successfully routed traces so far.

        Returns:
            Updated (routed_traces, still_failed) after recovery attempts.
        """
        MAX_PASSES = 3
        still_failed = list(failed_routes)

        for pass_n in range(1, MAX_PASSES + 1):
            if not still_failed:
                break
            print(f"\n[Rip-up pass {pass_n}/{MAX_PASSES}] Attempting to recover "
                  f"{len(still_failed)} unrouted net(s)...")
            next_failed: list[str] = []

            for fail_str in still_failed:
                # Parse: "net_id: comp_a/pin_a -> comp_b/pin_b"
                try:
                    net_id_part, route_part = fail_str.split(": ", 1)
                    src_str, tgt_str = route_part.split(" -> ", 1)
                    comp_a, pin_a = src_str.split("/", 1)
                    comp_b, pin_b = tgt_str.split("/", 1)
                except ValueError:
                    next_failed.append(fail_str)
                    continue

                net_id = net_id_part.strip()
                net_type = "SIGNAL"
                for edge in self._graph.edges:
                    if edge.net_id == net_id:
                        net_type = edge.net_type
                        break

                sx, sy = self._get_pin_cell(comp_a, pin_a)
                tx, ty = self._get_pin_cell(comp_b, pin_b)

                # Find ideal path ignoring all obstacles
                free_grid = self._grid.clone()
                free_grid.grid[free_grid.grid == CELL_BLOCKED] = CELL_FREE
                free_grid.grid[free_grid.grid == CELL_ROUTED]  = CELL_FREE
                ideal_path = LeeRouter(free_grid).route(sx, sy, tx, ty)
                if ideal_path is None:
                    print(f"  [FAIL] {net_id}: no path exists even on empty grid")
                    next_failed.append(fail_str)
                    continue

                # Find which ROUTED cells block the ideal path, and their traces
                blocking_traces: list[RoutedTrace] = []
                for cell in ideal_path[1:-1]:
                    cx, cy = cell
                    if self._grid.grid[cy, cx] not in (CELL_ROUTED, CELL_BLOCKED):
                        continue
                    for t in routed_traces:
                        if t.net_type == "POWER":
                            continue  # never rip power traces
                        if any(c == cell for c in t.path[1:-1]) and t not in blocking_traces:
                            blocking_traces.append(t)
                            break

                # Rip up blocking SIGNAL/GROUND traces
                ripped: list[RoutedTrace] = []
                for bt in blocking_traces:
                    self._grid.unmark_trace(bt.path)
                    routed_traces = [rt for rt in routed_traces if rt is not bt]
                    ripped.append(bt)

                # Route the failed net
                path = LeeRouter(self._grid).route_with_detour(sx, sy, tx, ty)
                if path is not None:
                    net_pin_cells = {self._get_pin_cell(comp_a, pin_a),
                                     self._get_pin_cell(comp_b, pin_b)}
                    trace = RoutedTrace(
                        net_id=net_id, net_type=net_type,
                        source_comp=comp_a, source_pin=pin_a,
                        target_comp=comp_b, target_pin=pin_b,
                        path=path, length=len(path),
                    )
                    self._grid.mark_trace(path, extra_free=net_pin_cells,
                                         net_id=net_id, net_type=net_type)
                    routed_traces.append(trace)
                    ripped_names = ", ".join(r.net_id for r in ripped)
                    print(f"  [OK] Recovered: {net_id}"
                          + (f" via rip-up of {ripped_names}" if ripped_names else ""))
                else:
                    # Failed even after rip-up — restore ripped traces
                    next_failed.append(fail_str)
                    print(f"  [FAIL] {net_id}: still unroutable after rip-up")

                # Re-route any ripped traces regardless of success
                for rt in ripped:
                    rsx, rsy = self._get_pin_cell(rt.source_comp, rt.source_pin)
                    rtx, rty = self._get_pin_cell(rt.target_comp, rt.target_pin)
                    repath = LeeRouter(self._grid).route_with_detour(rsx, rsy, rtx, rty)
                    if repath is not None:
                        re_trace = RoutedTrace(
                            net_id=rt.net_id, net_type=rt.net_type,
                            source_comp=rt.source_comp, source_pin=rt.source_pin,
                            target_comp=rt.target_comp, target_pin=rt.target_pin,
                            path=repath, length=len(repath),
                        )
                        self._grid.mark_trace(repath, net_id=rt.net_id, net_type=rt.net_type)
                        routed_traces.append(re_trace)
                    else:
                        re_fail = (f"{rt.net_id}: {rt.source_comp}/{rt.source_pin} "
                                   f"-> {rt.target_comp}/{rt.target_pin}")
                        next_failed.append(re_fail)
                        print(f"  [WARN] Could not re-route ripped: {rt.net_id}")

            still_failed = next_failed

        return routed_traces, still_failed

    # ------------------------------------------------------------------
    # Crossing elimination
    # ------------------------------------------------------------------

    def _eliminate_crossings(
        self, traces: list[RoutedTrace]
    ) -> list[RoutedTrace]:
        """Detect and eliminate wire crossings by ripping/rerouting SIGNAL nets.

        A crossing occurs when two traces from different nets share a grid cell.
        POWER and GROUND traces are never ripped.

        Args:
            traces: Full list of routed traces after initial routing.

        Returns:
            Updated trace list with crossings reduced or eliminated.
        """
        MAX_PASSES = 5

        for pass_n in range(1, MAX_PASSES + 1):
            # Exact crossing detection: cell -> set of net_ids
            cell_nets: dict[tuple, set[str]] = defaultdict(set)
            for t in traces:
                for cell in t.path[1:-1]:
                    cell_nets[cell].add(t.net_id)

            crossing_cells = {cell for cell, nets in cell_nets.items() if len(nets) > 1}
            before_count = len(crossing_cells)
            if before_count == 0:
                break

            # Collect ONE SIGNAL trace involved in a crossing (rip one at a time)
            rip_target: RoutedTrace | None = None
            for cell in crossing_cells:
                for t in traces:
                    if t.net_type == "SIGNAL" and t.net_id in cell_nets[cell]:
                        rip_target = t
                        break
                if rip_target is not None:
                    break

            if rip_target is None:
                # Only POWER/GROUND crossings remain — cannot fix without ripping power
                break

            # Rip and reroute this one signal trace
            self._grid.unmark_trace(rip_target.path)
            traces = [t for t in traces if t is not rip_target]

            sx, sy = self._get_pin_cell(rip_target.source_comp, rip_target.source_pin)
            tx_c, ty_c = self._get_pin_cell(rip_target.target_comp, rip_target.target_pin)

            # Level 1: strict (no ROUTED cells)
            path = LeeRouter(self._grid).route(sx, sy, tx_c, ty_c)
            if path is None:
                # Level 2: allow BLOCKED clearance zones
                relaxed = self._grid.clone()
                relaxed.grid[relaxed.grid == CELL_BLOCKED] = CELL_FREE
                path = LeeRouter(relaxed).route(sx, sy, tx_c, ty_c)

            if path is not None:
                net_pin_cells = {(sx, sy), (tx_c, ty_c)}
                re_trace = RoutedTrace(
                    net_id=rip_target.net_id, net_type=rip_target.net_type,
                    source_comp=rip_target.source_comp, source_pin=rip_target.source_pin,
                    target_comp=rip_target.target_comp, target_pin=rip_target.target_pin,
                    path=path, length=len(path),
                )
                self._grid.mark_trace(path, extra_free=net_pin_cells,
                                     net_id=rip_target.net_id, net_type=rip_target.net_type)
                traces.append(re_trace)
            # If truly no crossing-free path exists, leave net unrouted

            # Count crossings after this pass
            cell_nets_after: dict[tuple, set[str]] = defaultdict(set)
            for t in traces:
                for cell in t.path[1:-1]:
                    cell_nets_after[cell].add(t.net_id)
            after_count = sum(1 for nets in cell_nets_after.values() if len(nets) > 1)
            print(f"[Crossing fix pass {pass_n}] Crossings: {before_count} -> {after_count}")

            if after_count == 0:
                break

        return traces


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Visualiser
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_routing(
    graph: CircuitGraph,
    routing_grid: RoutingGrid,
    routed_traces: list[RoutedTrace],
    failed_routes: list[str],
    metrics: dict,
    output_path: Path | None = None,
) -> Path:
    """Render a three-panel dark-mode figure and save as PNG.

    Panels:
      Left   — Routed PCB layout (component boxes + copper trace polylines).
      Middle — RoutingGrid heatmap (cell state colours).
      Right  — Routing analytics text report.

    Args:
        graph:         CircuitGraph with final component positions.
        routing_grid:  RoutingGrid after all traces have been marked.
        routed_traces: List of successfully routed RoutedTrace objects.
        failed_routes: List of failed route descriptor strings.
        metrics:       Metrics dict from run_phase3.
        output_path:   Destination PNG; defaults to outputs/phase3_output.png.

    Returns:
        Path where the PNG was saved.
    """
    if output_path is None:
        output_path = _OUTPUT_DIR / "phase3_output.png"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(28, 11), facecolor=_BG)
    gs  = GridSpec(
        1, 3, figure=fig, width_ratios=[3, 2, 1.5], wspace=0.06,
        left=0.03, right=0.98, top=0.93, bottom=0.07,
    )
    ax_board = fig.add_subplot(gs[0])
    ax_heat  = fig.add_subplot(gs[1])
    ax_info  = fig.add_subplot(gs[2])

    gw = graph.metadata.width
    gh = graph.metadata.height

    # ── Left: Routed board canvas ──────────────────────────────────────
    ax_board.set_facecolor(_BG)
    ax_board.set_xlim(-0.5, gw + 0.5)
    ax_board.set_ylim(-0.5, gh + 0.5)
    ax_board.set_aspect("equal")
    ax_board.tick_params(colors=_DIM_C, labelsize=7)
    for spine in ax_board.spines.values():
        spine.set_color(_GRID_C)
    ax_board.set_title(
        "Phase 3 -- Routed PCB Layout",
        color=_TEXT_C, fontsize=11, pad=8, fontweight="bold",
    )
    ax_board.set_xlabel("x (grid units)", color=_DIM_C, fontsize=8)
    ax_board.set_ylabel("y (grid units)", color=_DIM_C, fontsize=8)

    for x in range(gw + 1):
        ax_board.axvline(x, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)
    for y in range(gh + 1):
        ax_board.axhline(y, color=_GRID_C, lw=0.3, alpha=0.6, zorder=0)

    # Copper traces
    for trace in routed_traces:
        if len(trace.path) < 2:
            continue
        col = _TRACE_COLORS.get(trace.net_type, "#aaaaaa")
        lw  = _TRACE_WIDTHS.get(trace.net_type, 1.5)
        xs  = [p[0] + 0.5 for p in trace.path]
        ys  = [p[1] + 0.5 for p in trace.path]
        ax_board.plot(xs, ys, color=col, lw=lw, alpha=0.9, zorder=3,
                      solid_capstyle="round", solid_joinstyle="round")
        # Via markers at bend points
        if len(xs) > 2:
            ax_board.plot(xs[1:-1], ys[1:-1], "o", color=col,
                          markersize=2.5, zorder=4)

    # Component boxes
    for comp in graph.nodes.values():
        color = _COMP_COLORS.get(comp.comp_type, "#888888")
        ax_board.add_patch(mpatches.FancyBboxPatch(
            (comp.x + 0.05, comp.y + 0.05),
            comp.footprint.width - 0.10, comp.footprint.height - 0.10,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="white", alpha=0.85, linewidth=1.5, zorder=5,
        ))
        cx = comp.x + comp.footprint.width  / 2.0
        cy = comp.y + comp.footprint.height / 2.0
        ax_board.text(cx, cy + 0.2, comp.id, color="white", fontsize=7,
                      ha="center", va="center", fontweight="bold", zorder=6)
        ax_board.text(cx, cy - 0.5, comp.name, color="#ccccdd", fontsize=5,
                      ha="center", va="center", zorder=6)
        for pin in comp.pins:
            px, py = round(pin.abs_x), round(pin.abs_y)
            ax_board.plot(px + 0.5, py + 0.5, "o", color="white",
                          markersize=2, zorder=7, alpha=0.7)

    present = {c.comp_type for c in graph.nodes.values()}
    ax_board.legend(
        handles=[
            mpatches.Patch(color=_COMP_COLORS.get(t, "#888"), label=t)
            for t in sorted(present)
        ],
        loc="upper right", fontsize=7,
        facecolor=_PANEL_BG, edgecolor=_GRID_C, labelcolor=_TEXT_C, framealpha=0.85,
    )

    # ── Middle: RoutingGrid heatmap ────────────────────────────────────
    ax_heat.set_facecolor(_BG)
    for spine in ax_heat.spines.values():
        spine.set_color(_GRID_C)
    ax_heat.tick_params(colors=_DIM_C, labelsize=7)
    ax_heat.set_title("Routing Grid State", color=_TEXT_C,
                       fontsize=10, pad=6, fontweight="bold")
    ax_heat.set_xlabel("x", color=_DIM_C, fontsize=8)
    ax_heat.set_ylabel("y", color=_DIM_C, fontsize=8)

    cmap   = mcolors.ListedColormap(_HEATMAP_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    im     = ax_heat.imshow(
        routing_grid.grid, cmap=cmap, norm=norm,
        origin="lower", aspect="equal", interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax_heat, ticks=[0, 1, 2, 3],
                         fraction=0.046, pad=0.04)
    cbar.set_ticklabels(["FREE", "COMP", "ROUTED", "BLOCKED"])
    cbar.ax.tick_params(colors=_DIM_C, labelsize=6)
    cbar.outline.set_edgecolor(_GRID_C)

    # ── Right: Analytics text panel ────────────────────────────────────
    ax_info.set_facecolor(_PANEL_BG)
    ax_info.axis("off")
    ax_info.set_title("Routing Report", color=_TEXT_C, fontsize=9, pad=6)

    routed_net_ids = {t.net_id for t in routed_traces}
    failed_net_ids = {f.split(":")[0].strip() for f in failed_routes}
    all_net_ids    = sorted(routed_net_ids | failed_net_ids)

    lines = [
        "=" * 27,
        "  ROUTING SUMMARY",
        "=" * 27,
        "",
        f"  Routed   : {metrics['total_routed']:>3}",
        f"  Failed   : {metrics['total_failed']:>3}",
        f"  Length   : {metrics['total_length']:>5} cells",
        f"  Longest  : {metrics['longest_trace']:>5} cells",
        f"  Shortest : {metrics['shortest_trace']:>5} cells",
        f"  Crossings: {metrics['crossing_count']:>5}",
        "",
        "  NET STATUS",
        "  " + "-" * 25,
    ]
    for nid in all_net_ids:
        tag = "[FAIL]" if nid in failed_net_ids else "[OK]  "
        lines.append(f"  {nid:<16} {tag}")

    if failed_routes:
        lines += ["", "  FAILED ROUTES", "  " + "-" * 25]
        for fr in failed_routes:
            lines.append(f"  {fr}")

    lines += [
        "",
        "=" * 27,
        "  Phase 2 --> Phase 3",
        "  [OK] Maze router done",
        "=" * 27,
    ]

    ax_info.text(
        0.05, 0.97, "\n".join(lines),
        transform=ax_info.transAxes,
        fontfamily="monospace", color=_TEXT_C,
        fontsize=6.5, va="top", ha="left", linespacing=1.5,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def _run_routing_attempt(
    graph: CircuitGraph,
    gw: int,
    gh: int,
) -> tuple[list[RoutedTrace], list[str], RoutingGrid]:
    """One complete routing attempt on a grid of given dimensions.

    Args:
        graph: CircuitGraph with GA-optimised component positions.
        gw:    Grid width for this attempt.
        gh:    Grid height for this attempt.

    Returns:
        (routed_traces, failed_routes, routing_grid)
    """
    r_grid = RoutingGrid(gw, gh)
    r_grid.initialize_from_graph(graph)
    net_router = NetRouter(graph, r_grid)
    routed_traces, failed_routes = net_router.route_all()
    return routed_traces, failed_routes, r_grid


def run_phase3(
    graph: CircuitGraph,
) -> tuple[CircuitGraph, list[RoutedTrace], dict]:
    """Phase 3 pipeline: CircuitGraph -> routed graph + traces + metrics dict.

    Steps:
      1. Initialise RoutingGrid (mark component obstacles).
      2. Prioritise nets (POWER -> GROUND -> SIGNAL).
      3. Run Lee's Algorithm + rip-up-and-reroute + crossing elimination.
      4. If still unrouted, expand grid by +4 each dimension and retry (max 2×).
      5. Compute and return routing metrics.

    Args:
        graph: Phase 2 CircuitGraph with GA-optimised component positions.

    Returns:
        Tuple of:
          - graph          : Same CircuitGraph (passed through; metadata may be
                             updated if grid was auto-expanded).
          - routed_traces  : List of RoutedTrace objects.
          - metrics        : Dict with keys total_routed, total_failed,
                             total_length, crossing_count, longest_trace,
                             shortest_trace, failed_routes.
    """
    gw, gh = graph.metadata.width, graph.metadata.height

    print(f"\n[Phase 3] Step 1/4  Initializing routing grid ({gw}x{gh}) ...")
    r_grid = RoutingGrid(gw, gh)
    r_grid.initialize_from_graph(graph)
    n_comp = r_grid.component_cell_count()
    print(f"   Grid initialized -- {n_comp} component cells blocked")

    print("\n[Phase 3] Step 2/4  Prioritizing nets ...")
    net_router  = NetRouter(graph, r_grid)
    ordered     = net_router._prioritize_nets()
    order_str   = " -> ".join(n.id for n in ordered)
    print(f"   Routing order: {order_str}")

    print("\n[Phase 3] Step 3/4  Running Lee's Algorithm maze router ...")
    routed_traces, failed_routes = net_router.route_all()

    # Grid expansion fallback (up to 2 attempts)
    MAX_EXPANSIONS = 2
    expansion = 0
    while failed_routes and expansion < MAX_EXPANSIONS:
        expansion += 1
        gw += 4
        gh += 4
        print(f"\n[Phase 3] Grid expansion #{expansion}: retrying at {gw}x{gh} ...")
        # Update graph metadata to reflect new grid size
        graph.metadata.width  = gw
        graph.metadata.height = gh
        routed_traces, failed_routes, r_grid = _run_routing_attempt(graph, gw, gh)
        if failed_routes:
            print(f"   Still {len(failed_routes)} unrouted after expansion")
        else:
            print(f"   All nets routed after expansion to {gw}x{gh}")

    print("\n[Phase 3] Step 4/4  Computing routing metrics ...")

    # Crossing count: cells occupied by 2+ DIFFERENT net_ids (exact set-based)
    cell_nets_final: dict[tuple[int, int], set[str]] = defaultdict(set)
    for trace in routed_traces:
        for cell in trace.path[1:-1]:  # skip pin endpoint cells
            cell_nets_final[cell].add(trace.net_id)
    crossing_count = sum(1 for nets in cell_nets_final.values() if len(nets) > 1)

    lengths = [t.length for t in routed_traces]
    metrics: dict = {
        "total_routed":   len(routed_traces),
        "total_failed":   len(failed_routes),
        "total_length":   sum(lengths) if lengths else 0,
        "crossing_count": crossing_count,
        "longest_trace":  max(lengths) if lengths else 0,
        "shortest_trace": min(lengths) if lengths else 0,
        "failed_routes":  failed_routes,
    }

    n_fail  = metrics["total_failed"]
    status  = "all nets connected" if n_fail == 0 else f"{n_fail} segment(s) failed"
    print(f"   Total traces routed : {metrics['total_routed']} segments ({len(ordered)} nets, {status})")
    print(f"   Total trace length  : {metrics['total_length']} cells")
    print(f"   Wire crossings      : {metrics['crossing_count']}")

    out_path = visualize_routing(graph, r_grid, routed_traces, failed_routes, metrics)
    print(f"   Saved -> {out_path}")

    return graph, routed_traces, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phase1_eda_engine import NetlistParser, InitialPlacer
    from phase2_genetic_placer import run_phase2

    _sample = Path(__file__).parent / "netlists" / "sample_netlist.json"
    with _sample.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    parser  = NetlistParser()
    netlist = parser.parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    graph = CircuitGraph.from_netlist(netlist)

    graph = run_phase2(graph)
    graph, traces, metrics = run_phase3(graph)

    print("\n[Phase 3] Complete. Graph + traces ready for Phase 4.")
    sys.exit(0)
