"""
Phase 7 — Reinforcement Learning Placement Agent

A drop-in alternative to the Phase 2 Genetic Algorithm placer.  This honours
the project's Strategy pattern: a Placer is a swappable algorithm that takes a
CircuitGraph and minimises HPWL by mutating only Component.x / Component.y.

This module is functionally an ALTERNATIVE to Phase 2 (selectable at runtime),
NOT a stage that runs after Phase 6.  Phase 3 consumes its output unchanged.

Sections
--------
1. Helpers          — grid dims from .env, partial-HPWL sub-graph reuse
2. PlacementEnv     — Gymnasium environment (sequential masked placement)
3. RLPlacer         — inference wrapper conforming to the Placer interface
4. run_phase7()     — pipeline entry-point mirroring run_phase2()
5. CLI entry-point

Design notes
------------
* Placement order: components are placed sequentially, ordered by descending
  node degree (most-connected first), fixed for the whole episode.
* Observation (Box, fixed 9-d): features of the component being placed next
  (footprint, pin count, degree, placed-neighbour centroid "pull" hint) plus
  global features (fraction placed, current bounding-box extent).  All spatial
  values are normalised by the grid diagonal / grid dimensions and clipped to
  [0, 1] so the vector is valid for any circuit in the training suite.
* Action (Discrete, GRID_WIDTH * GRID_HEIGHT): the top-left placement anchor
  of the current component.  An action mask (info["action_mask"] and the
  action_masks() method used by MaskablePPO) zeroes every cell that would put
  the footprint out of bounds or overlapping an already-placed component.
* Reward: per step  -(Δ partial-HPWL) / grid-diagonal  (reuses the Phase 2 /
  Phase 1 HPWL function — never reimplemented).  Terminal adds
  -(final HPWL / diagonal) minus overlap and out-of-grid penalties (~0 when
  masking is respected).

The HALF-PERIMETER WIRE LENGTH function is imported from the Phase 2 module
(which re-exports Phase 1's implementation).  It is never reimplemented here.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from dotenv import load_dotenv

from phase1_eda_engine import CircuitGraph, Component, GridMetadata
# Reuse the existing HPWL fitness function + pin-position sync from Phase 2.
from phase2_genetic_placer import (
    half_perimeter_wire_length,
    _update_graph_pin_positions,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_OUTPUT_DIR   = _PROJECT_ROOT / "outputs"
_MODELS_DIR   = _PROJECT_ROOT / "models"
MODEL_PATH    = _MODELS_DIR / "phase7_rl_placer.zip"

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
OBS_DIM   = 9       # fixed-size observation vector
_PIN_NORM = 40.0    # normalisation constant for pin counts (ESP32 has 38)

# Reward penalties (only bite when the agent ignores the mask)
_OVERLAP_PENALTY     = 5.0
_OUT_OF_GRID_PENALTY = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def grid_dims() -> tuple[int, int]:
    """Return (GRID_WIDTH, GRID_HEIGHT) read from the environment (.env).

    Returns:
        Tuple of grid width and height in cells (defaults 24 x 20).
    """
    return (int(os.getenv("GRID_WIDTH", "24")), int(os.getenv("GRID_HEIGHT", "20")))


def _placement_order(graph: CircuitGraph) -> list[str]:
    """Component ids ordered by descending node degree (id tie-break).

    Args:
        graph: CircuitGraph whose adjacency defines node degree.

    Returns:
        List of component ids, most-connected first, deterministic.
    """
    return sorted(
        graph.nodes.keys(),
        key=lambda cid: (-len(graph.adjacency.get(cid, set())), cid),
    )


def _partial_hpwl(graph: CircuitGraph, placed_ids: set[str]) -> float:
    """HPWL over the sub-graph of already-placed components.

    Builds a lightweight CircuitGraph containing only the placed nodes and the
    edges whose BOTH endpoints are placed, then delegates to the Phase 2 / 1
    HPWL function.  This reuses the canonical HPWL implementation exactly.

    Args:
        graph:      The working CircuitGraph (positions partially assigned).
        placed_ids: Set of component ids already placed this episode.

    Returns:
        Total HPWL over placed nets (0.0 if fewer than 2 placed comps share a net).
    """
    if len(placed_ids) < 2:
        return 0.0
    sub_edges = [
        e for e in graph.edges
        if e.source[0] in placed_ids and e.target[0] in placed_ids
    ]
    if not sub_edges:
        return 0.0
    sub = CircuitGraph(
        nodes={cid: graph.nodes[cid] for cid in placed_ids},
        edges=sub_edges,
        adjacency={},
        metadata=graph.metadata,
    )
    return half_perimeter_wire_length(sub)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — PlacementEnv  (Gymnasium)
# ═══════════════════════════════════════════════════════════════════════════════

class PlacementEnv(gym.Env):
    """Sequential, mask-guided component placement environment.

    One episode places every component of a single circuit exactly once, in
    descending-degree order.  The action selects the grid anchor for the
    current component; an action mask guarantees in-bounds, non-overlapping
    placements.  Reward shapes toward minimal HPWL (see module docstring).

    Args:
        graphs:  A single CircuitGraph or a list of them.  reset() samples one
                 circuit per episode, enabling training across a suite.
        grid_w:  Grid width  (defaults to GRID_WIDTH from .env).
        grid_h:  Grid height (defaults to GRID_HEIGHT from .env).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        graphs: CircuitGraph | list[CircuitGraph],
        grid_w: int | None = None,
        grid_h: int | None = None,
    ) -> None:
        super().__init__()
        env_w, env_h = grid_dims()
        self.grid_w = int(grid_w if grid_w is not None else env_w)
        self.grid_h = int(grid_h if grid_h is not None else env_h)
        self._diag  = float(np.hypot(self.grid_w, self.grid_h))

        self._templates: list[CircuitGraph] = (
            list(graphs) if isinstance(graphs, list) else [graphs]
        )
        if not self._templates:
            raise ValueError("PlacementEnv requires at least one CircuitGraph.")

        self.action_space = spaces.Discrete(self.grid_w * self.grid_h)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Per-episode state (populated by reset)
        self._graph: CircuitGraph | None = None
        self._order: list[str] = []
        self._n: int = 0
        self._idx: int = 0
        self._placed: set[str] = set()
        self._prev_partial_hpwl: float = 0.0
        self._mask: np.ndarray = np.ones(self.action_space.n, dtype=bool)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode on a freshly chosen circuit.

        Args:
            seed:    RNG seed (forwarded to gym.Env for reproducibility).
            options: Optional {"graph_index": int} to force a specific circuit.

        Returns:
            (observation, info) where info carries "action_mask".
        """
        super().reset(seed=seed)

        if options and "graph_index" in options:
            template = self._templates[int(options["graph_index"]) % len(self._templates)]
        else:
            template = self._templates[int(self.np_random.integers(len(self._templates)))]

        self._graph = copy.deepcopy(template)
        self._order = _placement_order(self._graph)
        self._n     = len(self._order)
        self._idx   = 0
        self._placed = set()
        self._prev_partial_hpwl = 0.0
        self._recompute_mask()

        return self._build_obs(), {"action_mask": self._mask.copy()}

    def step(self, action: int):
        """Place the current component at the decoded grid anchor.

        Args:
            action: Flat grid index (y * grid_w + x) for the top-left anchor.

        Returns:
            (observation, reward, terminated, truncated, info).
        """
        assert self._graph is not None, "step() called before reset()"
        action = int(action)
        x, y = action % self.grid_w, action // self.grid_w
        comp = self._graph.nodes[self._order[self._idx]]

        penalty = 0.0
        if not self._is_valid_anchor(comp, x, y):
            # check_env / un-masked agents may pick an illegal cell — snap to a
            # legal one (if any) and apply a shaping penalty.
            valid = np.flatnonzero(self._mask)
            if valid.size == 0:
                # No legal placement for this component — end the episode.
                obs = self._build_obs(terminal=True)
                return obs, -_OUT_OF_GRID_PENALTY, True, False, {
                    "action_mask": self._mask.copy()
                }
            a = int(valid[0])
            x, y = a % self.grid_w, a // self.grid_w
            penalty = _OVERLAP_PENALTY

        comp.x, comp.y = x, y
        self._placed.add(comp.id)
        self._idx += 1

        partial = _partial_hpwl(self._graph, self._placed)
        reward = -((partial - self._prev_partial_hpwl) / self._diag) - penalty
        self._prev_partial_hpwl = partial

        terminated = self._idx >= self._n
        if terminated:
            final_hpwl = half_perimeter_wire_length(self._graph)
            overlaps   = self._count_overlaps()
            out_grid   = self._count_out_of_grid()
            reward += -(final_hpwl / self._diag)
            reward += -_OVERLAP_PENALTY * overlaps
            reward += -_OUT_OF_GRID_PENALTY * out_grid
            obs = self._build_obs(terminal=True)
            return obs, reward, True, False, {"action_mask": self._mask.copy()}

        self._recompute_mask()
        return (
            self._build_obs(),
            reward,
            False,
            False,
            {"action_mask": self._mask.copy()},
        )

    # ------------------------------------------------------------------
    # Action masking (used by MaskablePPO + exposed in info)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return the boolean mask of legal actions for the current component.

        Returns:
            1-D bool array of length grid_w * grid_h (True = legal anchor).
        """
        return self._mask.copy()

    def _recompute_mask(self) -> None:
        """Recompute self._mask for the component currently due to be placed."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self._idx >= self._n:
            mask[0] = True            # dummy: never used (episode is terminal)
            self._mask = mask
            return
        comp = self._graph.nodes[self._order[self._idx]]
        fw, fh = comp.footprint.width, comp.footprint.height
        for y in range(self.grid_h - fh + 1):
            for x in range(self.grid_w - fw + 1):
                if not self._overlaps_placed(comp, x, y):
                    mask[y * self.grid_w + x] = True
        if not mask.any():
            mask[0] = True            # guarantee at least one legal action
        self._mask = mask

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _is_valid_anchor(self, comp: Component, x: int, y: int) -> bool:
        """True if comp at (x, y) is in-bounds and overlaps nothing placed."""
        fw, fh = comp.footprint.width, comp.footprint.height
        if x < 0 or y < 0 or x + fw > self.grid_w or y + fh > self.grid_h:
            return False
        return not self._overlaps_placed(comp, x, y)

    def _overlaps_placed(self, comp: Component, x: int, y: int) -> bool:
        """True if comp at (x, y) overlaps any already-placed component."""
        fw, fh = comp.footprint.width, comp.footprint.height
        for pid in self._placed:
            o = self._graph.nodes[pid]
            if not (
                x + fw <= o.x
                or o.x + o.footprint.width <= x
                or y + fh <= o.y
                or o.y + o.footprint.height <= y
            ):
                return True
        return False

    def _count_overlaps(self) -> int:
        """Number of overlapping placed component pairs (should be 0)."""
        comps = [self._graph.nodes[c] for c in self._placed]
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

    def _count_out_of_grid(self) -> int:
        """Number of placed components whose footprint exits the grid."""
        n = 0
        for cid in self._placed:
            c = self._graph.nodes[cid]
            if (c.x < 0 or c.y < 0
                    or c.x + c.footprint.width > self.grid_w
                    or c.y + c.footprint.height > self.grid_h):
                n += 1
        return n

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self, terminal: bool = False) -> np.ndarray:
        """Construct the fixed-size observation for the current component.

        Args:
            terminal: When True (all placed), return a valid all-zero vector.

        Returns:
            float32 array of shape (OBS_DIM,) within [0, 1].
        """
        if terminal or self._idx >= self._n:
            return np.zeros(OBS_DIM, dtype=np.float32)

        comp = self._graph.nodes[self._order[self._idx]]
        n_comp = max(1, self._n)

        # Placed neighbours of the current component → centroid "pull" hint
        neighbours = self._graph.adjacency.get(comp.id, set())
        placed_nb = [nb for nb in neighbours if nb in self._placed]
        if placed_nb:
            cx = np.mean([
                self._graph.nodes[nb].x + self._graph.nodes[nb].footprint.width / 2.0
                for nb in placed_nb
            ])
            cy = np.mean([
                self._graph.nodes[nb].y + self._graph.nodes[nb].footprint.height / 2.0
                for nb in placed_nb
            ])
            nb_x, nb_y, has_nb = cx / self.grid_w, cy / self.grid_h, 1.0
        else:
            nb_x, nb_y, has_nb = 0.5, 0.5, 0.0   # sentinel: centre, no neighbour

        # Current bounding-box extent of placed components (half-perimeter)
        if self._placed:
            xs, ys = [], []
            for cid in self._placed:
                c = self._graph.nodes[cid]
                xs += [c.x, c.x + c.footprint.width]
                ys += [c.y, c.y + c.footprint.height]
            bbox_extent = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / self._diag
        else:
            bbox_extent = 0.0

        obs = np.array([
            comp.footprint.width  / self.grid_w,
            comp.footprint.height / self.grid_h,
            len(comp.pins) / _PIN_NORM,
            len(neighbours) / max(1, n_comp - 1),
            nb_x,
            nb_y,
            has_nb,
            self._idx / n_comp,
            bbox_extent,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — RLPlacer  (inference wrapper — Placer interface)
# ═══════════════════════════════════════════════════════════════════════════════

class RLPlacer:
    """Loads the trained MaskablePPO policy and places greedily (masked).

    Conforms to the Placer Strategy interface (``__call__(graph) -> graph``),
    mutating only Component.x / Component.y and refreshing pin positions.

    Args:
        model_path: Path to the saved policy zip (defaults to the module path).

    Raises:
        FileNotFoundError: If the trained policy file does not exist.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RL policy not found at '{self.model_path}'. "
                f"Train it first:  python -m train_phase7_rl"
            )
        from sb3_contrib import MaskablePPO   # local import keeps SB3 optional
        self._model = MaskablePPO.load(str(self.model_path))

    # ------------------------------------------------------------------

    def place(self, graph: CircuitGraph) -> CircuitGraph:
        """Place all components deterministically with the trained policy.

        Args:
            graph: CircuitGraph to optimise (Component.x/.y are overwritten).

        Returns:
            The same CircuitGraph with optimised positions and refreshed pins.
        """
        env = PlacementEnv(graph)
        obs, info = env.reset(options={"graph_index": 0})
        terminated = False
        while not terminated:
            mask = info["action_mask"]
            action, _ = self._model.predict(
                obs, action_masks=mask, deterministic=True
            )
            obs, _reward, terminated, _truncated, info = env.step(int(action))

        # Copy the env's optimised positions back onto the caller's graph.
        for cid, comp in graph.nodes.items():
            comp.x = env._graph.nodes[cid].x
            comp.y = env._graph.nodes[cid].y
        _update_graph_pin_positions(graph)
        return graph

    # Placer protocol: an RLPlacer instance is itself a callable strategy.
    def __call__(self, graph: CircuitGraph) -> CircuitGraph:
        """Alias for place() so RLPlacer satisfies the Placer interface."""
        return self.place(graph)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Pipeline function
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase7(graph: CircuitGraph) -> CircuitGraph:
    """Phase 7 pipeline entry-point: CircuitGraph → RL-optimised CircuitGraph.

    Drop-in alternative to run_phase2().  Returns the SAME type so Phase 3
    consumes its output unchanged.

    Args:
        graph: CircuitGraph from Phase 1 (positions at seed values).

    Returns:
        The same CircuitGraph with Component.x/.y set by the RL policy and
        Pin.abs_x/.abs_y refreshed.  Ready for the Phase 3 router.

    Raises:
        FileNotFoundError: If the trained policy is missing.
    """
    hpwl_before = half_perimeter_wire_length(graph)
    print(f"\n[Phase 7] HPWL before RL : {hpwl_before:.2f}")

    placer = RLPlacer()
    graph  = placer.place(graph)

    hpwl_after  = half_perimeter_wire_length(graph)
    improvement = (hpwl_before - hpwl_after) / max(hpwl_before, 1e-9) * 100
    print(f"[Phase 7] HPWL after RL  : {hpwl_after:.2f}")
    print(f"[Phase 7] Improvement    : {improvement:.1f}%")
    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — CLI entry-point
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

    graph = run_phase7(graph)
    print("\n[Phase 7] Complete. CircuitGraph is ready for Phase 3.")
    sys.exit(0)
