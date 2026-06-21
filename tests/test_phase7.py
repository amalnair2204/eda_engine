"""Tests for Phase 7 — Reinforcement Learning placement agent.

One dedicated test per behaviour (no bundling).  Tests that require the trained
policy are skipped automatically if models/phase7_rl_placer.zip is absent, so
the suite stays green on a fresh checkout; with the policy present they fully
exercise RL inference, validity, and end-to-end routability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import app as app_module
from app import app

from phase1_eda_engine import (
    CircuitGraph,
    InitialPlacer,
    NetlistParser,
)
from phase7_rl_placer import (
    MODEL_PATH,
    PlacementEnv,
    RLPlacer,
    run_phase7,
)

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"

_HAS_MODEL = MODEL_PATH.exists()
_needs_model = pytest.mark.skipif(
    not _HAS_MODEL,
    reason="trained policy models/phase7_rl_placer.zip not present — run python -m train_phase7_rl",
)

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_graph() -> CircuitGraph:
    """Standard 4-component graph from sample_netlist.json (placed)."""
    raw = json.loads(_SAMPLE_JSON.read_text(encoding="utf-8"))
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    return CircuitGraph.from_netlist(netlist)


# ---------------------------------------------------------------------------
# test_check_env_passes
# ---------------------------------------------------------------------------

def test_check_env_passes(sample_graph: CircuitGraph) -> None:
    """PlacementEnv conforms to the Gymnasium API (check_env, no errors)."""
    from gymnasium.utils.env_checker import check_env

    env = PlacementEnv(sample_graph)
    check_env(env, skip_render_check=True)   # raises on any violation


# ---------------------------------------------------------------------------
# test_reset_returns_valid_observation
# ---------------------------------------------------------------------------

def test_reset_returns_valid_observation(sample_graph: CircuitGraph) -> None:
    """reset() yields an observation inside the declared observation space."""
    env = PlacementEnv(sample_graph)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "action_mask" in info
    assert info["action_mask"].shape == (env.action_space.n,)


# ---------------------------------------------------------------------------
# test_step_respects_action_mask
# ---------------------------------------------------------------------------

def test_step_respects_action_mask(sample_graph: CircuitGraph) -> None:
    """Every mask-legal action is in-bounds + overlap-free; a step stays valid."""
    env = PlacementEnv(sample_graph)
    obs, info = env.reset(seed=1)
    mask = info["action_mask"]
    assert mask.any(), "mask must permit at least one legal placement"

    # The first component to place (none placed yet → mask = in-bounds anchors).
    comp = env._graph.nodes[env._order[0]]
    fw, fh = comp.footprint.width, comp.footprint.height
    for idx in np.flatnonzero(mask):
        x, y = idx % env.grid_w, idx // env.grid_w
        assert 0 <= x and x + fw <= env.grid_w
        assert 0 <= y and y + fh <= env.grid_h

    # Stepping a legal action keeps the observation valid and the comp in-grid.
    legal = int(np.flatnonzero(mask)[0])
    obs2, reward, terminated, truncated, info2 = env.step(legal)
    assert env.observation_space.contains(obs2)
    placed = env._graph.nodes[env._order[0]]
    assert 0 <= placed.x and placed.x + fw <= env.grid_w
    assert 0 <= placed.y and placed.y + fh <= env.grid_h


# ---------------------------------------------------------------------------
# test_run_phase7_returns_valid_graph
# ---------------------------------------------------------------------------

@_needs_model
def test_run_phase7_returns_valid_graph(sample_graph: CircuitGraph) -> None:
    """run_phase7 returns a CircuitGraph with 0 overlaps, all comps in-grid."""
    result = run_phase7(sample_graph)
    assert isinstance(result, CircuitGraph)

    gw, gh = result.metadata.width, result.metadata.height
    comps = list(result.nodes.values())

    # All components inside the grid.
    for c in comps:
        assert c.x >= 0 and c.y >= 0
        assert c.x + c.footprint.width <= gw
        assert c.y + c.footprint.height <= gh

    # Zero overlapping pairs.
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            overlap = not (
                a.x + a.footprint.width <= b.x
                or b.x + b.footprint.width <= a.x
                or a.y + a.footprint.height <= b.y
                or b.y + b.footprint.height <= a.y
            )
            assert not overlap, f"{a.id} overlaps {b.id}"


# ---------------------------------------------------------------------------
# test_rl_placement_is_routable
# ---------------------------------------------------------------------------

@_needs_model
def test_rl_placement_is_routable(sample_graph: CircuitGraph) -> None:
    """RL placement routes end-to-end at 100% completion on the sample netlist."""
    from phase3_router import run_phase3
    from phase4_analytics import run_phase4

    graph = run_phase7(sample_graph)
    graph, traces, p3 = run_phase3(graph)
    board = run_phase4(graph, traces, p3)
    assert board.routing_completion_pct == 100.0, (
        f"RL placement not fully routable: {board.routing_completion_pct}% "
        f"(crossings={board.wire_crossing_count})"
    )


# ---------------------------------------------------------------------------
# test_app_accepts_placer_ga
# ---------------------------------------------------------------------------

def test_app_accepts_placer_ga() -> None:
    """POST /generate with placer=ga is accepted and runs the pipeline."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate", json={"prompt": "blink LED", "placer": "ga"})
    assert resp.status_code == 200
    assert resp.json().get("status") == "complete"
    # placer value forwarded to the pipeline
    assert m.call_args.args[1] == "ga"


# ---------------------------------------------------------------------------
# test_app_accepts_placer_rl
# ---------------------------------------------------------------------------

def test_app_accepts_placer_rl() -> None:
    """POST /generate with placer=rl is accepted and forwards 'rl'."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate", json={"prompt": "blink LED", "placer": "rl"})
    assert resp.status_code == 200
    assert m.call_args.args[1] == "rl"


# ---------------------------------------------------------------------------
# test_app_rejects_garbage_placer
# ---------------------------------------------------------------------------

def test_app_rejects_garbage_placer() -> None:
    """POST /generate with an unknown placer returns HTTP 400."""
    resp = client.post("/generate", json={"prompt": "blink LED", "placer": "bogus"})
    assert resp.status_code == 400
    assert "placer" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# test_app_default_placer_is_ga
# ---------------------------------------------------------------------------

def test_app_default_placer_is_ga() -> None:
    """Omitting placer defaults to 'ga' (GA remains the default everywhere)."""
    with patch.object(app_module, "_run_full_pipeline",
                      return_value={"status": "complete"}) as m:
        resp = client.post("/generate", json={"prompt": "blink LED"})
    assert resp.status_code == 200
    assert m.call_args.args[1] == "ga"
