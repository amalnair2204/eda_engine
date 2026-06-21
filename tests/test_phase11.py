"""Tests for Phase 11 — Design-Space Exploration (Pareto).

One behaviour per test.  Pipeline runs use a tiny option grid (GA only, small
generations/population) so the suite stays fast; the pure Pareto logic is tested
deterministically with no pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app as app_module
from app import app
from fastapi.testclient import TestClient

from phase1_eda_engine import NetlistParser, InitialPlacer, CircuitGraph
from phase11_explorer import (
    dominates,
    pareto_front_indices,
    compute_pareto,
    recommend,
    run_phase11,
)

_SAMPLE_JSON = Path(__file__).parent.parent / "netlists" / "sample_netlist.json"

# Fast, single-placer grid for the pipeline-running tests (GA only → no RL/model
# dependency, two routers → multiple candidates).
_FAST_CONFIG = {
    "placers":        ["ga"],
    "routers":        ["single", "multi"],
    "ga_generations": [15],
    "ga_pop":         12,
}


def _seed_graph() -> CircuitGraph:
    """Parse + seed-place the sample netlist into a fresh CircuitGraph."""
    raw = json.loads(_SAMPLE_JSON.read_text(encoding="utf-8"))
    netlist = NetlistParser().parse(raw)
    InitialPlacer(netlist.metadata).place(netlist)
    return CircuitGraph.from_netlist(netlist)


# ---------------------------------------------------------------------------
# 1. Non-dominated sorting on a hand-built set (deterministic, no pipeline)
# ---------------------------------------------------------------------------

def test_pareto_front_on_handbuilt_tuples() -> None:
    """pareto_front_indices identifies exactly the non-dominated points."""
    # All objectives minimised.  Points:
    #  0:(1,4) 1:(2,2) 2:(4,1)  → mutually non-dominated (the front)
    #  3:(3,3) dominated by 1 ; 4:(5,5) dominated by all
    points = [(1, 4), (2, 2), (4, 1), (3, 3), (5, 5)]
    assert pareto_front_indices(points) == [0, 1, 2]

    # dominates() basics
    assert dominates((1, 1), (2, 2)) is True
    assert dominates((1, 2), (2, 2)) is True       # equal in one, better in other
    assert dominates((2, 2), (2, 2)) is False      # identical → no domination
    assert dominates((1, 3), (3, 1)) is False      # trade-off → neither dominates


# ---------------------------------------------------------------------------
# 2. A candidate with completion < 100% is treated as dominated/invalid
# ---------------------------------------------------------------------------

def test_incomplete_candidate_excluded_from_pareto() -> None:
    """An incomplete candidate is never Pareto-optimal, even with best metrics."""
    candidates = [
        {  # best objectives but NOT fully routed → invalid
            "id": "incomplete", "completion": 80.0,
            "objectives": {"hpwl": 1.0, "crossings": 0, "trace_length": 1.0, "runtime_s": 1.0},
        },
        {  # worse objectives but fully routed → valid, should win the front
            "id": "complete", "completion": 100.0,
            "objectives": {"hpwl": 5.0, "crossings": 2, "trace_length": 9.0, "runtime_s": 3.0},
        },
    ]
    front = compute_pareto(candidates)
    front_ids = {c["id"] for c in front}
    assert "incomplete" not in front_ids
    assert front_ids == {"complete"}
    assert candidates[0]["pareto"] is False
    assert candidates[1]["pareto"] is True


# ---------------------------------------------------------------------------
# 3. run_phase11 returns >1 candidate and a non-empty Pareto set
# ---------------------------------------------------------------------------

def test_run_phase11_produces_candidates_and_pareto() -> None:
    """run_phase11 on the sample netlist sweeps >1 candidate with a Pareto set."""
    result = run_phase11(_seed_graph(), _FAST_CONFIG)
    assert len(result["candidates"]) > 1
    assert len(result["pareto"]) >= 1
    # Artifacts generated.
    assert Path(result["pareto_png"]).exists()
    assert Path(result["results_md"]).exists()


# ---------------------------------------------------------------------------
# 4. The recommendation is a member of the Pareto set
# ---------------------------------------------------------------------------

def test_recommendation_is_in_pareto_set() -> None:
    """The recommended candidate is drawn from the Pareto-optimal set."""
    result = run_phase11(_seed_graph(), _FAST_CONFIG)
    rec = result["recommendation"]
    assert rec is not None
    pareto_ids = {c["id"] for c in result["pareto"]}
    assert rec["id"] in pareto_ids
    assert rec["rationale"]   # non-empty rationale string


# ---------------------------------------------------------------------------
# 5. app.py /explore rejects when no netlist exists
# ---------------------------------------------------------------------------

def test_explore_rejects_without_netlist() -> None:
    """POST /explore returns 400 when no netlist has been generated yet."""
    client = TestClient(app, raise_server_exceptions=False)
    app_module._LAST_ROUTED.clear()   # ensure no cached netlist
    resp = client.post("/explore", json={})
    assert resp.status_code == 400
    assert "netlist" in resp.json()["detail"].lower()
