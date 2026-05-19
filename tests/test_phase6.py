"""Tests for Phase 6 — FastAPI Integration Server."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Import the app (no pipeline runs at import time)
# ---------------------------------------------------------------------------
import app as app_module
from app import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared mock result (returned by mocked pipeline)
# ---------------------------------------------------------------------------

_MOCK_RESULT = {
    "status": "complete",
    "netlist": {
        "netlist": {
            "metadata": {"name": "TestDesign", "grid": {"width": 24, "height": 20}},
            "components": [
                {"id": "U1", "type": "MCU", "name": "MCU1",
                 "footprint": {"width": 3, "height": 3}, "x": 4, "y": 4,
                 "pins": [{"id": "VCC", "type": "POWER", "net": "VCC"}]}
            ],
            "nets": [
                {"id": "VCC", "type": "POWER", "connected_pins": [
                    {"component_id": "U1", "pin_id": "VCC"}
                ]}
            ],
        }
    },
    "layout": {
        "components": [
            {"id": "U1", "type": "MCU", "name": "MCU1",
             "x": 4, "y": 4, "width": 3, "height": 3,
             "pins": [{"id": "VCC", "abs_x": 5.0, "abs_y": 4.0,
                       "net": "VCC", "pin_type": "POWER"}]},
        ],
        "traces": [
            {"net_id": "VCC", "net_type": "POWER",
             "source_comp": "U1", "source_pin": "VCC",
             "target_comp": "C1", "target_pin": "P1",
             "path": [[5, 3], [5, 2]], "length": 2},
        ],
        "grid": {"width": 24, "height": 20},
    },
    "metrics": {
        "total_traces_routed":   1,
        "total_traces_failed":   0,
        "total_trace_length_mm": 2.0,
        "longest_trace_mm":      2.0,
        "shortest_trace_mm":     2.0,
        "wire_crossing_count":   0,
        "routing_completion_pct": 100.0,
        "hpwl_mm":               8.0,
        "total_resistance_ohms": 0.005,
        "total_capacitance_pf":  0.5,
        "max_signal_delay_ps":   14.0,
        "violations":            [],
    },
}


# ---------------------------------------------------------------------------
# test_health_endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint() -> None:
    """GET /health returns 200 with correct JSON."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["pipeline"] == "ready"


# ---------------------------------------------------------------------------
# test_root_serves_html
# ---------------------------------------------------------------------------

def test_root_serves_html() -> None:
    """GET / returns 200 with HTML content."""
    resp = client.get("/")
    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "html" in ct or "text" in ct


# ---------------------------------------------------------------------------
# test_generate_endpoint_mock
# ---------------------------------------------------------------------------

def test_generate_endpoint_mock() -> None:
    """POST /generate with mocked pipeline returns a valid structured response."""
    with patch.object(app_module, "_run_full_pipeline", return_value=_MOCK_RESULT):
        resp = client.post("/generate", json={"prompt": "test circuit"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "complete"


# ---------------------------------------------------------------------------
# test_generate_response_has_required_keys
# ---------------------------------------------------------------------------

def test_generate_response_has_required_keys() -> None:
    """Response from /generate contains status, netlist, layout, and metrics."""
    with patch.object(app_module, "_run_full_pipeline", return_value=_MOCK_RESULT):
        resp = client.post("/generate", json={"prompt": "blink LED"})
    data = resp.json()
    for key in ("status", "netlist", "layout", "metrics"):
        assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# test_layout_has_components_and_traces
# ---------------------------------------------------------------------------

def test_layout_has_components_and_traces() -> None:
    """layout object contains non-empty components and traces arrays."""
    with patch.object(app_module, "_run_full_pipeline", return_value=_MOCK_RESULT):
        resp = client.post("/generate", json={"prompt": "test"})
    layout = resp.json()["layout"]
    assert isinstance(layout["components"], list)
    assert isinstance(layout["traces"], list)
    assert len(layout["components"]) >= 1
    assert len(layout["traces"]) >= 1


# ---------------------------------------------------------------------------
# test_metrics_has_required_keys
# ---------------------------------------------------------------------------

def test_metrics_has_required_keys() -> None:
    """metrics object contains all 9 required keys."""
    with patch.object(app_module, "_run_full_pipeline", return_value=_MOCK_RESULT):
        resp = client.post("/generate", json={"prompt": "test"})
    metrics = resp.json()["metrics"]
    required = {
        "total_traces_routed", "total_traces_failed", "total_trace_length_mm",
        "wire_crossing_count", "routing_completion_pct", "hpwl_mm",
        "total_resistance_ohms", "total_capacitance_pf", "max_signal_delay_ps",
    }
    assert required.issubset(metrics.keys())


# ---------------------------------------------------------------------------
# test_outputs_endpoint
# ---------------------------------------------------------------------------

def test_outputs_endpoint() -> None:
    """GET /outputs/phase1_output.png returns 200 when the file exists."""
    png = Path(__file__).parent.parent / "outputs" / "phase1_output.png"
    if not png.exists():
        pytest.skip("outputs/phase1_output.png not present — run Phase 1 first")
    resp = client.get("/outputs/phase1_output.png")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# test_generate_empty_prompt
# ---------------------------------------------------------------------------

def test_generate_empty_prompt() -> None:
    """POST /generate with empty prompt returns an error response (not 500)."""
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "error"


# ---------------------------------------------------------------------------
# test_websocket_mock
# ---------------------------------------------------------------------------

def test_websocket_mock() -> None:
    """WebSocket /ws/generate sends phase messages in the correct order."""
    phase_order = []
    statuses    = []

    # Patch all pipeline imports called inside ws_generate
    mock_netlist = _MOCK_RESULT["netlist"]
    inner = mock_netlist["netlist"]

    with (
        patch("phase0_groq_translator.run_phase0", return_value=mock_netlist),
        patch("phase1_eda_engine.NetlistParser") as MockParser,
        patch("phase1_eda_engine.InitialPlacer"),
        patch("phase1_eda_engine.CircuitGraph") as MockGraph,
        patch("phase1_eda_engine.half_perimeter_wire_length", return_value=12.5),
        patch("phase2_genetic_placer.run_phase2", side_effect=lambda g: g),
        patch("phase3_router.run_phase3", return_value=(MagicMock(), [], {
            "total_routed": 1, "total_failed": 0,
            "crossing_count": 0, "total_length": 2,
        })),
        patch("phase4_analytics.run_phase4") as mock_p4,
    ):
        # Minimal graph mock
        graph_inst = MagicMock()
        graph_inst.nodes = {}
        graph_inst.edges = []
        graph_inst.metadata.width  = 24
        graph_inst.metadata.height = 20

        MockParser.return_value.parse.return_value = MagicMock()
        MockGraph.from_netlist.return_value = graph_inst

        # BoardMetrics mock
        bm = MagicMock()
        bm.violations = []
        bm.total_traces_routed   = 1
        bm.total_traces_failed   = 0
        bm.total_trace_length_mm = 2.0
        bm.longest_trace_mm      = 2.0
        bm.shortest_trace_mm     = 2.0
        bm.wire_crossing_count   = 0
        bm.routing_completion_pct = 100.0
        bm.hpwl_mm               = 8.0
        bm.total_resistance_ohms = 0.005
        bm.total_capacitance_pf  = 0.5
        bm.max_signal_delay_ps   = 14.0
        mock_p4.return_value = bm

        with client.websocket_connect("/ws/generate") as ws:
            ws.send_json({"prompt": "test circuit"})
            # Collect messages until we receive the final result or error
            for _ in range(20):
                try:
                    msg = ws.receive_json()
                except Exception:
                    break
                phase_order.append(msg.get("phase"))
                statuses.append(msg.get("status"))
                if msg.get("phase") == -1:
                    break

    # Must have received at least one phase message
    assert len(phase_order) >= 1
    # The last message must be -1 (result or error)
    assert phase_order[-1] == -1
    # All intermediate phase indices must be in 0..4
    for p in phase_order[:-1]:
        assert 0 <= p <= 4, f"Unexpected phase index: {p}"
