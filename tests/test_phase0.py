"""
Tests for Phase 0 -- Groq API Translator.

All tests that touch the Groq API mock the _call_api method so the suite
runs without a real API key or network access.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase0_groq_translator import (
    FEW_SHOT_EXAMPLES,
    GroqTranslator,
    GroqAPIError,
    MAX_COMPLETION_TOKENS,
    NetlistParseError,
    NetlistValidationError,
    _check_component,
    _check_net,
    run_phase0,
)


def _mock_groq_response(content, finish_reason="stop", reasoning=None):
    """Build a MagicMock shaped like a Groq chat-completion response.

    Mirrors response.choices[0].message.content / .reasoning and
    response.choices[0].finish_reason.
    """
    message = MagicMock()
    message.content = content
    message.reasoning = reasoning
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_netlist() -> dict:
    """A minimal but fully valid netlist dict."""
    return {
        "netlist": {
            "metadata": {
                "name": "Test_Circuit",
                "version": "1.0",
                "generated_by": "GroqAPI",
                "grid": {"width": 24, "height": 20, "unit": "mm"},
            },
            "components": [
                {
                    "id": "U1",
                    "type": "MCU",
                    "name": "ESP32",
                    "footprint": {"width": 4, "height": 6},
                    "x": 8,
                    "y": 7,
                    "properties": {},
                    "pins": [
                        {"id": "VCC", "type": "POWER", "net": "VCC"},
                        {"id": "GND", "type": "POWER", "net": "GND"},
                    ],
                },
                {
                    "id": "C1",
                    "type": "CAPACITOR",
                    "name": "C_DECOUPLING",
                    "footprint": {"width": 1, "height": 1},
                    "x": 9,
                    "y": 5,
                    "properties": {},
                    "pins": [
                        {"id": "P1", "type": "PASSIVE", "net": "VCC"},
                        {"id": "P2", "type": "PASSIVE", "net": "GND"},
                    ],
                },
            ],
            "nets": [
                {
                    "id": "VCC",
                    "type": "POWER",
                    "connected_pins": [
                        {"component_id": "U1", "pin_id": "VCC"},
                        {"component_id": "C1", "pin_id": "P1"},
                    ],
                },
                {
                    "id": "GND",
                    "type": "GROUND",
                    "connected_pins": [
                        {"component_id": "U1", "pin_id": "GND"},
                        {"component_id": "C1", "pin_id": "P2"},
                    ],
                },
            ],
        }
    }


@pytest.fixture()
def translator(tmp_path, monkeypatch) -> GroqTranslator:
    """A GroqTranslator instance with the generated dir redirected to tmp_path."""
    monkeypatch.setattr(
        "phase0_groq_translator._GENERATED_DIR", tmp_path / "generated"
    )
    return GroqTranslator(
        api_key="test-key",
        model="llama-3.3-70b-versatile",
    )


# ---------------------------------------------------------------------------
# 1. Schema validation -- happy path
# ---------------------------------------------------------------------------

def test_schema_validation_passes(translator, valid_netlist):
    """A correctly structured netlist dict must pass _validate_schema."""
    assert translator._validate_schema(valid_netlist) is True


# ---------------------------------------------------------------------------
# 2. Schema validation -- missing components key
# ---------------------------------------------------------------------------

def test_schema_validation_fails_missing_components(translator):
    """A netlist dict that omits the 'components' key must raise NetlistValidationError."""
    bad = {
        "netlist": {
            "metadata": {"name": "X", "version": "1.0", "generated_by": "GroqAPI"},
            "nets": [{"id": "VCC", "type": "POWER", "connected_pins": []}],
        }
    }
    with pytest.raises(NetlistValidationError, match="components"):
        translator._validate_schema(bad)


# ---------------------------------------------------------------------------
# 3. Schema validation -- pin missing 'net' field
# ---------------------------------------------------------------------------

def test_schema_validation_fails_bad_pin(translator):
    """A pin dict without the 'net' key must fail _check_component."""
    bad_comp = {
        "id": "R1",
        "type": "RESISTOR",
        "name": "R_bad",
        "footprint": {"width": 1, "height": 2},
        "pins": [
            {"id": "P1", "type": "PASSIVE"},  # missing "net"
        ],
    }
    with pytest.raises(NetlistValidationError, match="net"):
        _check_component(bad_comp, 0)


# ---------------------------------------------------------------------------
# 4. _parse_response strips markdown fences
# ---------------------------------------------------------------------------

def test_parse_response_strips_markdown(translator):
    """JSON wrapped in ```json ... ``` must be extracted and parsed correctly."""
    payload = {"netlist": {"metadata": {}, "components": [], "nets": []}}
    raw = f"```json\n{json.dumps(payload)}\n```"
    result = translator._parse_response(raw)
    assert result == payload


def test_parse_response_strips_plain_fences(translator):
    """JSON wrapped in plain ``` fences (no 'json' tag) must also be handled."""
    payload = {"netlist": {"metadata": {}, "components": [], "nets": []}}
    raw = f"```\n{json.dumps(payload)}\n```"
    result = translator._parse_response(raw)
    assert result == payload


def test_parse_response_invalid_json_raises(translator):
    """Non-JSON text must raise NetlistParseError."""
    with pytest.raises(NetlistParseError):
        translator._parse_response("This is not JSON at all.")


# ---------------------------------------------------------------------------
# 5. Few-shot examples are valid JSON netlists
# ---------------------------------------------------------------------------

def test_few_shot_examples_are_valid_json():
    """Both assistant turns in FEW_SHOT_EXAMPLES must parse as valid JSON."""
    assistant_turns = [
        msg for msg in FEW_SHOT_EXAMPLES if msg["role"] == "assistant"
    ]
    assert len(assistant_turns) == 2, "Expected exactly 2 assistant few-shot examples"

    for i, turn in enumerate(assistant_turns):
        try:
            parsed = json.loads(turn["content"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"Few-shot example {i} is not valid JSON: {exc}")

        assert "netlist" in parsed, f"Few-shot example {i} missing root 'netlist' key"
        assert "components" in parsed["netlist"], (
            f"Few-shot example {i} missing 'components'"
        )
        assert "nets" in parsed["netlist"], (
            f"Few-shot example {i} missing 'nets'"
        )


# ---------------------------------------------------------------------------
# 6. _save_netlist creates a file in the generated directory
# ---------------------------------------------------------------------------

def test_save_netlist_creates_file(translator, valid_netlist, tmp_path, monkeypatch):
    """_save_netlist must write a JSON file into the generated directory."""
    gen_dir = tmp_path / "generated"
    monkeypatch.setattr("phase0_groq_translator._GENERATED_DIR", gen_dir)

    saved_path = translator._save_netlist(valid_netlist, "test prompt")

    assert saved_path.exists(), "Expected file to be created"
    assert saved_path.suffix == ".json"
    with saved_path.open() as fh:
        on_disk = json.load(fh)
    assert on_disk == valid_netlist


def test_save_netlist_filename_contains_design_name(
    translator, valid_netlist, tmp_path, monkeypatch
):
    """The saved filename must contain the metadata name."""
    gen_dir = tmp_path / "generated"
    monkeypatch.setattr("phase0_groq_translator._GENERATED_DIR", gen_dir)

    saved_path = translator._save_netlist(valid_netlist, "some prompt")
    assert "Test_Circuit" in saved_path.name


# ---------------------------------------------------------------------------
# 7. run_phase0 -- mocked full pipeline
# ---------------------------------------------------------------------------

def test_run_phase0_mock(valid_netlist, tmp_path, monkeypatch):
    """run_phase0 must return a valid netlist dict when _call_api is mocked."""
    monkeypatch.setattr("phase0_groq_translator._GENERATED_DIR", tmp_path / "generated")

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    raw_json = json.dumps(valid_netlist)

    with patch("phase0_groq_translator.GroqTranslator._call_api", return_value=raw_json):
        result = run_phase0("Connect an ESP32 to an LED")

    assert "netlist" in result
    assert len(result["netlist"]["components"]) == 2
    assert len(result["netlist"]["nets"]) == 2


def test_run_phase0_raises_without_api_key(monkeypatch):
    """run_phase0 must raise EnvironmentError when GROQ_API_KEY is absent."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("phase0_groq_translator.load_dotenv"):
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            run_phase0("Connect an ESP32 to an LED")


# ---------------------------------------------------------------------------
# 8. GroqAPIError propagation
# ---------------------------------------------------------------------------

def test_translate_propagates_api_error(translator, monkeypatch):
    """If _call_api raises GroqAPIError, translate must let it propagate."""
    with patch.object(translator, "_call_api", side_effect=GroqAPIError("timeout")):
        with pytest.raises(GroqAPIError, match="timeout"):
            translator.translate("some prompt")


# ---------------------------------------------------------------------------
# 8b. _call_api -- mocked Groq SDK response (complete JSON, json mode, reasoning)
# ---------------------------------------------------------------------------

def test_call_api_parses_complete_json_and_uses_json_mode(translator, valid_netlist):
    """_call_api returns the message content of a complete response, sets json
    mode + MAX_COMPLETION_TOKENS, captures finish_reason, and the parser then
    handles the complete JSON object correctly."""
    raw_json = json.dumps(valid_netlist)
    create = MagicMock(return_value=_mock_groq_response(raw_json, finish_reason="stop"))
    translator._client.chat.completions.create = create

    raw = translator._call_api("Connect an ESP32 to an LED")
    assert raw == raw_json
    assert translator._last_finish_reason == "stop"

    # JSON mode + token limit were actually passed to the SDK.
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == MAX_COMPLETION_TOKENS

    # The complete JSON parses + validates end-to-end.
    parsed = translator._parse_response(raw)
    assert translator._validate_schema(parsed) is True
    assert parsed == valid_netlist


def test_call_api_ignores_reasoning_field(translator, valid_netlist):
    """Reasoning content (gpt-oss) lives in message.reasoning and must NOT be
    parsed — only message.content (the JSON) is returned."""
    raw_json = json.dumps(valid_netlist)
    translator._client.chat.completions.create = MagicMock(
        return_value=_mock_groq_response(
            raw_json, reasoning="Let me think step by step... not JSON {{{"
        )
    )
    raw = translator._call_api("prompt")
    assert raw == raw_json
    assert translator._parse_response(raw) == valid_netlist


def test_parse_truncated_response_reports_length(translator):
    """A truncated (finish_reason='length') response raises NetlistParseError
    whose message names the truncation rather than a silent decode error."""
    translator._last_finish_reason = "length"
    truncated = '{"netlist": {"metadata": {"name": "X"}, "components": [{"id": "U1"'
    with pytest.raises(NetlistParseError, match="TRUNCATED"):
        translator._parse_response(truncated)


# ---------------------------------------------------------------------------
# 8c. translate -- missing 'nets' triggers a single automatic retry
# ---------------------------------------------------------------------------

def test_translate_retries_when_nets_missing(translator, valid_netlist):
    """A first response with components but no 'nets' must trigger ONE retry;
    the valid second response is then returned."""
    no_nets = {"netlist": {
        "metadata": valid_netlist["netlist"]["metadata"],
        "components": valid_netlist["netlist"]["components"],
        # no "nets" key at all
    }}
    create = MagicMock(side_effect=[
        _mock_groq_response(json.dumps(no_nets)),     # 1st call: missing nets
        _mock_groq_response(json.dumps(valid_netlist)),  # retry: valid
    ])
    translator._client.chat.completions.create = create

    result = translator.translate("Connect an ESP32 to an LED")

    assert create.call_count == 2, "expected exactly one retry"
    assert len(result["netlist"]["nets"]) == 2
    # Retry call carried the corrective follow-up message.
    retry_msgs = create.call_args_list[1].kwargs["messages"]
    assert any("nets" in m["content"] for m in retry_msgs if m["role"] == "user")


def test_translate_no_retry_when_valid(translator, valid_netlist):
    """A first valid response (with nets) must NOT trigger a retry."""
    create = MagicMock(return_value=_mock_groq_response(json.dumps(valid_netlist)))
    translator._client.chat.completions.create = create

    result = translator.translate("Connect an ESP32 to an LED")

    assert create.call_count == 1, "valid response must not retry"
    assert len(result["netlist"]["nets"]) == 2


def test_translate_raises_when_nets_missing_after_retry(translator, valid_netlist):
    """If 'nets' is still missing after the single retry, the validation error
    is raised (no silent failure, no further retries)."""
    no_nets = {"netlist": {
        "metadata": valid_netlist["netlist"]["metadata"],
        "components": valid_netlist["netlist"]["components"],
    }}
    create = MagicMock(return_value=_mock_groq_response(json.dumps(no_nets)))
    translator._client.chat.completions.create = create

    with pytest.raises(NetlistValidationError, match="nets"):
        translator.translate("Connect an ESP32 to an LED")
    assert create.call_count == 2, "exactly one retry, then give up"


# ---------------------------------------------------------------------------
# 9. _check_net validates correctly
# ---------------------------------------------------------------------------

def test_check_net_missing_type_raises():
    """A net dict without 'type' must raise NetlistValidationError."""
    bad_net = {"id": "VCC", "connected_pins": []}  # missing "type"
    with pytest.raises(NetlistValidationError, match="type"):
        _check_net(bad_net, 0)


def test_check_net_valid_passes():
    """A well-formed net dict must pass _check_net without raising."""
    good_net = {
        "id": "VCC",
        "type": "POWER",
        "connected_pins": [{"component_id": "U1", "pin_id": "VCC"}],
    }
    _check_net(good_net, 0)  # must not raise
