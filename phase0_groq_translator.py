"""
Phase 0 -- Groq API Translator
Converts a plain-English circuit description into a validated JSON netlist
that Phase 1's NetlistParser can consume directly.

Pipeline position: entry point -> produces raw JSON dict for Phase 1.

LLM backend: Groq (groq.com) running LLaMA 3.3 70B via the official groq SDK.
The Groq SDK interface is nearly identical to the OpenAI SDK; only the import,
client constructor, and env-var names differ.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError, APITimeoutError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_GENERATED_DIR = _PROJECT_ROOT / "netlists" / "generated"

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GroqAPIError(RuntimeError):
    """Raised when the Groq API call itself fails (network, auth, rate-limit)."""


class NetlistParseError(ValueError):
    """Raised when the API response cannot be decoded as valid JSON."""


class NetlistValidationError(ValueError):
    """Raised when the JSON does not conform to the expected netlist schema."""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

NETLIST_SYSTEM_PROMPT: str = """\
You are an expert Electronic Design Automation (EDA) netlist generator.

OUTPUT RULES -- follow these without exception:
1. Output ONLY valid JSON. No markdown, no backticks, no prose, no code fences.
2. The root key must be "netlist" and contain exactly three keys:
   "metadata", "components", "nets".

SCHEMA:
{
  "netlist": {
    "metadata": {
      "name": "<DesignName_no_spaces>",
      "version": "1.0",
      "generated_by": "GroqAPI",
      "grid": {"width": 24, "height": 20, "unit": "mm"}
    },
    "components": [
      {
        "id": "<unique string, e.g. U1, R1, C1, D1>",
        "type": "<MCU | RESISTOR | CAPACITOR | LED | IC | POWER>",
        "name": "<descriptive name>",
        "footprint": {"width": <int>, "height": <int>},
        "x": <int, 1..22>,
        "y": <int, 1..18>,
        "properties": { <key-value pairs relevant to this component type> },
        "pins": [
          {
            "id": "<pin name>",
            "type": "<OUTPUT | INPUT | PASSIVE | POWER | BIDIRECTIONAL>",
            "net": "<net id this pin connects to>"
          }
        ]
      }
    ],
    "nets": [
      {
        "id": "<net name>",
        "type": "<POWER | SIGNAL | GROUND>",
        "connected_pins": [
          {"component_id": "<comp id>", "pin_id": "<pin id>"}
        ]
      }
    ]
  }
}

COMPONENT TYPE RULES:
- MCU: microcontrollers (ESP32, Arduino, STM32, etc.) -- use type "MCU"
- IC: all other integrated circuits (555 timer, op-amp, logic gates, etc.)
- RESISTOR, CAPACITOR, LED: discrete passives / indicators
- POWER: explicit power supply or voltage regulator components

PIN TYPE RULES:
- OUTPUT: drives a signal (GPIO driving a load, IC output pin)
- INPUT: receives a signal (load input, comparator input)
- PASSIVE: no directionality (resistor legs, capacitor legs)
- POWER: power supply / ground pins (VCC, GND, 3V3, 5V)
- BIDIRECTIONAL: SDA, SCK, MOSI, etc.

NET TYPE RULES:
- POWER: VCC, 3V3, 5V, VBAT and similar supply rails
- GROUND: GND, AGND, DGND
- SIGNAL: everything else (data lines, drive signals, timer outputs)

MANDATORY EEE RULES you must always follow:
1. Every VCC/power rail must have at least one decoupling capacitor (100nF)
   placed within the design. Add it even if the user did not ask.
2. Net names must be descriptive (LED_DRIVE, TIMER_OUT, TRIG_IN) -- never
   generic (NET1, NET2, N001).
3. Place MCU/IC components away from grid edges: x in [3..20], y in [3..16].
4. Place passive components (R, C, LED) anywhere: x in [1..22], y in [1..18].
5. Power nets (VCC, 3V3, 5V) must never share a net with GND.
6. Every component's pins must appear in exactly the nets they connect to.

PLACEMENT: Spread components realistically -- do not stack them all at the
same coordinate. Use the full grid width (24) and height (20).
"""

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # --- Example 1 ---
    {
        "role": "user",
        "content": "Connect an ESP32 to a status LED via a 330 ohm resistor",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "netlist": {
                    "metadata": {
                        "name": "ESP32_LED_Circuit",
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
                            "properties": {"package": "WROOM-32", "frequency_mhz": 240},
                            "pins": [
                                {"id": "VCC",   "type": "POWER",  "net": "VCC"},
                                {"id": "GND",   "type": "POWER",  "net": "GND"},
                                {"id": "GPIO2", "type": "OUTPUT", "net": "LED_DRIVE"},
                            ],
                        },
                        {
                            "id": "R1",
                            "type": "RESISTOR",
                            "name": "R_330",
                            "footprint": {"width": 1, "height": 2},
                            "x": 14,
                            "y": 9,
                            "properties": {"resistance_ohm": 330, "power_rating_w": 0.25},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "LED_DRIVE"},
                                {"id": "P2", "type": "PASSIVE", "net": "LED_ANODE"},
                            ],
                        },
                        {
                            "id": "D1",
                            "type": "LED",
                            "name": "LED_STATUS",
                            "footprint": {"width": 1, "height": 2},
                            "x": 18,
                            "y": 9,
                            "properties": {
                                "color": "red",
                                "forward_voltage_v": 2.0,
                                "max_current_ma": 20,
                            },
                            "pins": [
                                {"id": "ANODE",   "type": "INPUT",   "net": "LED_ANODE"},
                                {"id": "CATHODE", "type": "PASSIVE", "net": "GND"},
                            ],
                        },
                        {
                            "id": "C1",
                            "type": "CAPACITOR",
                            "name": "C_DECOUPLING",
                            "footprint": {"width": 1, "height": 1},
                            "x": 9,
                            "y": 5,
                            "properties": {"capacitance_uf": 0.1, "voltage_rating_v": 10},
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
                                {"component_id": "D1", "pin_id": "CATHODE"},
                                {"component_id": "C1", "pin_id": "P2"},
                            ],
                        },
                        {
                            "id": "LED_DRIVE",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "U1", "pin_id": "GPIO2"},
                                {"component_id": "R1", "pin_id": "P1"},
                            ],
                        },
                        {
                            "id": "LED_ANODE",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "R1", "pin_id": "P2"},
                                {"component_id": "D1", "pin_id": "ANODE"},
                            ],
                        },
                    ],
                }
            },
            separators=(",", ":"),
        ),
    },
    # --- Example 2 ---
    {
        "role": "user",
        "content": "Build a 555 timer astable circuit with two resistors and a capacitor",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "netlist": {
                    "metadata": {
                        "name": "NE555_Astable_Oscillator",
                        "version": "1.0",
                        "generated_by": "GroqAPI",
                        "grid": {"width": 24, "height": 20, "unit": "mm"},
                    },
                    "components": [
                        {
                            "id": "U1",
                            "type": "IC",
                            "name": "NE555",
                            "footprint": {"width": 3, "height": 5},
                            "x": 10,
                            "y": 8,
                            "properties": {"package": "DIP-8"},
                            "pins": [
                                {"id": "GND",  "type": "POWER",  "net": "GND"},
                                {"id": "TRIG", "type": "INPUT",  "net": "TIMER_THRESHOLD"},
                                {"id": "OUT",  "type": "OUTPUT", "net": "ASTABLE_OUT"},
                                {"id": "RST",  "type": "INPUT",  "net": "VCC"},
                                {"id": "CV",   "type": "INPUT",  "net": "CV_BYPASS"},
                                {"id": "THR",  "type": "INPUT",  "net": "TIMER_THRESHOLD"},
                                {"id": "DIS",  "type": "OUTPUT", "net": "TIMER_CHARGE"},
                                {"id": "VCC",  "type": "POWER",  "net": "VCC"},
                            ],
                        },
                        {
                            "id": "R1",
                            "type": "RESISTOR",
                            "name": "RA_1K",
                            "footprint": {"width": 1, "height": 2},
                            "x": 10,
                            "y": 4,
                            "properties": {"resistance_ohm": 1000, "power_rating_w": 0.25},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "VCC"},
                                {"id": "P2", "type": "PASSIVE", "net": "TIMER_CHARGE"},
                            ],
                        },
                        {
                            "id": "R2",
                            "type": "RESISTOR",
                            "name": "RB_10K",
                            "footprint": {"width": 1, "height": 2},
                            "x": 14,
                            "y": 6,
                            "properties": {"resistance_ohm": 10000, "power_rating_w": 0.25},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "TIMER_CHARGE"},
                                {"id": "P2", "type": "PASSIVE", "net": "TIMER_THRESHOLD"},
                            ],
                        },
                        {
                            "id": "C1",
                            "type": "CAPACITOR",
                            "name": "CT_10UF",
                            "footprint": {"width": 1, "height": 2},
                            "x": 14,
                            "y": 13,
                            "properties": {"capacitance_uf": 10.0, "voltage_rating_v": 16},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "TIMER_THRESHOLD"},
                                {"id": "P2", "type": "PASSIVE", "net": "GND"},
                            ],
                        },
                        {
                            "id": "C2",
                            "type": "CAPACITOR",
                            "name": "C_DECOUPLING",
                            "footprint": {"width": 1, "height": 1},
                            "x": 11,
                            "y": 5,
                            "properties": {"capacitance_uf": 0.1, "voltage_rating_v": 10},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "VCC"},
                                {"id": "P2", "type": "PASSIVE", "net": "GND"},
                            ],
                        },
                        {
                            "id": "C3",
                            "type": "CAPACITOR",
                            "name": "CV_BYPASS_CAP",
                            "footprint": {"width": 1, "height": 1},
                            "x": 16,
                            "y": 10,
                            "properties": {"capacitance_uf": 0.01, "voltage_rating_v": 10},
                            "pins": [
                                {"id": "P1", "type": "PASSIVE", "net": "CV_BYPASS"},
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
                                {"component_id": "U1", "pin_id": "RST"},
                                {"component_id": "R1", "pin_id": "P1"},
                                {"component_id": "C2", "pin_id": "P1"},
                            ],
                        },
                        {
                            "id": "GND",
                            "type": "GROUND",
                            "connected_pins": [
                                {"component_id": "U1", "pin_id": "GND"},
                                {"component_id": "C1", "pin_id": "P2"},
                                {"component_id": "C2", "pin_id": "P2"},
                                {"component_id": "C3", "pin_id": "P2"},
                            ],
                        },
                        {
                            "id": "TIMER_CHARGE",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "R1", "pin_id": "P2"},
                                {"component_id": "R2", "pin_id": "P1"},
                                {"component_id": "U1", "pin_id": "DIS"},
                            ],
                        },
                        {
                            "id": "TIMER_THRESHOLD",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "R2", "pin_id": "P2"},
                                {"component_id": "C1", "pin_id": "P1"},
                                {"component_id": "U1", "pin_id": "THR"},
                                {"component_id": "U1", "pin_id": "TRIG"},
                            ],
                        },
                        {
                            "id": "ASTABLE_OUT",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "U1", "pin_id": "OUT"},
                            ],
                        },
                        {
                            "id": "CV_BYPASS",
                            "type": "SIGNAL",
                            "connected_pins": [
                                {"component_id": "U1", "pin_id": "CV"},
                                {"component_id": "C3", "pin_id": "P1"},
                            ],
                        },
                    ],
                }
            },
            separators=(",", ":"),
        ),
    },
]


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_COMPONENT_KEYS = {"id", "type", "name", "footprint", "pins"}
_REQUIRED_PIN_KEYS = {"id", "type", "net"}
_REQUIRED_NET_KEYS = {"id", "type", "connected_pins"}


def _check_component(comp: Any, idx: int) -> None:
    """Validate a single component dict, raising NetlistValidationError on failure."""
    if not isinstance(comp, dict):
        raise NetlistValidationError(f"components[{idx}] is not a dict")
    missing = _REQUIRED_COMPONENT_KEYS - comp.keys()
    if missing:
        raise NetlistValidationError(
            f"components[{idx}] (id={comp.get('id', '?')!r}) missing keys: {missing}"
        )
    if not isinstance(comp["pins"], list) or len(comp["pins"]) == 0:
        raise NetlistValidationError(
            f"components[{idx}] (id={comp['id']!r}) pins must be a non-empty list"
        )
    for j, pin in enumerate(comp["pins"]):
        if not isinstance(pin, dict):
            raise NetlistValidationError(
                f"components[{idx}].pins[{j}] is not a dict"
            )
        missing_pin = _REQUIRED_PIN_KEYS - pin.keys()
        if missing_pin:
            raise NetlistValidationError(
                f"components[{idx}].pins[{j}] missing keys: {missing_pin}"
            )


def _check_net(net: Any, idx: int) -> None:
    """Validate a single net dict, raising NetlistValidationError on failure."""
    if not isinstance(net, dict):
        raise NetlistValidationError(f"nets[{idx}] is not a dict")
    missing = _REQUIRED_NET_KEYS - net.keys()
    if missing:
        raise NetlistValidationError(
            f"nets[{idx}] (id={net.get('id', '?')!r}) missing keys: {missing}"
        )
    if not isinstance(net["connected_pins"], list):
        raise NetlistValidationError(
            f"nets[{idx}] (id={net['id']!r}) connected_pins must be a list"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GroqTranslator:
    """Translates a plain-English circuit description into a validated JSON netlist.

    Uses the Groq API (groq.com) with the LLaMA 3.3 70B model.  The system
    prompt plus two few-shot examples constrain the model to emit only our
    netlist schema so that the output can be passed straight into Phase 1.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """Initialise the translator with Groq connection parameters.

        Args:
            api_key: Groq API key from console.groq.com.
            model:   Groq model identifier, e.g. "llama-3.3-70b-versatile".
        """
        self._model = model
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def translate(self, prompt: str) -> dict:
        """Translate a plain-English prompt into a validated netlist dict.

        Steps:
          1. Call the Groq API with system prompt + few-shot examples + user prompt.
          2. Extract JSON from the raw response (strips markdown fences if present).
          3. Validate the schema structure.
          4. Persist the result to netlists/generated/.
          5. Return the validated dict for Phase 1.

        Args:
            prompt: Natural-language circuit description from the user.

        Returns:
            Validated netlist dict with root key "netlist".

        Raises:
            GroqAPIError: When the API call fails.
            NetlistParseError: When the response is not valid JSON.
            NetlistValidationError: When the JSON does not match the schema.
        """
        raw = self._call_api(prompt)
        netlist = self._parse_response(raw)
        self._validate_schema(netlist)
        self._save_netlist(netlist, prompt)
        return netlist

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Issue the chat-completion request to Groq and return the raw text.

        Args:
            prompt: User circuit description.

        Returns:
            Raw string content from the model's first choice.

        Raises:
            GroqAPIError: On network, auth, timeout, or rate-limit failures.
        """
        messages = [
            {"role": "system", "content": NETLIST_SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
        except APITimeoutError as exc:
            raise GroqAPIError(f"Groq API request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise GroqAPIError(f"Groq API connection failed: {exc}") from exc
        except APIStatusError as exc:
            raise GroqAPIError(
                f"Groq API returned HTTP {exc.status_code}: {exc.message}"
            ) from exc

        return response.choices[0].message.content

    def _parse_response(self, raw: str) -> dict:
        """Extract and parse a JSON object from the raw model response.

        Strips markdown code fences (```json ... ```) if the model emitted them
        despite the system-prompt instructions, then parses the remaining text.

        Args:
            raw: Raw text from the model.

        Returns:
            Parsed Python dict.

        Raises:
            NetlistParseError: When no valid JSON can be extracted.
        """
        text = raw.strip()

        # Remove markdown code fences if present
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1)

        # Locate the outermost JSON object in case of surrounding prose
        brace_start = text.find("{")
        brace_end   = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise NetlistParseError(
                f"Could not decode JSON from model response: {exc}\n"
                f"--- raw response begin ---\n{raw[:500]}\n--- raw response end ---"
            ) from exc

    def _validate_schema(self, netlist: dict) -> bool:
        """Validate that *netlist* conforms to the expected structural schema.

        Args:
            netlist: Parsed dict to validate.

        Returns:
            True when validation passes.

        Raises:
            NetlistValidationError: On the first schema violation found.
        """
        if "netlist" not in netlist:
            raise NetlistValidationError(
                "Root key 'netlist' missing from response"
            )
        inner = netlist["netlist"]

        for key in ("metadata", "components", "nets"):
            if key not in inner:
                raise NetlistValidationError(
                    f"netlist.{key} is required but missing"
                )

        if not isinstance(inner["components"], list) or len(inner["components"]) == 0:
            raise NetlistValidationError(
                "netlist.components must be a non-empty list"
            )
        if not isinstance(inner["nets"], list) or len(inner["nets"]) == 0:
            raise NetlistValidationError(
                "netlist.nets must be a non-empty list"
            )

        for i, comp in enumerate(inner["components"]):
            _check_component(comp, i)
        for i, net in enumerate(inner["nets"]):
            _check_net(net, i)

        return True

    def _save_netlist(self, netlist: dict, prompt: str) -> Path:
        """Persist the validated netlist JSON to netlists/generated/.

        Filename format: {timestamp}_{design_name}.json

        Args:
            netlist: Validated netlist dict.
            prompt:  Original user prompt (fallback for filename).

        Returns:
            Path object pointing to the saved file.
        """
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        try:
            design_name = netlist["netlist"]["metadata"]["name"]
        except (KeyError, TypeError):
            design_name = "_".join(prompt.split()[:3])

        design_name = re.sub(r"[^\w\-]", "_", design_name)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename    = f"{timestamp}_{design_name}.json"
        output_path = _GENERATED_DIR / filename

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(netlist, fh, indent=2)

        print(f"[Phase 0] Netlist saved -> {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def run_phase0(prompt: str) -> dict:
    """Phase 0 pipeline entry-point: prompt -> validated netlist dict.

    Loads credentials from the .env file, instantiates GroqTranslator,
    calls the API, validates the result, and returns it ready for Phase 1.

    Args:
        prompt: Plain-English circuit description from the user.

    Returns:
        Validated netlist dict (root key "netlist") for Phase 1 consumption.

    Raises:
        EnvironmentError: When GROQ_API_KEY is not set.
        GroqAPIError: When the API call fails.
        NetlistParseError: When JSON extraction fails.
        NetlistValidationError: When the schema is violated.
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Create a .env file based on .env.example and add your Groq API key."
        )
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    print(f"[Phase 0] Translating prompt with model={model} ...")
    translator = GroqTranslator(api_key=api_key, model=model)
    netlist = translator.translate(prompt)

    print("\n[Phase 0] -- Validated Netlist ------------------------------------------")
    print(json.dumps(netlist, indent=2))
    print("[Phase 0] ----------------------------------------------------------------\n")

    comp_count = len(netlist["netlist"]["components"])
    net_count  = len(netlist["netlist"]["nets"])
    print(f"[Phase 0] Done. {comp_count} components, {net_count} nets.")
    return netlist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0 -- Translate a plain-English circuit description "
                    "into a JSON netlist via the Groq API (LLaMA 3.3 70B)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='E.g. --prompt "Connect an ESP32 to an LED via a resistor"',
    )
    args = parser.parse_args()

    try:
        run_phase0(args.prompt)
    except (GroqAPIError, NetlistParseError, NetlistValidationError, EnvironmentError) as exc:
        print(f"[Phase 0] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
