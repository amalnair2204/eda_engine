"""
Component Pin Library — static datasheet-sourced definitions.

Provides exact pin counts, names, sides, and types for common components.
Used by Phase 1 InitialPlacer to position pins on the component perimeter.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass
class PinDef:
    """One physical pin from a component datasheet."""
    number: int       # physical pin number (1-indexed)
    name: str         # official name e.g. "GPIO2", "VDD", "CH0"
    side: str         # "left" | "right" | "top" | "bottom"
    pin_type: str     # OUTPUT | INPUT | PASSIVE | POWER | BIDIRECTIONAL


@dataclass
class ComponentDef:
    """Full component definition sourced from official datasheet."""
    canonical_names: list[str]   # all name variants Groq might use
    pin_defs: list[PinDef]
    footprint_w: int
    footprint_h: int
    package: str


# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

_ESP32 = ComponentDef(
    canonical_names=["esp32", "esp32-wroom", "esp32-wroom-32", "wroom-32", "wroom32", "esp32wroom"],
    pin_defs=[
        # Left side (top -> bottom), pins 1-19
        PinDef( 1, "GND",    "left", "POWER"),
        PinDef( 2, "3V3",    "left", "POWER"),
        PinDef( 3, "EN",     "left", "INPUT"),
        PinDef( 4, "VP",     "left", "INPUT"),
        PinDef( 5, "VN",     "left", "INPUT"),
        PinDef( 6, "GPIO34", "left", "INPUT"),
        PinDef( 7, "GPIO35", "left", "INPUT"),
        PinDef( 8, "GPIO32", "left", "BIDIRECTIONAL"),
        PinDef( 9, "GPIO33", "left", "BIDIRECTIONAL"),
        PinDef(10, "GPIO25", "left", "BIDIRECTIONAL"),
        PinDef(11, "GPIO26", "left", "BIDIRECTIONAL"),
        PinDef(12, "GPIO27", "left", "BIDIRECTIONAL"),
        PinDef(13, "GPIO14", "left", "BIDIRECTIONAL"),
        PinDef(14, "GPIO12", "left", "BIDIRECTIONAL"),
        PinDef(15, "GND2",   "left", "POWER"),
        PinDef(16, "GPIO13", "left", "BIDIRECTIONAL"),
        PinDef(17, "SD2",    "left", "BIDIRECTIONAL"),
        PinDef(18, "SD3",    "left", "BIDIRECTIONAL"),
        PinDef(19, "CMD",    "left", "BIDIRECTIONAL"),
        # Right side (top -> bottom), pins 20-38
        PinDef(20, "GND3",   "right", "POWER"),
        PinDef(21, "GPIO23", "right", "BIDIRECTIONAL"),
        PinDef(22, "GPIO22", "right", "BIDIRECTIONAL"),
        PinDef(23, "TXD0",   "right", "OUTPUT"),
        PinDef(24, "RXD0",   "right", "INPUT"),
        PinDef(25, "GPIO21", "right", "BIDIRECTIONAL"),
        PinDef(26, "GND4",   "right", "POWER"),
        PinDef(27, "GPIO19", "right", "BIDIRECTIONAL"),
        PinDef(28, "GPIO18", "right", "BIDIRECTIONAL"),
        PinDef(29, "GPIO5",  "right", "BIDIRECTIONAL"),
        PinDef(30, "GPIO17", "right", "BIDIRECTIONAL"),
        PinDef(31, "GPIO16", "right", "BIDIRECTIONAL"),
        PinDef(32, "GPIO4",  "right", "BIDIRECTIONAL"),
        PinDef(33, "GPIO0",  "right", "BIDIRECTIONAL"),
        PinDef(34, "GPIO2",  "right", "BIDIRECTIONAL"),
        PinDef(35, "GPIO15", "right", "BIDIRECTIONAL"),
        PinDef(36, "SD1",    "right", "BIDIRECTIONAL"),
        PinDef(37, "SD0",    "right", "BIDIRECTIONAL"),
        PinDef(38, "CLK",    "right", "BIDIRECTIONAL"),
    ],
    footprint_w=8,
    footprint_h=20,
    package="MODULE",
)

_MCP3008 = ComponentDef(
    canonical_names=["mcp3008", "mcp3004", "mcp3208"],
    pin_defs=[
        # Left side (top -> bottom)
        PinDef(1, "CH0",  "left", "INPUT"),
        PinDef(2, "CH1",  "left", "INPUT"),
        PinDef(3, "CH2",  "left", "INPUT"),
        PinDef(4, "CH3",  "left", "INPUT"),
        PinDef(5, "CH4",  "left", "INPUT"),
        PinDef(6, "CH5",  "left", "INPUT"),
        PinDef(7, "CH6",  "left", "INPUT"),
        PinDef(8, "CH7",  "left", "INPUT"),
        # Right side (bottom -> top: pin 9 at bottom, pin 16 at top)
        PinDef( 9, "DGND", "right", "POWER"),
        PinDef(10, "CS",   "right", "INPUT"),
        PinDef(11, "DIN",  "right", "INPUT"),
        PinDef(12, "DOUT", "right", "OUTPUT"),
        PinDef(13, "CLK",  "right", "INPUT"),
        PinDef(14, "AGND", "right", "POWER"),
        PinDef(15, "VREF", "right", "POWER"),
        PinDef(16, "VDD",  "right", "POWER"),
    ],
    footprint_w=4,
    footprint_h=9,
    package="DIP16",
)

_LM35 = ComponentDef(
    canonical_names=["lm35", "lm35dz", "lm35d", "lm35cz"],
    pin_defs=[
        PinDef(1, "VS",   "right", "POWER"),
        PinDef(2, "VOUT", "right", "OUTPUT"),
        PinDef(3, "GND",  "left",  "POWER"),
    ],
    footprint_w=2,
    footprint_h=2,
    package="TO-92",
)

_NE555 = ComponentDef(
    canonical_names=["ne555", "555", "lm555", "na555", "ne555p"],
    pin_defs=[
        # Left side (top -> bottom)
        PinDef(1, "GND",   "left", "POWER"),
        PinDef(2, "TRIG",  "left", "INPUT"),
        PinDef(3, "OUT",   "left", "OUTPUT"),
        PinDef(4, "RESET", "left", "INPUT"),
        # Right side (bottom -> top)
        PinDef(5, "VCC",   "right", "POWER"),
        PinDef(6, "DIS",   "right", "OUTPUT"),
        PinDef(7, "THR",   "right", "INPUT"),
        PinDef(8, "CTRL",  "right", "INPUT"),
    ],
    footprint_w=3,
    footprint_h=5,
    package="DIP8",
)

_ATMEGA328P = ComponentDef(
    canonical_names=["atmega328p", "atmega328", "atmega", "arduino", "arduino-uno"],
    pin_defs=[
        # Left side (top -> bottom)
        PinDef( 1, "RESET", "left", "INPUT"),
        PinDef( 2, "RXD",   "left", "INPUT"),
        PinDef( 3, "TXD",   "left", "OUTPUT"),
        PinDef( 4, "INT0",  "left", "BIDIRECTIONAL"),
        PinDef( 5, "INT1",  "left", "BIDIRECTIONAL"),
        PinDef( 6, "PD4",   "left", "BIDIRECTIONAL"),
        PinDef( 7, "VCC",   "left", "POWER"),
        PinDef( 8, "GND",   "left", "POWER"),
        PinDef( 9, "XTAL1", "left", "INPUT"),
        PinDef(10, "XTAL2", "left", "OUTPUT"),
        PinDef(11, "PD5",   "left", "BIDIRECTIONAL"),
        PinDef(12, "PD6",   "left", "BIDIRECTIONAL"),
        PinDef(13, "PD7",   "left", "BIDIRECTIONAL"),
        PinDef(14, "PB0",   "left", "BIDIRECTIONAL"),
        # Right side (bottom -> top)
        PinDef(15, "PB1",  "right", "BIDIRECTIONAL"),
        PinDef(16, "SS",   "right", "INPUT"),
        PinDef(17, "MOSI", "right", "BIDIRECTIONAL"),
        PinDef(18, "MISO", "right", "BIDIRECTIONAL"),
        PinDef(19, "SCK",  "right", "BIDIRECTIONAL"),
        PinDef(20, "AVCC", "right", "POWER"),
        PinDef(21, "AREF", "right", "PASSIVE"),
        PinDef(22, "GND2", "right", "POWER"),
        PinDef(23, "A0",   "right", "INPUT"),
        PinDef(24, "A1",   "right", "INPUT"),
        PinDef(25, "A2",   "right", "INPUT"),
        PinDef(26, "A3",   "right", "INPUT"),
        PinDef(27, "SDA",  "right", "BIDIRECTIONAL"),
        PinDef(28, "SCL",  "right", "BIDIRECTIONAL"),
    ],
    footprint_w=5,
    footprint_h=15,
    package="DIP28",
)

_LM741 = ComponentDef(
    canonical_names=["lm741", "ua741", "op-amp", "opamp", "lm741cn"],
    pin_defs=[
        # Left side (top -> bottom)
        PinDef(1, "OFFSET_N1", "left", "PASSIVE"),
        PinDef(2, "IN-",       "left", "INPUT"),
        PinDef(3, "IN+",       "left", "INPUT"),
        PinDef(4, "V-",        "left", "POWER"),
        # Right side (bottom -> top)
        PinDef(5, "OFFSET_N2", "right", "PASSIVE"),
        PinDef(6, "OUT",       "right", "OUTPUT"),
        PinDef(7, "V+",        "right", "POWER"),
        PinDef(8, "NC",        "right", "PASSIVE"),
    ],
    footprint_w=3,
    footprint_h=5,
    package="DIP8",
)

_RESISTOR = ComponentDef(
    canonical_names=["resistor", "res", "r_"],
    pin_defs=[
        PinDef(1, "P1", "left",  "PASSIVE"),
        PinDef(2, "P2", "right", "PASSIVE"),
    ],
    footprint_w=2,
    footprint_h=1,
    package="AXIAL",
)

_CAPACITOR = ComponentDef(
    canonical_names=["capacitor", "cap", "c_decoupling", "c_bypass", "c_"],
    pin_defs=[
        PinDef(1, "P1", "left",  "PASSIVE"),
        PinDef(2, "P2", "right", "PASSIVE"),
    ],
    footprint_w=1,
    footprint_h=1,
    package="SMD",
)

_LED = ComponentDef(
    canonical_names=["led", "led_status", "led_"],
    pin_defs=[
        PinDef(1, "ANODE",   "left",  "INPUT"),
        PinDef(2, "CATHODE", "right", "PASSIVE"),
    ],
    footprint_w=2,
    footprint_h=2,
    package="LED",
)


COMPONENT_LIBRARY: list[ComponentDef] = [
    _ESP32,
    _MCP3008,
    _LM35,
    _NE555,
    _ATMEGA328P,
    _LM741,
    _RESISTOR,
    _CAPACITOR,
    _LED,
]


def _make_generic_ic(pin_count: int) -> ComponentDef:
    """Fallback: split pins evenly left/right for unknown ICs."""
    left_n  = ceil(pin_count / 2)
    right_n = pin_count - left_n
    pins: list[PinDef] = []
    for i in range(left_n):
        pins.append(PinDef(i + 1, f"P{i+1}", "left", "PASSIVE"))
    for i in range(right_n):
        pnum = left_n + i + 1
        pins.append(PinDef(pnum, f"P{pnum}", "right", "PASSIVE"))
    return ComponentDef(
        canonical_names=["ic"],
        pin_defs=pins,
        footprint_w=4,
        footprint_h=ceil(pin_count / 2) + 1,
        package="DIP",
    )


def lookup(name: str, pin_count: int = 8) -> ComponentDef:
    """Case-insensitive partial-string lookup by component name.

    Args:
        name:      Component name or type (e.g. "ESP32", "RESISTOR", "MCP3008").
        pin_count: Fallback pin count for generic IC if no match found.

    Returns:
        ComponentDef from library, or a generic IC fallback.
    """
    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "").replace(" ", "")

    name_n = _norm(name)

    # 1. Exact match against canonical names
    for comp_def in COMPONENT_LIBRARY:
        for canon in comp_def.canonical_names:
            if _norm(canon) == name_n:
                return comp_def

    # 2. Substring match (name contains a canonical name or vice versa)
    for comp_def in COMPONENT_LIBRARY:
        for canon in comp_def.canonical_names:
            canon_n = _norm(canon)
            if canon_n in name_n or name_n in canon_n:
                return comp_def

    return _make_generic_ic(max(pin_count, 2))
