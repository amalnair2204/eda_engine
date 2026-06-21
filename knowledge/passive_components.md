# Passive Component Reference — Resistors, LEDs, Current Limiting

## LED current-limiting resistor
For an LED driven from a GPIO, size the series resistor as:

    R = (V_supply - V_forward) / I_led

Example: 3.3 V GPIO, red LED V_forward = 2.0 V, target I = 10 mA:
    R = (3.3 - 2.0) / 0.010 = 130 ohm  → use 150 ohm (nearest standard, safe).

A 330 ohm resistor on a 3.3 V rail gives about 3.9 mA — dim but very safe and a
common default. Lower resistance = brighter but more current and power.

## Resistor power rating
Power dissipated: P = I^2 * R. A 330 ohm resistor passing 4 mA dissipates
~5 mW, far below a 0.25 W (250 mW) rating, so a standard 1/4 W part is fine.

## Typical power draw of small parts
- Standard indicator LED: 2–20 mA.
- Pull-up/pull-down resistor (10 kohm on 3.3 V): ~0.33 mA.
- Decoupling capacitor: negligible steady-state current (leakage only).

In a small MCU + LED design the microcontroller almost always dominates total
power; passives are negligible by comparison.
