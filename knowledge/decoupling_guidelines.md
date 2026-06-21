# Decoupling and Bypass Capacitor Guidelines

Decoupling capacitors supply transient current to an IC so the local supply
voltage stays stable when the chip switches.

## Rules of thumb
- Place one 100 nF (0.1 µF) ceramic decoupling capacitor per power pin of every
  IC, MCU, ADC, or op-amp. Place it within 2 grid cells of the power pin — the
  shorter the loop, the lower the parasitic inductance.
- Add one bulk capacitor (1 µF to 10 µF) per supply rail, near the regulator or
  the board's power entry, to handle lower-frequency current demand.
- For high-speed digital ICs, a 10 nF + 100 nF pair gives a wider effective
  frequency range than a single value.

## Why placement matters
A decoupling cap placed far from the power pin forms a larger current loop with
more inductance, which defeats its purpose. The 2-cell rule in this engine
enforces that proximity.

## Common mistakes
- One shared cap for several ICs: each IC needs its own local cap.
- Electrolytic only: electrolytics have high ESR/ESL and are poor at high
  frequency — always pair with a ceramic 100 nF.
