# ESP32-WROOM-32 — Power Characteristics (reference summary)

The ESP32-WROOM-32 module operates from a 3.3 V supply (recommended operating
range 3.0 V to 3.6 V). It is NOT 5 V tolerant on its IO pins.

## Current consumption
- Wi-Fi TX peak (802.11b, +20 dBm): up to ~240 mA, transient peaks ~500 mA.
- Wi-Fi RX / active without TX: ~80–100 mA.
- Modem-sleep (CPU active, radio off): ~20–30 mA.
- Light-sleep: ~0.8 mA.
- Deep-sleep: ~10 µA (RTC only).

## Power supply guidance
- Provide a bulk capacitor of at least 10 µF on the 3.3 V rail close to the module.
- Add a 100 nF decoupling capacitor on each VDD pin, within 2 grid cells of the pin.
- Brown-out resets are common when the supply sags during Wi-Fi TX peaks; size the
  regulator for at least 500 mA to handle transients.

## Low-power alternatives
- ESP32-C3: single RISC-V core, Wi-Fi TX peak ~330 mA but deep-sleep ~5 µA, lower
  active current (~20 mA modem-sleep). Good drop-in for lower average power.
- ESP8266 (ESP-12F): cheaper, Wi-Fi TX peak ~170 mA, deep-sleep ~20 µA, but no
  Bluetooth and fewer GPIO.
