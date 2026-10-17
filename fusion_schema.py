"""Shared simulator contract; labels describe synthetic rules, not fault physics."""
import numpy as np
import pandas as pd

SENSORS = {
    "dominant_frequency_hz": (40., 5000., 800., "Dominant frequency (Hz)"),
    "rms": (.01, .5, .15, "Audio RMS (amplitude)"),
    "accel_z_std": (.1, 5., 1.2, "Z-axis standard deviation (simulated units)"),
    "accel_z_spectral_centroid_hz": (50., 2000., 1500., "Vibration centroid (Hz)"),
    "engine_speed": (800., 6000., 2200., "Engine speed (simulated RPM)"),
    "throttle_position": (0., 100., 45., "Throttle (%)"),
    "injected_fuel_mass": (0., 50., 12., "Fuel mass (simulated mg)"),
    "air_mass_flow": (10., 500., 150., "Air mass flow (simulated units)"),
    "intake_manifold_pressure": (80000., 150000., 95000., "Intake pressure (simulated Pa)"),
    "Intercooler_pressure": (80000., 160000., 105000., "Intercooler pressure (simulated Pa)"),
    "intercooler_temperature": (290., 380., 310., "Intercooler temperature (simulated K)"),
    "wastegate_position": (0., 100., 65., "Wastegate (%)"),
    "ambient_pressure": (95000., 105000., 101325., "Ambient pressure (simulated Pa)"),
    "ambient_temperature": (270., 310., 295., "Ambient temperature (simulated K)"),
}
FEATURE_NAMES = list(SENSORS)
DEFAULTS = {key: spec[2] for key, spec in SENSORS.items()}
PRESETS = {
    "Nominal Operation": dict(DEFAULTS),
    "Mechanical Knocking": {**DEFAULTS, "accel_z_std": 4.7, "dominant_frequency_hz": 350., "intake_manifold_pressure": 145000.},
    "Valve Train Wear": {**DEFAULTS, "dominant_frequency_hz": 4800., "engine_speed": 5600., "intercooler_temperature": 375.},
    "Engine Misfire": {**DEFAULTS, "engine_speed": 900., "accel_z_std": 4.8, "air_mass_flow": 20.},
    "Turbocharger Stress": {**DEFAULTS, "throttle_position": 95., "wastegate_position": 5., "Intercooler_pressure": 155000.},
}


def assign_fault(row):
    if row['accel_z_std'] > 3.5 and row['dominant_frequency_hz'] < 1000 and row['intake_manifold_pressure'] > 120000:
        return "Mechanical Knocking"
    if row['dominant_frequency_hz'] > 4000 and row['engine_speed'] > 4500 and row['intercooler_temperature'] > 350:
        return "Valve Train Wear"
    if row['engine_speed'] < 1200 and row['accel_z_std'] > 4.0 and row['air_mass_flow'] < 50:
        return "Engine Misfire"
    if row['throttle_position'] > 80 and row['wastegate_position'] < 20 and row['Intercooler_pressure'] > 140000:
        return "Turbocharger Stress"
    return "Nominal Operation"


def input_frame(values):
    if set(values) != set(FEATURE_NAMES):
        raise ValueError("Fusion requires exactly the 14 defined sensor features.")
    frame = pd.DataFrame([values], columns=FEATURE_NAMES, dtype=float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Fusion inputs must be finite numbers.")
    for key, (low, high, _, _) in SENSORS.items():
        if not low <= frame.at[0, key] <= high:
            raise ValueError(f"{key} is outside the simulator range.")
    return frame
