"""Causal features for one complete, time-ordered LiU recording.

Ratios are numerical relationships, not calibrated physical fault thresholds.
Rolling statistics include only current/past samples and never cross recordings.
"""
import numpy as np
import pandas as pd

RAW = ["Intercooler_pressure", "intake_manifold_pressure", "air_mass_flow",
       "engine_speed", "throttle_position", "wastegate_position", "injected_fuel_mass"]
REQUIRED = ["time", *RAW, "ambient_pressure"]


def build_features(recording, temporal=False):
    x = recording[REQUIRED].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Inputs must be finite.")
    t = x.time.to_numpy()
    if len(t) == 0 or (np.diff(t) <= 0).any():
        raise ValueError("Expected a nonempty recording with increasing time.")
    for c in ("ambient_pressure", "Intercooler_pressure", "engine_speed", "injected_fuel_mass"):
        if (x[c] <= 0).any():
            raise ValueError(f"{c} must be positive for ratio features.")
    f = x[RAW].copy()
    f["manifold_to_ambient"] = x.intake_manifold_pressure / x.ambient_pressure
    f["intercooler_to_ambient"] = x.Intercooler_pressure / x.ambient_pressure
    f["manifold_to_intercooler"] = x.intake_manifold_pressure / x.Intercooler_pressure
    f["air_to_fuel"] = x.air_mass_flow / x.injected_fuel_mass
    f["air_per_speed"] = x.air_mass_flow / x.engine_speed
    f["fuel_per_speed"] = x.injected_fuel_mass / x.engine_speed
    ready = pd.Series(True, index=f.index)
    if temporal:
        timed = f.copy()
        timed.index = pd.to_timedelta(t, unit="s")
        window = timed.rolling("1s", min_periods=10)
        for suffix, values in (("mean_1s", window.mean()), ("std_1s", window.std(ddof=0))):
            values.index = f.index
            for c in timed.columns:
                f[f"{c}_{suffix}"] = values[c]
        previous = np.searchsorted(t, t - 1.0, side="right") - 1
        previous_safe = np.maximum(previous, 0)
        elapsed = t - t[previous_safe]
        for c in RAW:
            f[f"{c}_change_per_second"] = np.divide(
                x[c].to_numpy() - x[c].to_numpy()[previous_safe], elapsed,
                out=np.full(len(t), np.nan), where=elapsed > 0)
        # After a sampling gap, wait for a fresh second of history.
        gaps = np.r_[True, np.diff(t) > 0.1]
        segment_start = np.maximum.accumulate(np.where(gaps, t, -np.inf))
        ready = pd.Series((t - segment_start >= 1.0) & (previous >= 0), index=f.index)
        ready &= f.notna().all(axis=1)
    return f, ready
