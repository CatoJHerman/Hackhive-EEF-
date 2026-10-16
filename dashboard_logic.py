"""Small, testable helpers for the upload-driven Streamlit dashboard."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from liu_residual_model import TARGETS

REQUIRED = ["time", "Intercooler_pressure", "intake_manifold_pressure",
            "intercooler_temperature", "air_mass_flow", "engine_speed",
            "throttle_position", "wastegate_position", "injected_fuel_mass"]
OPTIONAL = ["ambient_pressure", "ambient_temperature"]
FEATURES = ["rms", "peak_amplitude", "crest_factor", "dominant_frequency_hz"]


def read_telemetry(content: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV has no readings.")
    for column in REQUIRED + [name for name in OPTIONAL if name in frame.columns]:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must contain numeric readings; no values are filled or converted.") from exc
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} has missing or non-finite readings; no values are filled.")
        frame[column] = values
    if (frame.time < 0).any() or (np.diff(frame.time.to_numpy()) <= 0).any():
        raise ValueError("Time must be nonnegative and strictly increasing in file order. The recording was not reordered.")
    for target in TARGETS:
        if (frame[target] <= 0).any():
            raise ValueError(f"{target} must be positive for the frozen model's log-scale prediction.")
    return frame


def alert_intervals(rows: pd.DataFrame, max_gap_seconds: float = .1) -> pd.DataFrame:
    """Consecutive flagged samples, split by unavailable rows or time gaps."""
    intervals = []
    start = end = peak = None
    previous_time = None
    for row in rows.itertuples(index=False):
        t = float(row.time)
        flagged = bool(row.prediction_available) and row.status == "anomaly_detected"
        if start is not None and (not flagged or t - previous_time > max_gap_seconds):
            intervals.append((start, end, peak))
            start = end = peak = None
        if flagged:
            if start is None:
                start, peak = t, float(row.anomaly_score)
            end = t
            peak = max(peak, float(row.anomaly_score))
        previous_time = t
    if start is not None:
        intervals.append((start, end, peak))
    return pd.DataFrame(intervals, columns=["start_seconds", "end_seconds", "max_anomaly_score"])
