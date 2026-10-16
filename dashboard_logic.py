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


def recording_quality(frame: pd.DataFrame) -> dict:
    """Describe validated input without assuming physical units or filling gaps."""
    steps = np.diff(frame.time.to_numpy(dtype=float))
    return {
        "readings": len(frame),
        "duration_seconds": float(frame.time.iloc[-1] - frame.time.iloc[0]),
        "median_interval_seconds": float(np.median(steps)) if len(steps) else None,
        "gaps_over_0_1_seconds": int((steps > .1).sum()),
        "constant_signals": [c for c in REQUIRED[1:] if frame[c].nunique() == 1],
    }


def analysis_summary(rows: pd.DataFrame, filename: str, model_hash: str) -> dict:
    available = rows.prediction_available.fillna(False).astype(bool)
    flagged = available & rows.status.eq("anomaly_detected")
    scores = rows.loc[available, "anomaly_score"].dropna()
    return {
        "recording": filename,
        "model_sha256": model_hash,
        "readings": len(rows),
        "available_readings": int(available.sum()),
        "flagged_readings": int(flagged.sum()),
        "flagged_fraction_of_available": float(flagged.sum() / available.sum()) if available.any() else None,
        "peak_anomaly_score": float(scores.max()) if len(scores) else None,
        "status": "anomaly_detected" if flagged.any() else (
            "no_anomaly_detected" if available.any() else "insufficient_history"),
        "alert_intervals": alert_intervals(rows).to_dict(orient="records"),
        "interpretation": "Sensor disagreement, not a probability or component diagnosis. No anomaly detected does not establish engine health.",
    }


def inspection_guidance(rows: pd.DataFrame, synthetic: bool = False) -> dict:
    """Translate existing alerts into review guidance, never a fault probability."""
    available = rows.prediction_available.fillna(False).astype(bool)
    flagged = available & rows.status.eq("anomaly_detected")
    count, alerts = int(available.sum()), int(flagged.sum())
    if synthetic:
        title = "Demo only — no engine inspection decision"
        message = "This is the supplied synthetic recording. Its alerts only demonstrate the analysis workflow."
    elif not count:
        title = "Unable to assess — more usable data needed"
        message = "No readings had enough history for prediction. Upload a longer continuous recording."
    elif alerts:
        title = "Inspection recommended — review sensor alerts"
        message = "Unusual sensor relationships were detected. Verify the recording and have the flagged behaviour reviewed; a component fault is not confirmed."
    else:
        title = "No alert-based inspection trigger"
        message = "No alerts were found in the analyzed readings. This model can miss faults and cannot establish that the engine needs no inspection."
    return {"title": title, "message": message, "synthetic_demo": synthetic,
            "available_readings": count, "flagged_readings": alerts,
            "unavailable_readings": len(rows) - count,
            "alert_percentage": 100 * alerts / count if count else None,
            "fault_risk_percentage": None}


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
