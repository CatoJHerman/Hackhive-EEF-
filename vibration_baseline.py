"""Strict waveform loading and descriptive per-axis reference comparison."""
import hashlib
import io
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parent / "vibration/engineguard_quickstart(ai)"
AXES = ("accel_x", "accel_y", "accel_z")
METRICS = ("std", "rms", "peak_abs", "peak_to_peak", "crest_factor",
           "kurtosis_pearson", "dominant_frequency_hz", "spectral_centroid_hz")
COMPARISON = [f"{axis}_{metric}" for axis in AXES for metric in METRICS]
MAX_CSV_BYTES = 18_000_000


def read_signal(content):
    if not content or len(content) > MAX_CSV_BYTES:
        raise ValueError("Vibration CSV must be nonempty and below 18 MB.")
    try:
        data = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read vibration CSV: {exc}") from exc
    required = ["time_seconds", "sampling_rate_hz", *AXES]
    if not set(required).issubset(data):
        raise ValueError("Expected time_seconds, sampling_rate_hz, accel_x, accel_y, accel_z. "
                         "PAK exports are not verified raw three-axis waveforms and cannot be relabeled automatically.")
    try:
        values = data[required].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Time, sample rate and acceleration must be numeric; no readings are filled.") from exc
    if len(values) < 8 or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Need at least eight finite numeric samples; no readings are filled or removed.")
    rates = values.sampling_rate_hz.unique()
    if len(rates) != 1 or rates[0] < 8 or rates[0] != round(rates[0]):
        raise ValueError("Provide one integer sampling rate of at least 8 Hz.")
    rate = int(rates[0])
    steps = np.diff(values.time_seconds.to_numpy(float))
    if (steps <= 0).any() or not np.allclose(steps, 1/rate, rtol=.001, atol=1e-9):
        raise ValueError("Timestamps must increase uniformly at the declared sample rate; duplicate or uneven timestamps are rejected.")
    if len(values) < rate:
        raise ValueError("At least one complete second is required.")
    return values, rate


def window_features(data, rate):
    # Reuse the supplied package's feature definitions, without its classifier/server.
    source = PACKAGE / "vibration_analysis.py"
    if not source.is_file():
        raise ValueError(f"Feature extraction module is missing: {source}")
    spec = importlib.util.spec_from_file_location("engineguard_signal_features", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for start in range(0, len(data) - rate + 1, rate):
        window = data.iloc[start:start + rate]
        row = {"start_seconds": float(window.time_seconds.iloc[0]),
               "end_seconds": float(window.time_seconds.iloc[0] + 1), "sampling_rate_hz": rate}
        for axis in AXES:
            signal = window[axis].to_numpy(float)
            row.update(module.feature_for_axis(signal, rate, axis))
            if np.ptp(signal) == 0:
                row[axis + "_kurtosis_pearson"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def demo_reference():
    """Synthetic normal examples across speeds; never label these as measured idle."""
    frame = pd.read_csv(PACKAGE / "training_features.csv")
    ids = frame.run_id.str.split("_").str[-1].astype(int)
    return frame.loc[frame.condition_label.eq("normal") & ids.le(8)].copy()


def reference_recordings(content, filename):
    """One feature-median vector per uploaded CSV; ZIP members remain separate recordings."""
    if filename.lower().endswith(".csv"):
        members = [(filename, content)]
    elif filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                info = [m for m in archive.infolist() if m.filename.lower().endswith(".csv") and not m.is_dir()]
                if not 1 <= len(info) <= 100 or sum(m.file_size for m in info) > 100_000_000:
                    raise ValueError("Reference ZIP needs 1–100 CSV recordings and at most 100 MB uncompressed.")
                if any(m.file_size > MAX_CSV_BYTES for m in info):
                    raise ValueError("Every reference CSV must be below 18 MB.")
                members = [(m.filename, archive.read(m)) for m in info]
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError(f"Could not read reference ZIP: {exc}") from exc
    else:
        raise ValueError("Upload a reference CSV or a ZIP containing CSV recordings.")
    rows, hashes = [], set()
    for name, raw in members:
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate reference recording content: {name}")
        hashes.add(digest)
        try:
            data, rate = read_signal(raw)
            features = window_features(data, rate)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
        row = features[COMPARISON].median(numeric_only=True).to_dict()
        row.update({"reference_recording": name, "sampling_rate_hz": rate,
                    "reference_windows": len(features)})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_reference(features, reference):
    if len(reference) == 0:
        raise ValueError("No reference recordings supplied.")
    if set(features.sampling_rate_hz) != set(reference.sampling_rate_hz):
        raise ValueError("Input and reference sample rates must match; no resampling is guessed.")
    can_compare_range = len(reference) >= 3
    records = []
    for key in COMPARISON:
        ref = pd.to_numeric(reference[key], errors="coerce").to_numpy(float)
        valid = ref[np.isfinite(ref)]
        if not len(valid):
            continue
        low, high = float(valid.min()), float(valid.max())
        axis, metric = key[:7], key[8:]
        uploaded = pd.to_numeric(features[key], errors="coerce").to_numpy(float)
        available = uploaded[np.isfinite(uploaded)]
        value = float(np.median(available)) if len(available) else float("nan")
        status = "unavailable" if not np.isfinite(value) else (
            "descriptive only" if not can_compare_range or len(valid) < 3 else
            "below reference" if value < low else "above reference" if value > high else "within reference")
        records.append({"axis": axis, "feature": metric, "uploaded_recording_median": value,
                        "reference_min": low, "reference_max": high,
                        "reference_recordings_with_feature": len(valid), "comparison": status})
    if not records:
        raise ValueError("Reference has no usable comparison features.")
    return pd.DataFrame(records)
