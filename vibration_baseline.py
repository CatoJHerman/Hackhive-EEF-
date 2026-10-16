"""Strict waveform loading and descriptive per-axis reference comparison."""
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd

from scipy.signal import stft, get_window
from scipy.stats import kurtosis

BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 5000)]
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


def feature_for_axis(x: np.ndarray, fs: float, axis: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return {f"{axis}_{k}": np.nan for k in ["mean", "std", "rms", "peak_abs", "peak_to_peak", "kurtosis_pearson", "crest_factor", "dominant_frequency_hz", "spectral_centroid_hz"]}

    # Detrending by subtracting the mean is a baseline-removal step, not physical calibration.
    xc = x - np.mean(x)
    rms = float(np.sqrt(np.mean(xc ** 2)))
    peak_abs = float(np.max(np.abs(xc)))
    window = get_window("hann", len(xc), fftbins=True)
    spectrum = np.abs(np.fft.rfft(xc * window))
    freq = np.fft.rfftfreq(len(xc), d=1.0 / fs)
    if len(spectrum):
        spectrum[0] = 0.0
    power = spectrum ** 2
    total_power = float(power.sum())
    dominant = float(freq[int(np.argmax(spectrum))]) if np.any(spectrum) else 0.0
    centroid = float(np.sum(freq * power) / total_power) if total_power > 0 else 0.0

    out = {
        f"{axis}_mean": float(np.mean(x)),
        f"{axis}_std": float(np.std(xc)),
        f"{axis}_rms": rms,
        f"{axis}_peak_abs": peak_abs,
        f"{axis}_peak_to_peak": float(np.ptp(xc)),
        f"{axis}_kurtosis_pearson": float(kurtosis(xc, fisher=False, bias=False)),
        f"{axis}_crest_factor": float(peak_abs / rms) if rms else 0.0,
        f"{axis}_dominant_frequency_hz": dominant,
        f"{axis}_spectral_centroid_hz": centroid,
    }
    for lo, hi in BANDS:
        mask = (freq >= lo) & (freq < hi)
        frac = float(power[mask].sum() / total_power) if total_power > 0 else 0.0
        out[f"{axis}_band_{lo}_{hi}hz_power_fraction"] = frac

    nperseg = min(1024, len(xc))
    noverlap = min(int(nperseg * 0.75), nperseg - 1)
    _, _, z = stft(xc, fs=fs, window="hann", nperseg=nperseg,
                   noverlap=noverlap, boundary=None, padded=False)
    stft_power = np.abs(z) ** 2
    out[f"{axis}_stft_mean_power"] = float(stft_power.mean()) if stft_power.size else 0.0
    out[f"{axis}_stft_max_power"] = float(stft_power.max()) if stft_power.size else 0.0
    return out



def window_features(data, rate):
    records = []
    for start in range(0, len(data) - rate + 1, rate):
        window = data.iloc[start:start + rate]
        row = {"start_seconds": float(window.time_seconds.iloc[0]),
               "end_seconds": float(window.time_seconds.iloc[0] + 1), "sampling_rate_hz": rate}
        for axis in AXES:
            signal = window[axis].to_numpy(float)
            row.update(feature_for_axis(signal, rate, axis))
            if np.ptp(signal) == 0:
                row[axis + "_kurtosis_pearson"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


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
