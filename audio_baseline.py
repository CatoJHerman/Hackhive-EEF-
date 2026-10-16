"""Descriptive comparisons against saved one-second idle audio features."""
import numpy as np
import pandas as pd

FEATURES = ["rms", "peak_amplitude", "crest_factor", "dominant_frequency_hz"]


def compare_idle(upload: pd.DataFrame, baseline: pd.DataFrame):
    required = FEATURES + ["sample_rate", "start_seconds", "end_seconds"]
    for name, frame in [("Upload", upload), ("Idle baseline", baseline)]:
        if frame.empty or not set(required).issubset(frame.columns):
            raise ValueError(f"{name} is empty or missing feature/window columns.")
        values = frame[required].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (frame[FEATURES] < 0).any().any():
            raise ValueError(f"{name} contains invalid feature values.")
        if not np.allclose(frame.end_seconds - frame.start_seconds, 1):
            raise ValueError(f"{name} must use complete one-second windows.")
    rates = upload.sample_rate.unique()
    if len(rates) != 1 or rates[0] <= 0:
        raise ValueError("Upload must have one positive sample rate.")
    reference = baseline.loc[baseline.sample_rate.eq(rates[0])]
    if len(reference) < 20:
        raise ValueError(f"Need at least 20 idle windows at {rates[0]:g} Hz. "
                         "Use a recording at the baseline sample rate; changing a file header is not resampling.")
    lower = reference[FEATURES].quantile(.05)
    upper = reference[FEATURES].quantile(.95)
    median = reference[FEATURES].median()
    outside = upload[FEATURES].lt(lower) | upload[FEATURES].gt(upper)
    reference_outside = reference[FEATURES].lt(lower) | reference[FEATURES].gt(upper)
    windows = upload.copy()
    for feature in FEATURES:
        windows[feature + "_comparison"] = np.where(upload[feature] < lower[feature], "below idle range",
                                                     np.where(upload[feature] > upper[feature], "above idle range", "within idle range"))
    windows["differs_from_idle"] = outside.any(axis=1)
    windows["different_features"] = outside.apply(lambda row: ", ".join(row.index[row]), axis=1)
    table = pd.DataFrame({"feature": FEATURES, "idle_low": lower.values,
                          "idle_median": median.values, "idle_high": upper.values,
                          "uploaded_median": upload[FEATURES].median().values,
                          "windows_outside_percent": outside.mean().values * 100})
    summary = {"baseline_windows": len(reference), "sample_rate": int(rates[0]),
               "uploaded_windows": len(upload), "different_windows": int(outside.any(axis=1).sum()),
               "different_windows_percent": float(outside.any(axis=1).mean() * 100),
               "idle_reference_outside_percent": float(reference_outside.any(axis=1).mean() * 100),
               "method": "Outside the idle feature 5th–95th percentile range in at least one feature.",
               "interpretation": "Descriptive difference from idle, not fault probability or diagnosis."}
    return table, windows, summary
