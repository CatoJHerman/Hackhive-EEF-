"""Normal-behaviour residual detector; all calibration uses training data only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = {
    "intake_manifold_pressure": ["air_mass_flow", "engine_speed", "intercooler_temperature", "throttle_position", "injected_fuel_mass"],
    "Intercooler_pressure": ["intake_manifold_pressure", "air_mass_flow", "engine_speed", "throttle_position", "wastegate_position"],
    "air_mass_flow": ["injected_fuel_mass", "engine_speed", "throttle_position", "intake_manifold_pressure", "Intercooler_pressure"],
}
SETTINGS = dict(max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
                l2_regularization=2.0, early_stopping=False, random_state=42)


def residuals(models, data):
    result = pd.DataFrame(index=data.index)
    for target, inputs in TARGETS.items():
        actual = data[target].to_numpy(float)
        if not np.isfinite(actual).all() or (actual <= 0).any():
            raise ValueError(f"Expected positive finite {target}")
        predicted_log = models[target].predict(data[inputs])
        result[target] = np.abs(np.log(actual) - predicted_log)
    return result


def smooth_by_recording(values, data):
    output = pd.DataFrame(np.nan, index=data.index, columns=values.columns)
    for _, group in data.groupby("recording_id", sort=False):
        t = group.time.to_numpy()
        if (np.diff(t) <= 0).any():
            raise ValueError("Each recording must be time ordered.")
        v = values.loc[group.index].copy()
        v.index = pd.to_timedelta(t, unit="s")
        smoothed = v.rolling("1s", min_periods=10).mean().to_numpy(copy=True)
        starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(t) > .1], t, -np.inf))
        smoothed[t - starts < 1] = np.nan
        output.loc[group.index] = smoothed
    return output


def fit_detector(data):
    # Four 30-second blocks for fitting, one for calibration; 2-second boundary guards.
    # This is internal training calibration, NOT an independent performance test.
    phase = data.time % 30
    block = (data.time // 30).astype(int)
    normal = data.training_eligible.eq(1) & data.fault_present.eq(0)
    guard = (phase >= 2) & (phase < 28)
    # Early normal portions end before the fifth block. Reserve 62-88 s
    # instead, consistently in every recording, so normal-only early runs calibrate too.
    fit = normal & guard & (block % 5 != 2)
    calibration = normal & guard & (block % 5 == 2)
    if fit.sum() < 100 or calibration.sum() < 100:
        raise ValueError("Insufficient normal fitting/calibration data.")
    models = {}
    for target, inputs in TARGETS.items():
        models[target] = HistGradientBoostingRegressor(**SETTINGS).fit(
            data.loc[fit, inputs], np.log(data.loc[fit, target]))
    smoothed = smooth_by_recording(residuals(models, data), data)
    mask = calibration & smoothed.notna().all(axis=1)
    scales = smoothed.loc[mask].quantile(.5).clip(lower=1e-6)
    score = smoothed.div(scales).max(axis=1, skipna=False)
    threshold = float(score.loc[mask].quantile(.99))
    return {"models": models, "scales": scales, "threshold": threshold,
            "fit_rows": int(fit.sum()), "calibration_rows": int(mask.sum()),
            "settings": SETTINGS, "features_by_target": TARGETS,
            "status": "Experimental normal-behaviour residual detector; not a confirmed diagnosis.",
            "calibration_policy": "Normal training rows only; 30s blocks numbered modulo 5 == 2 reserved for calibration; 2s block guards. Joint residual-score 99th percentile threshold.",
            "smoothing_seconds": 1, "units": "Original LiU CSV scale"}


def score_detector(bundle, data):
    smoothed = smooth_by_recording(residuals(bundle["models"], data), data)
    score = smoothed.div(bundle["scales"]).max(axis=1, skipna=False)
    prediction = (score > bundle["threshold"]).astype("Int64")
    prediction[score.isna()] = pd.NA
    return pd.DataFrame({"anomaly_score": score / bundle["threshold"],
                         "prediction_available": score.notna(),
                         "prediction": prediction}, index=data.index)
