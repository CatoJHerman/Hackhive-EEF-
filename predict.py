"""Run the frozen experimental LiU consistency monitor on one CSV recording.

Use original LiU signal scales. No fault labels or filename-derived inputs are used.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from liu_residual_model import TARGETS, score_detector


def predict_recording(frame, bundle):
    required = sorted({"time", *TARGETS, *(c for cols in TARGETS.values() for c in cols)})
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = frame[required].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    if len(data) == 0 or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Expected nonempty finite numeric readings.")
    if (data.time < 0).any() or (np.diff(data.time) <= 0).any():
        raise ValueError("Time must be nonnegative and strictly increasing.")
    data["recording_id"] = "input_recording"
    scored = score_detector(bundle, data)
    out = data[["time"]].join(scored)
    out["status"] = "insufficient_history"
    ready = out.prediction_available
    out.loc[ready, "status"] = "no_anomaly_detected"
    out.loc[ready & out.prediction.eq(1).fillna(False), "status"] = "anomaly_detected"
    for target, inputs in TARGETS.items():
        out[target + "_actual"] = data[target]
        out[target + "_predicted"] = np.exp(bundle["models"][target].predict(data[inputs]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("outputs/liu_residuals/residual_model.joblib"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists; choose a new path.")
    result = predict_recording(pd.read_csv(args.input), joblib.load(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.status.value_counts().to_string())
    print("Experimental consistency monitor; no anomaly detected does not establish engine health.")


if __name__ == "__main__":
    main()
