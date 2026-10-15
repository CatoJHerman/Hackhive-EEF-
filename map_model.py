import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


HEALTHY_GROUPS = (1, 7)
FEATURES = ("LOAD_PCT", "ECT", "RPM", "VSS", "IAT", "MAF", "FRP", "VPWR", "AAT")
THRESHOLD = 2.0


def _validate_numeric(df, columns):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = df.loc[:, list(columns)].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Required sensor columns must contain finite values and at least one row")


def _fit(rows):
    return HistGradientBoostingRegressor(
        max_iter=100, max_depth=3, random_state=0
    ).fit(rows.loc[:, list(FEATURES)], rows["MAP"])


def train_and_save(csv_path, model_path):
    """Fit on confirmed healthy groups; save model and fixed scoring settings."""
    df = pd.read_csv(csv_path)
    _validate_numeric(df, (*FEATURES, "MAP", "BARO"))
    for column in ("vehicle_group", "faulty", "label_uncertain"):
        if column not in df:
            raise ValueError(f"Missing column: {column}")

    healthy = {}
    for group in HEALTHY_GROUPS:
        rows = df.loc[df.vehicle_group == group]
        if rows.empty or not ((rows.faulty == 0) & (rows.label_uncertain == 0)).all():
            raise ValueError(f"Group {group} is missing or not confirmed healthy")
        healthy[group] = rows

    # Each reference error comes from predicting a car absent from that fit.
    held_out_errors = []
    for group in HEALTHY_GROUPS:
        other = next(g for g in HEALTHY_GROUPS if g != group)
        predicted = _fit(healthy[other]).predict(healthy[group].loc[:, list(FEATURES)])
        held_out_errors.append(np.abs(healthy[group].MAP.to_numpy() - predicted))
    reference_errors = np.concatenate(held_out_errors)
    healthy_scale = float(np.median(reference_errors))
    if healthy_scale <= 0:
        raise ValueError("Healthy reference error is zero; normalized scores are undefined")

    artifact = {
        "model": _fit(pd.concat([healthy[g] for g in HEALTHY_GROUPS])),
        "features": FEATURES,
        "healthy_scale": healthy_scale,
        "healthy_error_p99": float(np.percentile(reference_errors, 99)),
        "threshold": THRESHOLD,
        "healthy_groups": HEALTHY_GROUPS,
        "sklearn_version": sklearn.__version__,
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return artifact


def score_vehicle(vehicle_df, model_path):
    """Return per-row MAP evidence and one vehicle-level anomaly summary.

    The flag is based on the median absolute MAP gap across this entire input,
    not on individual rows. Pass one vehicle recording at a time.
    """
    _validate_numeric(vehicle_df, (*FEATURES, "MAP", "BARO"))
    if "vehicle_group" in vehicle_df and vehicle_df.vehicle_group.nunique() != 1:
        raise ValueError("score_vehicle expects rows from one vehicle_group")
    artifact = joblib.load(model_path)
    if tuple(artifact["features"]) != FEATURES:
        raise ValueError("Saved model feature order does not match this code")
    predicted = artifact["model"].predict(vehicle_df.loc[:, list(FEATURES)])
    actual = vehicle_df.MAP.to_numpy(dtype=float)
    gap = np.abs(actual - predicted)
    below_baro = actual < vehicle_df.BARO.to_numpy(dtype=float)
    score = float(np.median(gap) / artifact["healthy_scale"])

    rows = pd.DataFrame({
        "row_number": np.arange(len(vehicle_df)),
        "actual_map": actual,
        "predicted_map": predicted,
        "abs_gap": gap,
        "map_below_baro": below_baro,
    })
    summary = {
        "group_score": score,
        "possible_anomaly": bool(score >= artifact["threshold"]),
        "threshold": artifact["threshold"],
        "median_abs_gap": float(np.median(gap)),
        "healthy_scale": artifact["healthy_scale"],
        "map_below_baro_pct": float(100 * below_baro.mean()),
        "row_count": len(rows),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "score"):
        command = sub.add_parser(action)
        command.add_argument("csv_path")
        command.add_argument("model_path")
        if action == "score":
            command.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.action == "train":
            artifact = train_and_save(args.csv_path, args.model_path)
            print(f"Saved MAP model: {args.model_path}")
            print(f"Healthy scale: {artifact['healthy_scale']:.4f}; threshold: {artifact['threshold']:.1f}")
        else:
            df = pd.read_csv(args.csv_path)
            if "vehicle_group" not in df:
                raise ValueError("Missing column: vehicle_group")
            rows = df.loc[df.vehicle_group == args.group]
            if rows.empty:
                raise ValueError(f"Group {args.group} not found")
            _, summary = score_vehicle(rows, args.model_path)
            print(f"Group {args.group}: score {summary['group_score']:.2f} "
                  f"(threshold {summary['threshold']:.1f}); "
                  f"possible anomaly: {summary['possible_anomaly']}; "
                  f"MAP<BARO: {summary['map_below_baro_pct']:.1f}%")
            if args.group == 3:
                print("Healthy label is unconfirmed.")
            if args.group == 6:
                print("Idle-only recording: score is not meaningful.")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"MAP model error: {exc}\n")


if __name__ == "__main__":
    main()