import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


HEALTHY_GROUPS = (1, 7)
SENSORS = ("LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", "MAF", "FRP", "BARO", "VPWR", "AAT")
FEATURES = tuple(c for c in SENSORS if c not in ("MAP", "BARO"))
THRESHOLD = 2.0


def fit_model(rows):
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=0)
    return model.fit(rows.loc[:, FEATURES], rows["MAP"])


def main(path):
    df = pd.read_csv(path)
    required = set(SENSORS) | {"vehicle_group", "fault_family", "faulty", "label_uncertain"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if df[list(SENSORS)].isna().any().any() or not np.isfinite(df[list(SENSORS)].to_numpy(float)).all():
        raise ValueError("Sensor values must be finite and nonmissing")

    healthy = {}
    for group in HEALTHY_GROUPS:
        rows = df[df.vehicle_group == group]
        if rows.empty or not ((rows.faulty == 0) & (rows.label_uncertain == 0)).all():
            raise ValueError(f"Group {group} is missing or not confirmed healthy")
        healthy[group] = rows

    # Predict each confirmed healthy car with a model trained only on the other.
    # This is the reference error for a car unseen during training.
    held_out_errors = []
    for group in HEALTHY_GROUPS:
        other = next(g for g in HEALTHY_GROUPS if g != group)
        model = fit_model(healthy[other])
        errors = np.abs(healthy[group].MAP.to_numpy() - model.predict(healthy[group].loc[:, FEATURES]))
        held_out_errors.extend(errors)
    healthy_scale = float(np.median(held_out_errors))
    if healthy_scale <= 0:
        raise ValueError("Healthy error scale is zero; scores cannot be calculated")

    model = fit_model(df[df.vehicle_group.isin(HEALTHY_GROUPS)])
    print("Exploratory MAP consistency check")
    print("Training: confirmed healthy groups 1 and 7 only")
    print("Healthy reference: leave-one-healthy-group-out median absolute MAP error")
    print(f"Healthy error scale: {healthy_scale:.4f} (MAP units unconfirmed)")
    print(f"Possible anomaly if normalized score >= {THRESHOLD:.1f}")
    print("\nGroup  Label                 Rows   MAP gap  Score  MAP<BARO  Interpretation")
    for group, rows in df.groupby("vehicle_group", sort=True):
        gap = np.abs(rows.MAP.to_numpy() - model.predict(rows.loc[:, FEATURES]))
        median_gap = float(np.median(gap))
        score = median_gap / healthy_scale
        below_baro = float((rows.MAP < rows.BARO).mean())
        if group == 6:
            note = "idle-only; score not meaningful"
        elif group == 3:
            note = "healthy label unconfirmed"
        else:
            note = "possible anomaly" if score >= THRESHOLD else "not flagged"
        label = str(rows.fault_family.iloc[0])
        print(f"{group:>5}  {label:<20} {len(rows):>5}  {median_gap:>8.3f}  {score:>5.2f}  {below_baro:>8.1%}  {note}")
    print("\nMAP<BARO is a separate physical comparison, not an input to the model.")
    print("Only two confirmed healthy groups; this result is exploratory, not fault validation.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python obd_consistency_test.py diesel_obd_clean.csv")
    main(sys.argv[1])
