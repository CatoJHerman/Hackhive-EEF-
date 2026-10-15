"""
Clean MASTER_TRAIN_X.xlsx (DieselOBD) for fault classification.

Steps
 1. Rebuild vehicle groups from contiguous blocks of identical DTC labels
    (the file has no vehicle_id; rows are ordered by vehicle).
 2. Drop exact duplicate sensor rows within each vehicle group.
 3. Collapse DTC columns that are identical on every row into one column.
 4. Build targets: binary (healthy/faulty) and multiclass (DTC family).
 5. Create two leakage-aware splits (see notes in the printed summary).

Usage:  python clean_diesel_obd.py MASTER_TRAIN_X.xlsx out_dir
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", "MAF", "FRP", "BARO", "VPWR", "AAT"]


def main(src: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(src)
    dtc_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    raw_rows = len(df)

    # 1. vehicle groups = contiguous blocks with identical label vector
    label_key = df[dtc_cols].astype(str).agg("".join, axis=1)
    df["vehicle_group"] = (label_key != label_key.shift()).cumsum() - 1

    # 2. remove duplicate sensor rows inside each group (order preserved)
    df = df.drop_duplicates(subset=SENSORS + ["vehicle_group"]).reset_index(drop=True)

    # 3. merge DTC columns that are identical on every row
    merged = {}
    for c in dtc_cols:
        twin = next((k for k, v in merged.items() if df[k].equals(df[c])), None)
        if twin is None:
            merged[c] = [c]
        else:
            merged[twin].append(c)
    dtc_unique = list(merged)

    # 4. targets
    has_dtc = df[[c for c in dtc_unique if c != "P0000"]].sum(axis=1) > 0
    df["faulty"] = has_dtc.astype(int)
    # a block with every label 0 (including P0000) is probably healthy but unconfirmed
    df["label_uncertain"] = ((df[dtc_cols].sum(axis=1) == 0)).astype(int)

    def family(row):
        if row["faulty"] == 0:
            return "healthy"
        return "+".join(sorted(c for c in dtc_unique if c != "P0000" and row[c] == 1))

    fam_by_group = df.groupby("vehicle_group").apply(lambda g: family(g.iloc[0]))
    df["fault_family"] = df["vehicle_group"].map(fam_by_group)
    df["class_id"] = df["fault_family"].astype("category").cat.codes

    # 5a. binary split: hold out whole vehicles (1 healthy + 2 faulty) -> tests generalisation
    sizes = df.groupby("vehicle_group").size()
    healthy_groups = [g for g in sizes.index if fam_by_group[g] == "healthy"]
    faulty_groups = [g for g in sizes.index if fam_by_group[g] != "healthy"]
    test_groups = {healthy_groups[0], *faulty_groups[:2]}
    df["split_binary_vehicle"] = np.where(df["vehicle_group"].isin(test_groups), "test", "train")

    # 5b. multiclass split: chronological 70/15/15 inside each vehicle group
    df["split_multiclass_time"] = "train"
    for g, idx in df.groupby("vehicle_group").groups.items():
        n = len(idx)
        idx = list(idx)
        df.loc[idx[int(n * 0.70): int(n * 0.85)], "split_multiclass_time"] = "val"
        df.loc[idx[int(n * 0.85):], "split_multiclass_time"] = "test"

    drop_cols = [c for twins in merged.values() for c in twins[1:]]
    df = df.drop(columns=drop_cols)
    df.to_csv(out / "diesel_obd_clean.csv", index=False)

    print(f"rows: {raw_rows} -> {len(df)} after dedup ({raw_rows - len(df)} removed)")
    print("merged identical DTC columns:", {k: v for k, v in merged.items() if len(v) > 1})
    print("\nvehicle groups:")
    print(df.groupby("vehicle_group").agg(rows=("faulty", "size"), family=("fault_family", "first"),
                                          uncertain=("label_uncertain", "max"),
                                          binary_split=("split_binary_vehicle", "first")).to_string())
    print("\nbinary test vehicles (whole-vehicle holdout):", sorted(test_groups))
    print("\nNOTE: each fault family comes from a single vehicle, so a vehicle-wise split cannot")
    print("test multiclass generalisation. split_multiclass_time is the best available option,")
    print("and its scores should be described as same-vehicle performance.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
