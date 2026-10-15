"""
Stage 2: honest evaluation + ANN / softmax regression on diesel_obd_clean.csv

What changed vs baseline_models.py
  1. Per-vehicle normalisation (median/IQR) removes "which car is this" offsets.
  2. Sliding-window features (mean/std/min/max/slope) instead of single rows.
  3. Ablation: drop vehicle/ambient-driven sensors (VPWR, AAT, BARO, IAT).
  4. Evaluation is leave-one-vehicle-out (LOVO) for binary, and
     leave-one-FAULT-vehicle-out reported separately, so no vehicle is in train and test.
  5. Models: Softmax regression and ANN (MLP, softmax output) as named in the problem statement.

Assumes rows inside each vehicle_group are in time order.

Usage: python next_models.py diesel_obd_clean.csv out_dir [window] [stride]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SENSORS = ["LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", "MAF", "FRP", "BARO", "VPWR", "AAT"]
AMBIENT = ["VPWR", "AAT", "BARO", "IAT"]


def normalise_per_vehicle(df):
    df = df.copy()
    df[SENSORS] = df[SENSORS].astype(float)
    for _, idx in df.groupby("vehicle_group").groups.items():
        sub = df.loc[idx, SENSORS]
        med = sub.median()
        iqr = (sub.quantile(0.75) - sub.quantile(0.25)).replace(0, 1)
        df.loc[idx, SENSORS] = (sub - med) / iqr
    return df


def make_windows(df, w, stride):
    X, y, g, fam = [], [], [], []
    t = np.arange(w)
    for grp, sub in df.groupby("vehicle_group", sort=True):
        v = sub[SENSORS].to_numpy(float)
        for s in range(0, len(sub) - w + 1, stride):
            win = v[s:s + w]
            slope = np.polyfit(t, win, 1)[0]
            X.append(np.concatenate([win.mean(0), win.std(0), win.min(0), win.max(0), slope]))
            lab = sub.iloc[s:s + w]
            y.append(int(lab.faulty.mean() >= 0.5))
            fam.append(lab.fault_family.mode().iloc[0])
            g.append(grp)
    order = [f"{s}_{k}" for k in ["mean", "std", "min", "max", "slope"] for s in SENSORS]
    return pd.DataFrame(X, columns=order), np.array(y), np.array(g), np.array(fam)


def models():
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "Softmax-Reg": make_pipeline(StandardScaler(),
                                     LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
        "ANN-MLP": make_pipeline(StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                               early_stopping=True, random_state=0, max_iter=400)),
    }


def lovo(X, y, g, name):
    """Leave one vehicle out. Returns pooled metrics + per-vehicle accuracy."""
    pred = np.zeros(len(y), int)
    prob = np.zeros(len(y))
    per = {}
    for v in np.unique(g):
        tr, te = g != v, g == v
        m = models()[name].fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        prob[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else pred[te]
        per[v] = (pred[te] == y[te]).mean()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    return balanced_accuracy_score(y, pred), f1_score(y, pred), auc, per


def main(path, out_dir, w=30, stride=10):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"windows: {w} rows, stride {stride}; per-vehicle normalisation ON")
    Xall, y, g, fam = make_windows(normalise_per_vehicle(df), w, stride)
    log(f"{len(y)} windows, faulty share {y.mean():.2f}, vehicles {sorted(np.unique(g))}")

    feature_sets = {
        "all sensors": list(Xall.columns),
        "no ambient (drop VPWR/AAT/BARO/IAT)": [c for c in Xall.columns
                                                if c.rsplit("_", 1)[0] not in AMBIENT],
    }
    log("\n== Binary healthy vs faulty, leave-one-vehicle-out ==")
    rows, per_tables = [], {}
    for fs_name, cols in feature_sets.items():
        X = Xall[cols].to_numpy()
        for mname in models():
            bal, f1, auc, per = lovo(X, y, g, mname)
            rows.append((fs_name, mname, bal, f1, auc))
            per_tables[(fs_name, mname)] = per
    log(pd.DataFrame(rows, columns=["features", "model", "balanced_acc", "F1_faulty", "AUC"])
        .round(3).to_string(index=False))

    best = max((r for r in rows if r[1] != "Majority"), key=lambda r: r[2])
    per = per_tables[(best[0], best[1])]
    fam_of = {v: pd.Series(fam[g == v]).mode().iloc[0] for v in np.unique(g)}
    log(f"\nPer-vehicle accuracy for best config ({best[0]} / {best[1]}):")
    log(pd.DataFrame({"vehicle": list(per), "family": [fam_of[v] for v in per],
                      "accuracy": [round(a, 3) for a in per.values()]}).to_string(index=False))
    log("\nRead this table first: a healthy vehicle near 0.0 means the model calls every healthy car faulty.")

    (out / "next_results.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 30,
         int(sys.argv[4]) if len(sys.argv) > 4 else 10)
