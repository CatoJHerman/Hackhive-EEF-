"""
Stage 2 v3: Freeze Run with Strict Non-Overlapping Evaluation
(Causal first-300 calibration, test windows evaluated only at start >= 300).

Usage: python next_models_v3.py diesel_obd_clean.csv out_dir [n_cal] [stride]
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
SEEDS = [0, 1, 2, 3, 4]


def normalise_per_vehicle(df, mode="full", n_cal=300):
    df = df.copy()
    df[SENSORS] = df[SENSORS].astype(float)
    if mode == "none":
        return df
    for _, idx in df.groupby("vehicle_group").groups.items():
        sub = df.loc[idx, SENSORS]
        cal_sub = sub.iloc[:n_cal] if mode == "first300" else sub
        med = cal_sub.median()
        iqr = (cal_sub.quantile(0.75) - cal_sub.quantile(0.25)).replace(0, 1)
        df.loc[idx, SENSORS] = (sub - med) / iqr
    return df


def make_windows(df, w, stride=10):
    X, y, g, fam, starts = [], [], [], [], []
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
            starts.append(s)
    order = [f"{s}_{k}" for k in ["mean", "std", "min", "max", "slope"] for s in SENSORS]
    return (pd.DataFrame(X, columns=order), np.array(y), np.array(g),
            np.array(fam), np.array(starts))


def get_model(name, seed=0):
    if name == "Majority":
        return DummyClassifier(strategy="most_frequent")
    elif name == "Softmax-Reg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed))
    elif name == "ANN-MLP":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                           early_stopping=True, random_state=seed, max_iter=400))
    raise ValueError(f"Unknown model {name}")


def lovo_single(X, y, g, starts, name, n_cal=300, seed=0):
    """
    Leave-one-vehicle-out:
    Training uses ALL windows of training vehicles (starts >= 0).
    Testing on vehicle v evaluates ONLY windows starting at s >= n_cal (strictly causal).
    """
    vehicles = np.unique(g)
    test_mask = starts >= n_cal
    pred = np.full(len(y), -1, dtype=int)
    prob = np.full(len(y), np.nan, dtype=float)
    per = {}

    for v in vehicles:
        tr = g != v
        te = (g == v) & test_mask
        m = get_model(name, seed).fit(X[tr], y[tr])
        if np.sum(te) > 0:
            pred[te] = m.predict(X[te])
            if hasattr(m, "predict_proba"):
                prob[te] = m.predict_proba(X[te])[:, 1]
            else:
                prob[te] = pred[te]
            per[v] = float((pred[te] == y[te]).mean())
        else:
            per[v] = np.nan

    eval_idx = test_mask
    y_eval = y[eval_idx]
    pred_eval = pred[eval_idx]
    prob_eval = prob[eval_idx]

    auc = roc_auc_score(y_eval, prob_eval) if len(np.unique(y_eval)) > 1 else np.nan
    bal = balanced_accuracy_score(y_eval, pred_eval)
    f1 = f1_score(y_eval, pred_eval)
    return bal, f1, auc, per


def lovo_eval(X, y, g, starts, name, n_cal=300):
    if name in ["Majority", "Softmax-Reg"]:
        bal, f1, auc, per = lovo_single(X, y, g, starts, name, n_cal=n_cal, seed=0)
        return {
            "bal_mean": bal, "bal_std": 0.0,
            "f1_mean": f1, "f1_std": 0.0,
            "auc_mean": auc, "auc_std": 0.0,
            "per_vehicle": per,
        }
    else:
        # ANN across 5 seeds
        bals, f1s, aucs, pers = [], [], [], []
        for s in SEEDS:
            b, f, a, p = lovo_single(X, y, g, starts, name, n_cal=n_cal, seed=s)
            bals.append(b)
            f1s.append(f)
            aucs.append(a)
            pers.append(p)
        avg_per = {v: float(np.mean([p[v] for p in pers if not np.isnan(p[v])])) for v in np.unique(g)}
        return {
            "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "per_vehicle": avg_per,
        }


def format_stat(mean, std):
    if std > 0.0005:
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean:.3f}"


def main(path, out_dir, n_cal=300, stride=10):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(path)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 75)
    log("FROZEN RIGOROUS VEHICLE-WISE OBD CLASSIFICATION REPORT (v3)")
    log("=" * 75)
    log(f"Calibration threshold: {n_cal} rows. Stride: {stride} rows.")
    log(f"Evaluation policy: Held-out vehicles scored ONLY on windows with start >= {n_cal}.")
    log(f"Training vehicles retain all windows (from row 0).")
    log(f"Total dataset: {len(df_raw)} rows across 8 vehicles:")
    for g, sub in df_raw.groupby("vehicle_group"):
        log(f"  Vehicle {g} ({sub.fault_family.iloc[0]}): {len(sub)} rows "
            f"(post-cal eval rows: {len(sub) - n_cal})")

    # 1. Calibration Check (Window=30 rows)
    log("\n" + "=" * 75)
    log("1. CALIBRATION CHECK (Window=30 rows, scored only on start >= 300)")
    log("=" * 75)
    calib_modes = ["none", f"first{n_cal}", "full (optimistic)"]
    models_list = ["Majority", "Softmax-Reg", "ANN-MLP"]

    calib_results = []
    per_store_w30 = {}

    for cm in calib_modes:
        actual_mode = "first300" if cm.startswith("first") else ("full" if "full" in cm else "none")
        df_norm = normalise_per_vehicle(df_raw, mode=actual_mode, n_cal=n_cal)
        Xall, y, g, fam, starts = make_windows(df_norm, w=30, stride=stride)

        feature_sets = {
            "all sensors": list(Xall.columns),
            "no ambient": [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT],
        }

        for fs_name, cols in feature_sets.items():
            X = Xall[cols].to_numpy()
            for mname in models_list:
                res = lovo_eval(X, y, g, starts, mname, n_cal=n_cal)
                calib_results.append({
                    "calib": cm,
                    "features": fs_name,
                    "model": mname,
                    "bal_mean": res["bal_mean"],
                    "bal_std": res["bal_std"],
                    "f1_mean": res["f1_mean"],
                    "f1_std": res["f1_std"],
                    "auc_mean": res["auc_mean"],
                    "auc_std": res["auc_std"],
                })
                per_store_w30[(cm, fs_name, mname)] = res["per_vehicle"]

    cal_df = pd.DataFrame(calib_results)
    disp_cal = cal_df.copy()
    disp_cal["balanced_acc"] = [format_stat(r["bal_mean"], r["bal_std"]) for _, r in cal_df.iterrows()]
    disp_cal["F1_faulty"] = [format_stat(r["f1_mean"], r["f1_std"]) for _, r in cal_df.iterrows()]
    disp_cal["AUC"] = [format_stat(r["auc_mean"], r["auc_std"]) for _, r in cal_df.iterrows()]
    log(disp_cal[["calib", "features", "model", "balanced_acc", "F1_faulty", "AUC"]].to_string(index=False))

    # 2. Frozen Configuration: Window Sweep across 30, 60, 120
    log("\n" + "=" * 75)
    log("2. FROZEN CONFIGURATION: WINDOW SWEEP (first300 calibration, no ambient sensors)")
    log("   Pre-declared: Softmax regression is the primary deployable model;")
    log("   ANN-MLP (5 seeds) is reported side-by-side. Evaluated on start >= 300.")
    log("=" * 75)

    windows = [30, 60, 120]
    window_summary = []
    per_store_frozen = {}

    df_norm_frozen = normalise_per_vehicle(df_raw, mode="first300", n_cal=n_cal)
    for w in windows:
        Xall, y, g, fam, starts = make_windows(df_norm_frozen, w=w, stride=stride)
        cols = [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT]
        X = Xall[cols].to_numpy()
        n_eval_windows = int(np.sum(starts >= n_cal))

        for mname in ["Majority", "Softmax-Reg", "ANN-MLP"]:
            res = lovo_eval(X, y, g, starts, mname, n_cal=n_cal)
            window_summary.append({
                "window": w,
                "model": mname,
                "balanced_acc": format_stat(res["bal_mean"], res["bal_std"]),
                "F1_faulty": format_stat(res["f1_mean"], res["f1_std"]),
                "AUC": format_stat(res["auc_mean"], res["auc_std"]),
                "eval_windows": n_eval_windows,
                "total_windows": len(y),
            })
            if mname == "Softmax-Reg":
                per_store_frozen[w] = res["per_vehicle"]

    log(pd.DataFrame(window_summary).to_string(index=False))

    # 3. Per-Vehicle Accuracy Table across Window Lengths
    log("\n" + "=" * 75)
    log("3. PER-VEHICLE ACCURACY ACROSS WINDOW LENGTHS")
    log("   (Frozen config: Softmax-Reg / first300 / no ambient; windows with start >= 300)")
    log("   Also shows full (optimistic) w=30 for comparison.")
    log("=" * 75)
    fam_of = {v: df_raw[df_raw.vehicle_group == v].fault_family.iloc[0] for v in sorted(df_raw.vehicle_group.unique())}
    per_full_w30 = per_store_w30.get(("full (optimistic)", "no ambient", "Softmax-Reg"), {})

    pv_rows = []
    for v in sorted(fam_of.keys()):
        pv_rows.append({
            "vehicle": v,
            "family": fam_of[v],
            "acc_w30": round(per_store_frozen.get(30, {}).get(v, np.nan), 3),
            "acc_w60": round(per_store_frozen.get(60, {}).get(v, np.nan), 3),
            "acc_w120": round(per_store_frozen.get(120, {}).get(v, np.nan), 3),
            "full_opt_w30": round(per_full_w30.get(v, np.nan), 3),
        })
    log(pd.DataFrame(pv_rows).to_string(index=False))

    # 4. Sensitivity Run: Removing Vehicle 3
    log("\n" + "=" * 75)
    log("4. SENSITIVITY RUN: REMOVING VEHICLE 3 (Unconfirmed Healthy)")
    log("   Frozen config (first300 / no ambient) tested across 30, 60, 120 without Vehicle 3")
    log("=" * 75)
    df_no3 = df_raw[df_raw.vehicle_group != 3].reset_index(drop=True)
    df_no3_norm = normalise_per_vehicle(df_no3, mode="first300", n_cal=n_cal)

    sens_summary = []
    for w in windows:
        Xall, y, g, fam, starts = make_windows(df_no3_norm, w=w, stride=stride)
        cols = [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT]
        X = Xall[cols].to_numpy()
        n_eval_windows = int(np.sum(starts >= n_cal))

        for mname in ["Softmax-Reg", "ANN-MLP"]:
            res = lovo_eval(X, y, g, starts, mname, n_cal=n_cal)
            sens_summary.append({
                "window": w,
                "model": mname,
                "balanced_acc": format_stat(res["bal_mean"], res["bal_std"]),
                "F1_faulty": format_stat(res["f1_mean"], res["f1_std"]),
                "AUC": format_stat(res["auc_mean"], res["auc_std"]),
                "eval_windows": n_eval_windows,
            })
    log(pd.DataFrame(sens_summary).to_string(index=False))

    log("\n" + "=" * 75)
    log("METHODOLOGY SUMMARY:")
    log(" - Zero-leakage causal evaluation: calibration window (rows 0-300) is excluded from")
    log("   test scoring for all held-out vehicles, eliminating calibration overlap bias.")
    log(" - Training folds retain all rows (0 to end) to maximize training density.")
    log(" - Frozen models: Softmax-Reg (linear baseline) + MLP (64, 32) across 5 seeds.")
    log("=" * 75)

    (out / "final_results_v3.txt").write_text("\n".join(lines))
    log(f"\nFinal report saved to {out / 'final_results_v3.txt'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 300,
         int(sys.argv[4]) if len(sys.argv) > 4 else 10)
