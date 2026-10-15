"""
Stage 2 v2: honest evaluation + ANN / softmax regression on diesel_obd_clean.csv
Adds calibration check (none, first-N-rows, full), seeds (mean ± std),
window sweep, and vehicle 3 sensitivity analysis.

Usage: python next_models_v2.py diesel_obd_clean.csv out_dir [n_cal]
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


def lovo_single(X, y, g, name, seed=0):
    pred = np.zeros(len(y), int)
    prob = np.zeros(len(y))
    per = {}
    for v in np.unique(g):
        tr, te = g != v, g == v
        m = get_model(name, seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        prob[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else pred[te]
        per[v] = (pred[te] == y[te]).mean()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    bal = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    return bal, f1, auc, per


def lovo_eval(X, y, g, name):
    if name in ["Majority", "Softmax-Reg"]:
        bal, f1, auc, per = lovo_single(X, y, g, name, seed=0)
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
            b, f, a, p = lovo_single(X, y, g, name, seed=s)
            bals.append(b)
            f1s.append(f)
            aucs.append(a)
            pers.append(p)
        avg_per = {v: float(np.mean([p[v] for p in pers])) for v in np.unique(g)}
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


def main(path, out_dir, n_cal=300):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(path)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("EXPLORATORY VEHICLE-WISE OBD CLASSIFICATION REPORT (v2)")
    log("=" * 70)
    log(f"Calibration length: {n_cal} rows. Evaluation stride: 10 rows.")
    log(f"Total rows: {len(df_raw)} across vehicles {sorted(df_raw.vehicle_group.unique())}")
    for g, sub in df_raw.groupby("vehicle_group"):
        log(f"  Vehicle {g} ({sub.fault_family.iloc[0]}): {len(sub)} rows")

    # 1. Calibration Check (w=30)
    log("\n" + "=" * 70)
    log("1. CALIBRATION CHECK (Window=30 rows)")
    log("=" * 70)
    calib_modes = ["none", f"first{n_cal}", "full (optimistic)"]
    models_list = ["Majority", "Softmax-Reg", "ANN-MLP"]
    
    calib_results = []
    per_vehicle_store = {}
    
    for cm in calib_modes:
        actual_mode = "first300" if cm.startswith("first") else ("full" if "full" in cm else "none")
        df_norm = normalise_per_vehicle(df_raw, mode=actual_mode, n_cal=n_cal)
        Xall, y, g, fam = make_windows(df_norm, w=30, stride=10)
        
        feature_sets = {
            "all sensors": list(Xall.columns),
            "no ambient": [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT],
        }
        
        for fs_name, cols in feature_sets.items():
            X = Xall[cols].to_numpy()
            for mname in models_list:
                res = lovo_eval(X, y, g, mname)
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
                per_vehicle_store[(cm, fs_name, mname)] = res["per_vehicle"]

    # Display Calibration table
    cal_df = pd.DataFrame(calib_results)
    disp_cal = cal_df.copy()
    disp_cal["balanced_acc"] = [format_stat(r["bal_mean"], r["bal_std"]) for _, r in cal_df.iterrows()]
    disp_cal["F1_faulty"] = [format_stat(r["f1_mean"], r["f1_std"]) for _, r in cal_df.iterrows()]
    disp_cal["AUC"] = [format_stat(r["auc_mean"], r["auc_std"]) for _, r in cal_df.iterrows()]
    log(disp_cal[["calib", "features", "model", "balanced_acc", "F1_faulty", "AUC"]].to_string(index=False))

    # Identify best deployable calibration (exclude 'full (optimistic)' and 'Majority')
    deployable = [r for r in calib_results if "optimistic" not in r["calib"] and r["model"] != "Majority"]
    best_dep = max(deployable, key=lambda r: r["bal_mean"])
    log(f"\nBest deployable config: {best_dep['calib']} / {best_dep['features']} / {best_dep['model']}")
    log(f"  Balanced Accuracy: {format_stat(best_dep['bal_mean'], best_dep['bal_std'])}, "
        f"F1: {format_stat(best_dep['f1_mean'], best_dep['f1_std'])}, "
        f"AUC: {format_stat(best_dep['auc_mean'], best_dep['auc_std'])}")

    # Per-vehicle comparison table
    fam_of = {v: df_raw[df_raw.vehicle_group == v].fault_family.iloc[0] for v in sorted(df_raw.vehicle_group.unique())}
    per_dep = per_vehicle_store[(best_dep["calib"], best_dep["features"], best_dep["model"])]
    
    # Also find corresponding full (optimistic) config for comparison
    full_match = next((r for r in calib_results if "optimistic" in r["calib"] 
                       and r["features"] == best_dep["features"] and r["model"] == best_dep["model"]), None)
    per_full = per_vehicle_store[(full_match["calib"], full_match["features"], full_match["model"])] if full_match else {}

    log(f"\nPer-vehicle accuracy comparison [Deployable: {best_dep['calib']} vs Full (Optimistic)]:")
    comp_rows = []
    for v in sorted(fam_of.keys()):
        comp_rows.append({
            "vehicle": v,
            "family": fam_of[v],
            f"acc_{best_dep['calib']}": round(per_dep.get(v, np.nan), 3),
            "acc_full_optimistic": round(per_full.get(v, np.nan), 3) if per_full else np.nan,
        })
    log(pd.DataFrame(comp_rows).to_string(index=False))

    # 2. Window Sweep on Best Deployable Calibration
    best_calib_actual = "first300" if best_dep["calib"].startswith("first") else "none"
    log("\n" + "=" * 70)
    log(f"2. WINDOW SWEEP ({best_dep['calib']}, {best_dep['features']})")
    log("=" * 70)
    
    window_results = []
    for w in [30, 60, 120]:
        df_norm = normalise_per_vehicle(df_raw, mode=best_calib_actual, n_cal=n_cal)
        Xall, y, g, fam = make_windows(df_norm, w=w, stride=10)
        cols = list(Xall.columns) if best_dep["features"] == "all sensors" else [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT]
        X = Xall[cols].to_numpy()
        
        for mname in ["Softmax-Reg", "ANN-MLP"]:
            res = lovo_eval(X, y, g, mname)
            window_results.append({
                "window": w,
                "model": mname,
                "balanced_acc": format_stat(res["bal_mean"], res["bal_std"]),
                "F1_faulty": format_stat(res["f1_mean"], res["f1_std"]),
                "AUC": format_stat(res["auc_mean"], res["auc_std"]),
                "windows_count": len(y),
            })
    log(pd.DataFrame(window_results).to_string(index=False))

    # 3. Sensitivity Run: Removing Vehicle 3 (unconfirmed healthy)
    log("\n" + "=" * 70)
    log(f"3. SENSITIVITY RUN: REMOVING VEHICLE 3 (Unconfirmed Healthy)")
    log("=" * 70)
    df_no3 = df_raw[df_raw.vehicle_group != 3].reset_index(drop=True)
    df_no3_norm = normalise_per_vehicle(df_no3, mode=best_calib_actual, n_cal=n_cal)
    Xall, y, g, fam = make_windows(df_no3_norm, w=30, stride=10)
    cols = list(Xall.columns) if best_dep["features"] == "all sensors" else [c for c in Xall.columns if c.rsplit("_", 1)[0] not in AMBIENT]
    X = Xall[cols].to_numpy()
    
    sens_results = []
    for mname in ["Softmax-Reg", "ANN-MLP"]:
        res = lovo_eval(X, y, g, mname)
        sens_results.append({
            "dataset": "Without Vehicle 3",
            "model": mname,
            "balanced_acc": format_stat(res["bal_mean"], res["bal_std"]),
            "F1_faulty": format_stat(res["f1_mean"], res["f1_std"]),
            "AUC": format_stat(res["auc_mean"], res["auc_std"]),
            "windows_count": len(y),
        })
    log(pd.DataFrame(sens_results).to_string(index=False))
    
    log("\n" + "=" * 70)
    log("NOTES ON METHODOLOGY & EXPLORATORY FRAMING:")
    log(" - 'full' calibration normalises using full vehicle data (including future and faulty rows); labelled optimistic.")
    log(f" - 'first{n_cal}' calibration uses only the first {n_cal} rows per car. Windows overlapping this calibration")
    log("   window are retained in evaluation, which is a minor optimism in deployment.")
    log(" - LOVO guarantees no vehicle is shared between training and testing sets.")
    log("=" * 70)

    (out / "next_results_v2.txt").write_text("\n".join(lines))
    log(f"\nResults saved to {out / 'next_results_v2.txt'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 300)
