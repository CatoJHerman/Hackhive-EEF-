"""Compare two predefined feature sets on the original recording-level folds.

Run: OMP_NUM_THREADS=2 ./.venv-liu/bin/python compare_liu_models.py
No official test data is read. Selection scores are development results only.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from define_liu_evaluation import define_plan
from liu_features import build_features, REQUIRED
from train_model import SETTINGS, metrics


def summarize(data, scores):
    folds = []
    for name, indices in data.groupby("recording_id", sort=True).groups.items():
        idx = np.asarray(list(indices))
        folds.append({"recording_id": name, **metrics(data.loc[idx, "fault_present"], scores[idx] >= 0.5)})
    fpr = float(np.mean([f["false_positive_rate"] for f in folds if f["false_positive_rate"] is not None]))
    recall = float(np.mean([f["fault_recall"] for f in folds if f["fault_recall"] is not None]))
    return {"pooled": metrics(data.fault_present, scores >= 0.5), "folds": folds,
            "mean_recording_false_positive_rate": fpr, "mean_fault_recording_recall": recall,
            "selection_score": (recall + 1 - fpr) / 2}


def main():
    prepared = Path("outputs/liu_preparation")
    output = Path("outputs/liu_feature_comparison")
    if output.exists():
        raise FileExistsError("Comparison output already exists; preserve previous results.")
    plan = json.loads((prepared / "evaluation_plan.json").read_text())
    if define_plan(prepared) != plan:
        raise ValueError("Data or plan changed.")
    original = pd.read_csv(prepared / "liu_training_labelled.csv")
    matrices, readiness = {}, None
    for name, temporal in (("relationships", False), ("relationships_history_1s", True)):
        pieces, masks = [], []
        for _, group in original.groupby("recording_id", sort=False):
            f, ready = build_features(group, temporal=temporal)
            pieces.append(f)
            masks.append(ready)
        matrices[name] = pd.concat(pieces).sort_index()
        if temporal:
            readiness = pd.concat(masks).sort_index()
    usable = original.training_eligible.eq(1) & readiness
    data = original.loc[usable].reset_index(drop=True)
    y = data.fault_present.astype(int)
    matrices = {name: f.loc[usable].reset_index(drop=True) for name, f in matrices.items()}
    if not all(np.isfinite(f.to_numpy()).all() for f in matrices.values()):
        raise ValueError("Non-finite model inputs.")
    old = pd.read_csv("outputs/liu_baseline/held_out_predictions.csv")
    aligned = data[["recording_id", "source_row"]].merge(old, on=["recording_id", "source_row"], how="left", validate="one_to_one", sort=False)
    if aligned.fault_score.isna().any():
        raise ValueError("Missing baseline predictions.")
    results = {"baseline_common_rows": summarize(data, aligned.fault_score.to_numpy())}
    all_scores = {}
    for name, X in matrices.items():
        scores = np.full(len(data), np.nan)
        coverage = np.zeros(len(data), dtype=int)
        for fold in plan["folds"]:
            tr = data.recording_id.isin(fold["train_recordings"])
            va = data.recording_id.isin(fold["validation_recordings"])
            model = HistGradientBoostingClassifier(**SETTINGS).fit(X.loc[tr], y.loc[tr])
            scores[va] = model.predict_proba(X.loc[va])[:, list(model.classes_).index(1)]
            coverage[va] += 1
            print(name, fold["validation_recordings"][0], json.dumps(metrics(y.loc[va], scores[va] >= 0.5)), flush=True)
        if not (coverage == 1).all() or not np.isfinite(scores).all():
            raise ValueError("Invalid held-out prediction coverage.")
        all_scores[name] = scores
        results[name] = summarize(data, scores)
    # One fixed metric and two predefined candidates; no threshold search.
    best = max(results, key=lambda name: results[name]["selection_score"])
    report = {"status": "Development comparison only; repeated use of validation folds is model selection, not independent testing.",
              "rows_compared": len(data), "eligible_rows_without_history": int(original.training_eligible.eq(1).sum() - len(data)),
              "threshold": 0.5, "model_settings": SETTINGS,
              "selection_rule": "Maximize 0.5 * (equal-fault-recording recall + 1 - equal-recording false-positive rate).",
              "best_development_candidate": best, "results": results,
              "changes": "Remove raw intercooler temperature; add numerical ratios. Temporal candidate adds causal 1-second means, standard deviations and changes. No raw ambient inputs.",
              "feature_columns": {name: list(X.columns) for name, X in matrices.items()},
              "limitations": plan["limitations"], "prepared_csv_sha256": plan["prepared_csv_sha256"]}
    output.mkdir(parents=True)
    held = data[["recording_id", "source_row", "time", "fault_present"]].copy()
    held["baseline_score"] = aligned.fault_score.to_numpy()
    for name, scores in all_scores.items():
        held[name + "_score"] = scores
    held.to_csv(output / "held_out_predictions.csv", index=False)
    if best in matrices:
        X = matrices[best]
        fitted = HistGradientBoostingClassifier(**SETTINGS).fit(X, y)
        bundle = {"model": fitted, "feature_variant": best, "features": list(X.columns),
                  "required_raw_columns": REQUIRED, "threshold": 0.5,
                  "feature_module": "liu_features.build_features", "common_warmup_seconds": 1,
                  "status": "Experimental development-selected model; not independently tested.",
                  "input_scale": "Original LiU CSV values", "prepared_csv_sha256": plan["prepared_csv_sha256"]}
        joblib.dump(bundle, output / "candidate_model.joblib")
        restored = joblib.load(output / "candidate_model.joblib")
        sample = X.iloc[::1000]
        np.testing.assert_allclose(fitted.predict_proba(sample), restored["model"].predict_proba(sample))
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in result.items() if k != "folds"} for name, result in results.items()}, indent=2), flush=True)
    print("Best development candidate:", best, flush=True)


if __name__ == "__main__":
    main()
