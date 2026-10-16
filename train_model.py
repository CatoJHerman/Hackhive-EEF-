"""Train a LiU binary baseline with recording-level development validation.

Run: ./.venv-liu/bin/python train_model.py
The official test set is never read. Saved model is experimental until tested.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from define_liu_evaluation import define_plan


# Preserve CSV scale. Ambient variables can encode laboratory recording identity.
FEATURES = ["Intercooler_pressure", "intercooler_temperature",
            "intake_manifold_pressure", "air_mass_flow", "engine_speed",
            "throttle_position", "wastegate_position", "injected_fuel_mass"]
SETTINGS = dict(max_iter=100, max_leaf_nodes=15, max_depth=6,
                learning_rate=0.1, min_samples_leaf=50, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False, random_state=42)
THRESHOLD = 0.5


def metrics(y, predicted):
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "fault_recall": tp / (tp + fn) if tp + fn else None,
            "missed_fault_rate": fn / (tp + fn) if tp + fn else None}


def train(prepared_dir, output_dir):
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory to preserve earlier results.")
    plan = json.loads((prepared_dir / "evaluation_plan.json").read_text())
    if define_plan(prepared_dir) != plan:
        raise ValueError("Data or evaluation plan changed; regenerate and review the plan.")
    data = pd.read_csv(prepared_dir / "liu_training_labelled.csv")
    data = data.loc[data.training_eligible.eq(1)].copy().reset_index(drop=True)
    X = data[FEATURES].astype(float)
    y = data.fault_present.astype(int)
    if not np.isfinite(X.to_numpy()).all() or set(y.unique()) != {0, 1}:
        raise ValueError("Expected finite inputs and both binary target classes.")
    scores = np.full(len(data), np.nan)
    coverage = np.zeros(len(data), dtype=int)
    folds = []
    for fold in plan["folds"]:
        training = data.recording_id.isin(fold["train_recordings"])
        validation = data.recording_id.isin(fold["validation_recordings"])
        if (training & validation).any() or not (training | validation).all():
            raise ValueError("Invalid fold membership.")
        model = HistGradientBoostingClassifier(**SETTINGS)
        model.fit(X.loc[training], y.loc[training])
        fold_scores = model.predict_proba(X.loc[validation])[:, list(model.classes_).index(1)]
        scores[validation] = fold_scores
        coverage[validation] += 1
        result = {"recording_id": fold["validation_recordings"][0],
                  **metrics(y.loc[validation], fold_scores >= THRESHOLD),
                  "notes": fold["interpretation_notes"]}
        folds.append(result)
        print(json.dumps(result), flush=True)
    if not np.all(coverage == 1) or not np.isfinite(scores).all():
        raise ValueError("Every eligible row must have exactly one held-out prediction.")
    predictions = (scores >= THRESHOLD).astype(int)
    report = {
        "status": "Experimental baseline; development validation only. Official test not evaluated.",
        "model": "HistGradientBoostingClassifier", "settings": SETTINGS,
        "features": FEATURES, "threshold": THRESHOLD,
        "threshold_policy": "Fixed 0.5 baseline; not tuned using validation or test scores.",
        "score_interpretation": "Uncalibrated model score, not a verified probability of fault.",
        "preprocessing": "Original CSV scale; no imputation, scaling, temporal features or smoothing.",
        "class_weight": "Balanced weights calculated using training rows separately in each fold.",
        "validation": "Leave one recording out; early stopping disabled to avoid internal random validation.",
        "pooled": metrics(y, predictions),
        "mean_recording_false_positive_rate": float(np.mean([f["false_positive_rate"] for f in folds if f["false_positive_rate"] is not None])),
        "mean_fault_recording_recall": float(np.mean([f["fault_recall"] for f in folds if f["fault_recall"] is not None])),
        "reference_always_normal": metrics(y, np.zeros(len(y), dtype=int)),
        "reference_always_fault": metrics(y, np.ones(len(y), dtype=int)),
        "folds": folds, "limitations": plan["limitations"],
        "prepared_csv_sha256": plan["prepared_csv_sha256"],
        "sklearn_version": sklearn.__version__,
    }
    # After development validation, fit the reusable baseline on all eligible training data.
    final_model = HistGradientBoostingClassifier(**SETTINGS).fit(X, y)
    bundle = {"model": final_model, "features": FEATURES, "threshold": THRESHOLD,
              "classes": {0: "normal", 1: "fault_detected"},
              "input_scale": "Original LiU CSV units; no physical-unit conversions.",
              "status": report["status"], "sklearn_version": sklearn.__version__,
              "prepared_csv_sha256": plan["prepared_csv_sha256"]}
    output_dir.mkdir(parents=True)
    joblib.dump(bundle, output_dir / "baseline_model.joblib")
    restored = joblib.load(output_dir / "baseline_model.joblib")
    sample = X.iloc[::max(1, len(X)//100)]
    np.testing.assert_allclose(final_model.predict_proba(sample), restored["model"].predict_proba(sample))
    held_out = data[["recording_id", "source_row", "time", "fault_present"]].copy()
    held_out["fault_score"] = scores
    held_out["prediction"] = predictions
    held_out.to_csv(output_dir / "held_out_predictions.csv", index=False)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Pooled metrics:", json.dumps(report["pooled"]), flush=True)
    print(f"Saved model, development predictions and report to {output_dir.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/liu_preparation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/liu_baseline"))
    args = parser.parse_args()
    train(args.prepared_dir, args.output_dir)
