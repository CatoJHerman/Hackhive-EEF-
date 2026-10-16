"""Recording-level evaluation of a normal-only fitted residual detector."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from define_liu_evaluation import define_plan
from liu_residual_model import fit_detector, score_detector
from liu_metrics import metrics


def main():
    root = Path("outputs/liu_preparation")
    out = Path("outputs/liu_residuals")
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((root / "evaluation_plan.json").read_text())
    if define_plan(root) != plan:
        raise ValueError("Plan changed")
    data = pd.read_csv(root / "liu_training_labelled.csv")
    predictions, folds = [], []
    for fold in plan["folds"]:
        training = data[data.recording_id.isin(fold["train_recordings"])]
        validation = data[data.recording_id.isin(fold["validation_recordings"])]
        bundle = fit_detector(training)
        scored = score_detector(bundle, validation)
        use = validation.training_eligible.eq(1) & scored.prediction_available
        result = {"recording_id": fold["validation_recordings"][0],
                  **metrics(validation.loc[use, "fault_present"], scored.loc[use, "prediction"]),
                  "fit_rows": bundle["fit_rows"], "calibration_rows": bundle["calibration_rows"],
                  "threshold": bundle["threshold"], "notes": fold["interpretation_notes"]}
        folds.append(result)
        p = validation[["recording_id", "source_row", "time", "fault_present", "training_eligible"]].join(scored)
        predictions.append(p)
        print(json.dumps(result), flush=True)
    held = pd.concat(predictions).sort_index()
    use = held.training_eligible.eq(1) & held.prediction_available
    report = {"status": "Development validation only; official test unused.",
              "pooled": metrics(held.loc[use, "fault_present"], held.loc[use, "prediction"]),
              "eligible_rows": int(held.training_eligible.eq(1).sum()),
              "scored_rows": int(use.sum()), "folds": folds,
              "mean_recording_false_positive_rate": float(np.mean([f["false_positive_rate"] for f in folds])),
              "mean_fault_recording_recall": float(np.mean([f["fault_recall"] for f in folds if f["fault_recall"] is not None])),
              "limitations": plan["limitations"],
              "prepared_csv_sha256": plan["prepared_csv_sha256"]}
    bundle = fit_detector(data)
    bundle["prepared_csv_sha256"] = plan["prepared_csv_sha256"]
    out.mkdir(parents=True)
    held.to_csv(out / "held_out_predictions.csv", index=False)
    joblib.dump(bundle, out / "residual_model.joblib")
    (out / "model_manifest.json").write_text(json.dumps({
        "model_sha256": hashlib.sha256((out / "residual_model.joblib").read_bytes()).hexdigest(),
        "provenance": "Locally rebuilt from LiU training data; not the frozen official-test artifact.",
        "prepared_csv_sha256": plan["prepared_csv_sha256"],
    }, indent=2) + "\n")
    (out / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("POOLED", json.dumps(report["pooled"]), flush=True)


if __name__ == "__main__":
    main()
