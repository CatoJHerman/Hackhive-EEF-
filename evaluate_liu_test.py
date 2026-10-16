"""One frozen-model evaluation on the official LiU test archive; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import joblib
import pandas as pd
from predict import predict_recording
from liu_metrics import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-zip", type=Path, required=True)
    args = parser.parse_args()
    output = Path("outputs/liu_residuals_official_test")
    if output.exists():
        raise FileExistsError("Official evaluation already exists. Do not overwrite it.")
    model_path = Path("outputs/liu_residuals/residual_model.joblib")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = joblib.load(model_path)
    results, scored_files = [], []
    with zipfile.ZipFile(args.test_zip) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith("testdata/") and n.endswith(".csv"))
        if len(names) != 17:
            raise ValueError("Expected 17 official test recordings.")
        for name in names:
            recording = PurePosixPath(name).stem
            data = pd.read_csv(archive.open(name))
            score = predict_recording(data, model)
            normal_recording = recording == "wltp_NF_2"
            if not normal_recording and not recording.startswith(("wltp_f_pic_", "wltp_f_pim_", "wltp_f_waf_", "wltp_f_iml_")):
                raise ValueError(f"Unknown documented scenario: {recording}")
            eligible = pd.Series(True, index=data.index) if normal_recording else (data.time < 110) | (data.time > 130)
            target = pd.Series(0, index=data.index) if normal_recording else (data.time > 130).astype(int)
            use = eligible & score.prediction_available
            result = {"recording_id": recording, **metrics(target[use], score.loc[use, "prediction"]),
                      "rows": len(data), "scored_rows": int(use.sum()),
                      "unavailable_rows": int((~score.prediction_available).sum())}
            results.append(result)
            score["recording_id"] = recording
            score["evaluation_eligible"] = use
            score["fault_present"] = target.where(eligible)
            scored_files.append(score)
            print(json.dumps(result), flush=True)
    all_scores = pd.concat(scored_files, ignore_index=True)
    use = all_scores.evaluation_eligible
    report = {"model_sha256": model_hash, "test_zip_sha256": hashlib.sha256(args.test_zip.read_bytes()).hexdigest(),
              "model_frozen_before_test": True, "threshold": model["threshold"],
              "label_policy": "NF_2 normal; other files normal before 110s, faulty after 130s, transition excluded. Approximate timing assumptions, not precise row ground truth.",
              "pooled": metrics(all_scores.loc[use, "fault_present"], all_scores.loc[use, "prediction"]),
              "recordings": results,
              "limitations": "Same benchmark engine; no evidence of generalisation across vehicles. No fault-family diagnosis. Revisions after inspecting this test require new independent evidence."}
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
        raise ValueError("Model changed during evaluation.")
    output.mkdir(parents=True)
    all_scores.to_csv(output / "predictions.csv", index=False)
    (output / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("POOLED", json.dumps(report["pooled"]), flush=True)


if __name__ == "__main__":
    main()
