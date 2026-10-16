"""Define recording-level validation folds; does not fit models or read test data."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_liu_data import SCENARIOS, label_sample


def define_plan(prepared_dir):
    prepared_dir = Path(prepared_dir)
    data_path = prepared_dir / "liu_training_labelled.csv"
    summary_path = prepared_dir / "preparation_summary.json"
    summary = json.loads(summary_path.read_text())
    counts = {name: Counter() for name in SCENARIOS}
    seen = set()
    with data_path.open(newline="") as source:
        for row in csv.DictReader(source):
            recording = row["recording_id"]
            if recording not in counts:
                raise ValueError(f"Unexpected recording: {recording}")
            identity = (recording, int(row["source_row"]))
            if identity in seen:
                raise ValueError(f"Duplicate row identity: {identity}")
            seen.add(identity)
            expected = label_sample(SCENARIOS[recording], float(row["time"]),
                                    summary["normal_before_seconds"],
                                    summary["fault_after_seconds"])
            actual = tuple(row[c] for c in ("fault_family", "fault_present",
                                            "training_eligible", "label_basis"))
            if actual != tuple(map(str, expected)):
                raise ValueError(f"Label mismatch: {identity}")
            counts[recording]["uncertain" if not expected[2] else
                              "fault" if expected[1] else "normal"] += 1
    for recording in summary["recordings"]:
        actual = counts[recording["recording_id"]]
        if any(actual[key] != recording[key] for key in ("normal", "fault", "uncertain")):
            raise ValueError("Prepared CSV and summary counts differ.")

    folds = []
    for held_out in sorted(counts):
        training = sorted(set(counts) - {held_out})
        train_counts = {key: sum(counts[r][key] for r in training)
                        for key in ("normal", "fault")}
        if not all(train_counts.values()):
            raise ValueError("A training fold lacks one binary class.")
        notes = []
        if held_out == "wltp_NF":
            notes.append("Normal-only validation: report false alarms; recall and ROC AUC are unavailable.")
            notes.append("Training normal examples cover only early pre-fault operation; this is an operating-range stress test.")
        if held_out == "wltp_f_iml_6mm":
            notes.append("No leak recording remains in training: this tests detection of a fault family unseen during this fold.")
        folds.append({
            "fold_id": f"holdout_{held_out}",
            "train_recordings": training,
            "validation_recordings": [held_out],
            "train_eligible_counts": train_counts,
            "validation_eligible_counts": {key: counts[held_out][key]
                                           for key in ("normal", "fault")},
            "validation_excluded_rows": counts[held_out]["uncertain"],
            "interpretation_notes": notes,
        })
    return {
        "version": 1,
        "target": "fault_present",
        "method": "leave_one_recording_out",
        "prepared_csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_archive_sha256": summary["source_sha256"],
        "normal_before_seconds": summary["normal_before_seconds"],
        "fault_after_seconds": summary["fault_after_seconds"],
        "fit_and_metric_filter": "training_eligible == 1",
        "sequence_policy": "Process each recording independently in time order. Reset history between recordings. Uncertain rows may supply causal history but never targets or scored labels.",
        "preprocessing_policy": "Fit preprocessing and any learned features on each fold's training recordings only.",
        "input_exclusions": summary["forbidden_model_inputs"],
        "development_policy": "Use these folds for model and threshold selection. Their results are development estimates, not final independent test scores.",
        "metrics": {
            "per_recording": ["false_positive_rate", "fault_recall", "confusion_counts"],
            "aggregate": "Report pooled confusion counts plus equal-recording mean false-positive rate and mean recall over fault-containing recordings. Show every fold separately.",
            "undefined": "Use null, never zero, when a metric denominator is zero.",
            "delay": "Any detection-delay estimate must state approximate onset and alert persistence rule; report pre-onset alarms separately.",
        },
        "final_test_policy": {
            "source": "https://vehsys.gitlab-pages.liu.se/diagnostic_competition/competition/test_data/testdata.zip",
            "expected_recordings": 17,
            "use": "After freezing features, preprocessing, model settings, threshold and alert rule, refit on all eligible training rows and evaluate the official test recordings once.",
            "restriction": "Never use official test readings to fit preprocessing or select settings. If changes follow test inspection, those scores are no longer an untouched final test.",
        },
        "limitations": [
            "Recording-level separation does not establish independence across engines or vehicles.",
            "Labels around fault onset are provisional; excluded interval is not an official timing bound.",
            "Ambient conditions can identify recordings; compare the baseline without raw ambient inputs during development.",
        ],
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/liu_preparation"))
    args = parser.parse_args()
    plan = define_plan(args.prepared_dir)
    output = args.prepared_dir / "evaluation_plan.json"
    with output.open("x") as destination:
        json.dump(plan, destination, indent=2)
        destination.write("\n")
    print(f"Saved {len(plan['folds'])} validation folds to {output}")
    for fold in plan["folds"]:
        counts = fold["validation_eligible_counts"]
        print(f"{fold['validation_recordings'][0]}: {counts}")


if __name__ == "__main__":
    main()
