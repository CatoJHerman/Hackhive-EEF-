"""Train an SVM to approximate synthetic labeling rules, not diagnose real engines."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from fusion_schema import SENSORS, FEATURE_NAMES, PRESETS, assign_fault, input_frame

ROOT = Path(__file__).resolve().parent


def generate_data(per_class=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    constraints = {
        "Mechanical Knocking": {"accel_z_std": (3.5001, 5), "dominant_frequency_hz": (40, 999.99), "intake_manifold_pressure": (120000.01, 150000)},
        "Valve Train Wear": {"dominant_frequency_hz": (4000.01, 5000), "engine_speed": (4500.01, 6000), "intercooler_temperature": (350.001, 380)},
        "Engine Misfire": {"engine_speed": (800, 1199.99), "accel_z_std": (4.0001, 5), "air_mass_flow": (10, 49.999)},
        "Turbocharger Stress": {"throttle_position": (80.001, 100), "wastegate_position": (0, 19.999), "Intercooler_pressure": (140000.01, 160000)},
    }
    for label in PRESETS:
        count = 0
        while count < per_class:
            row = {key: rng.uniform(spec[0], spec[1]) for key, spec in SENSORS.items()}
            for key, bounds in constraints.get(label, {}).items():
                row[key] = rng.uniform(*bounds)
            if assign_fault(row) == label:
                rows.append(row)
                labels.append(label)
                count += 1
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float), pd.Series(labels, name="fault_class")


def main():
    X, y = generate_data()
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), SVC(C=30, kernel="rbf", probability=True, random_state=42))
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    model.fusion_schema_version = 1
    model.data_origin = "synthetic_rule_demo"
    report = {"data_origin": model.data_origin, "generated_rows": len(X), "training_rows": len(train_x),
              "held_out_rows": len(test_x), "features": FEATURE_NAMES,
              "class_counts": y.value_counts().to_dict(),
              "classification_report": classification_report(test_y, predicted, output_dict=True),
              "classes": model.classes_.tolist(), "confusion_matrix": confusion_matrix(test_y, predicted, labels=model.classes_).tolist(),
              "preset_predictions": {label: str(model.predict(input_frame(values))[0]) for label, values in PRESETS.items()},
              "limitations": "Held-out synthetic rows measure reproduction of hand-coded rules only. Independent random draws are not synchronized physical measurements. No real-engine accuracy or real-world fault probabilities are established."}
    out = ROOT / "outputs/fusion_model"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multimodal_svm.pkl"
    if path.exists() and not (out / "multimodal_svm.before_fix.pkl").exists():
        (out / "multimodal_svm.before_fix.pkl").write_bytes(path.read_bytes())
    joblib.dump(model, path)
    (out / "synthetic_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
