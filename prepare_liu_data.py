"""Prepare LiU training labels without changing source readings or training a model.

Run: python prepare_liu_data.py --input /path/to/trainingdata.zip
Uses only Python's standard library. Outputs are deliberately under outputs/.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


COLUMNS = [
    "time", "Intercooler_pressure", "intercooler_temperature",
    "intake_manifold_pressure", "air_mass_flow", "engine_speed",
    "throttle_position", "wastegate_position", "injected_fuel_mass",
    "ambient_pressure", "ambient_temperature",
]
SCENARIOS = {
    "wltp_NF": "normal",
    "wltp_f_pim_080": "intake_manifold_pressure_sensor",
    "wltp_f_pim_090": "intake_manifold_pressure_sensor",
    "wltp_f_pic_090": "intercooler_pressure_sensor",
    "wltp_f_pic_110": "intercooler_pressure_sensor",
    "wltp_f_waf_105": "air_mass_flow_sensor",
    "wltp_f_waf_110": "air_mass_flow_sensor",
    "wltp_f_iml_6mm": "intake_manifold_leak",
}
METADATA = ["recording_id", "source_row", "scenario", "fault_family",
            "fault_present", "training_eligible", "label_basis"]


def label_sample(scenario, time, normal_before=110.0, fault_after=130.0):
    """Boundary rows are uncertain; timing limits are assumptions, not ground truth."""
    if scenario == "normal":
        return "normal", 0, 1, "documented_normal_recording"
    if time < normal_before:
        return "normal", 0, 1, "assumed_pre_fault"
    if time > fault_after:
        return scenario, 1, 1, "assumed_post_fault"
    return "uncertain", "", 0, "approximate_onset_excluded"


def prepare(input_path, output_dir, normal_before=110.0, fault_after=130.0):
    if not (math.isfinite(normal_before) and math.isfinite(fault_after)
            and 0 <= normal_before < fault_after):
        raise ValueError("Require 0 <= normal-before < fault-after, both finite.")
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_csv = output_dir / "liu_training_labelled.csv"
    output_summary = output_dir / "preparation_summary.json"
    if output_csv.exists() or output_summary.exists():
        raise FileExistsError("Output already exists; choose a new --output-dir.")

    # Validate all source data before writing any output. Never extract ZIP paths.
    rows, recordings = [], []
    with zipfile.ZipFile(input_path) as archive:
        names = [n for n in archive.namelist()
                 if n.startswith("trainingdata/") and n.endswith(".csv")]
        ids = [PurePosixPath(n).stem for n in names]
        if len(ids) != len(SCENARIOS) or set(ids) != set(SCENARIOS):
            raise ValueError("Expected exactly the eight documented training CSVs.")
        for name in sorted(names):
            recording_id = PurePosixPath(name).stem
            scenario = SCENARIOS[recording_id]
            counts = {"normal": 0, "fault": 0, "uncertain": 0}
            previous_time = -math.inf
            with archive.open(name) as source:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
                if reader.fieldnames != COLUMNS:
                    raise ValueError(f"Unexpected columns in {name}")
                for row_number, row in enumerate(reader, 1):
                    if None in row or any(row[c] is None for c in COLUMNS):
                        raise ValueError(f"Malformed row: {name}:{row_number}")
                    values = [float(row[c]) for c in COLUMNS]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f"Non-finite value: {name}:{row_number}")
                    time = values[0]
                    if time < 0 or time <= previous_time:
                        raise ValueError(f"Invalid time order: {name}:{row_number}")
                    previous_time = time
                    family, target, eligible, basis = label_sample(
                        scenario, time, normal_before, fault_after)
                    counts["uncertain" if not eligible else "fault" if target else "normal"] += 1
                    # Original numeric strings are retained, without scaling or conversion.
                    row.update(dict(zip(METADATA, [recording_id, row_number, scenario,
                                                   family, target, eligible, basis])))
                    rows.append(row)
            if sum(counts.values()) == 0:
                raise ValueError(f"Empty recording: {name}")
            recordings.append({"recording_id": recording_id, "scenario": scenario,
                               "rows": sum(counts.values()), **counts})

    totals = {key: sum(r[key] for r in recordings)
              for key in ("rows", "normal", "fault", "uncertain")}
    summary = {
        "source_archive": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "documentation": "https://vehsys.gitlab-pages.liu.se/diagnostic_competition/index_competition_24",
        "normal_before_seconds": normal_before,
        "fault_after_seconds": fault_after,
        "timing_status": "Provisional assumptions around documented approximate 120-second onset.",
        "units": "Original CSV scale retained; complete physical-unit dictionary unverified.",
        "target": "fault_present: 0 normal, 1 fault, blank uncertain",
        "training_filter": "Use only training_eligible == 1; never fill uncertain targets with zero.",
        "forbidden_model_inputs": ["time", *METADATA],
        "evaluation": "Keep source recordings grouped. Official test data is not included here.",
        "source_row_definition": "1-based data row, excluding CSV header",
        "totals": totals, "recordings": recordings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS + METADATA)
        writer.writeheader()
        writer.writerows(rows)
    with output_summary.open("x") as destination:
        json.dump(summary, destination, indent=2)
        destination.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/liu_preparation"))
    parser.add_argument("--normal-before", type=float, default=110.0)
    parser.add_argument("--fault-after", type=float, default=130.0)
    args = parser.parse_args()
    summary = prepare(args.input, args.output_dir, args.normal_before, args.fault_after)
    print(json.dumps(summary["totals"], indent=2))
    print(f"Prepared data and summary: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
