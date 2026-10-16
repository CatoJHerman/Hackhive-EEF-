"""Recording-level comparisons and full-dashboard vibration upload checks."""
import io
from pathlib import Path
import unittest
import zipfile
import numpy as np
import pandas as pd
from vibration_baseline import (AXES, COMPARISON, read_signal, window_features,
                                reference_recordings, compare_reference)


def signal_csv(amplitude=1., seconds=2):
    rate = 128
    time = np.arange(rate * seconds) / rate
    frame = pd.DataFrame({"time_seconds": time, "sampling_rate_hz": rate})
    for i, axis in enumerate(AXES):
        frame[axis] = amplitude * np.sin(2 * np.pi * (10 + i) * time)
    return frame.to_csv(index=False).encode()


def reference_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i, amplitude in enumerate([.9, 1., 1.1]):
            archive.writestr(f"reference_{i}.csv", signal_csv(amplitude))
    return buffer.getvalue()


class VibrationChecks(unittest.TestCase):
    def test_one_vector_per_recording_and_changed_signal(self):
        refs = reference_recordings(reference_zip(), "reference.zip")
        self.assertEqual(len(refs), 3)
        self.assertEqual(refs.reference_windows.tolist(), [2, 2, 2])
        data, rate = read_signal(signal_csv(4))
        result = compare_reference(window_features(data, rate), refs)
        self.assertTrue(result.loc[result.feature.eq("rms"), "comparison"].eq("above reference").all())

    def test_one_and_two_references_are_descriptive(self):
        data, rate = read_signal(signal_csv())
        features = window_features(data, rate)
        refs = reference_recordings(reference_zip(), "reference.zip")
        for count in (1, 2):
            self.assertEqual(set(compare_reference(features, refs.iloc[:count]).comparison), {"descriptive only"})
        single = reference_recordings(signal_csv(seconds=5), "idle.csv")
        self.assertEqual(len(single), 1)

    def test_invalid_schema_timestamps_and_duplicate_zip(self):
        with self.assertRaisesRegex(ValueError, "Expected time_seconds"):
            read_signal(b"Time,Channel 1\n0,1\n")
        frame = pd.read_csv(io.BytesIO(signal_csv()))
        frame.loc[1, "time_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "Timestamps"):
            read_signal(frame.to_csv(index=False).encode())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("a.csv", "b.csv"):
                archive.writestr(name, signal_csv())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reference_recordings(buffer.getvalue(), "duplicate.zip")

    def test_missing_feature_values_do_not_establish_range(self):
        data, rate = read_signal(signal_csv())
        features = window_features(data, rate)
        refs = reference_recordings(reference_zip(), "reference.zip")
        refs.loc[0, COMPARISON] = np.nan
        self.assertEqual(set(compare_reference(features, refs).comparison), {"descriptive only"})

    def test_preloaded_demos(self):
        folder = Path(__file__).resolve().parent / "demo_samples"
        refs = reference_recordings((folder / "synthetic_vibration_reference.zip").read_bytes(), "refs.zip")
        for scenario, expected in [("matching", 0), ("changed", 13)]:
            data, rate = read_signal((folder / f"synthetic_vibration_{scenario}.csv").read_bytes())
            result = compare_reference(window_features(data, rate), refs)
            self.assertEqual(int(result.comparison.isin(["above reference", "below reference"]).sum()), expected)

    def test_full_dashboard_upload_modes(self):
        from streamlit.testing.v1 import AppTest
        root = str(Path(__file__).resolve().parent)
        for scenario in ("matching", "changed", "invalid"):
            source = f"""
import io, runpy
from pathlib import Path
from unittest.mock import patch
root = Path({root!r})
scenario = {scenario!r}
def uploader(label, **kwargs):
    if kwargs.get("key") != "vibration_input": return None
    raw = b"Time,Channel 1\\n0,1\\n" if scenario == "invalid" else (root / "demo_samples" / f"synthetic_vibration_{{scenario}}.csv").read_bytes()
    result = io.BytesIO(raw); result.name = "input.csv"; return result
with patch("streamlit.file_uploader", uploader):
    runpy.run_path(str(root / "dashboard.py"), run_name="__main__")
"""
            app = AppTest.from_string(source).run(timeout=30)
            self.assertFalse(list(app.exception), str(list(app.exception)))
            if scenario == "invalid":
                self.assertTrue(any("Vibration input incompatible" in x.value for x in app.error))
            else:
                self.assertFalse(list(app.error))
                messages = list(app.warning) + list(app.info)
                expected = "Outside baseline range" if scenario == "changed" else "Within baseline range"
                self.assertTrue(any(expected in x.value for x in messages))


if __name__ == "__main__":
    unittest.main()
