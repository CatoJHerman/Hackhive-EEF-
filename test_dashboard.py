"""Run with .venv-liu/bin/python -m unittest test_dashboard.py."""
import io
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.io import wavfile
from audio_processing import load_audio, iter_windows
from dashboard_logic import REQUIRED, read_telemetry, alert_intervals, recording_quality, analysis_summary, inspection_guidance


class DashboardChecks(unittest.TestCase):
    def test_all_three_branches_together(self):
        from streamlit.testing.v1 import AppTest
        root = str(Path(__file__).resolve().parent)
        source = f'''
import io, runpy
from pathlib import Path
from unittest.mock import patch
root = Path({root!r})
def uploader(label, **kwargs):
    names = {{"telemetry_upload": "synthetic_telemetry_demo.csv", "audio_upload": "synthetic_audio_comparison.wav", "vibration_input": "synthetic_vibration_changed.csv"}}
    name = names.get(kwargs.get("key"))
    if name is None: return None
    result = io.BytesIO((root / "demo_samples" / name).read_bytes())
    result.name = name
    return result
with patch("streamlit.file_uploader", uploader):
    runpy.run_path(str(root / "dashboard.py"), run_name="__main__")
'''
        app = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(list(app.exception), str(list(app.exception)))
        self.assertFalse(list(app.error), str(list(app.error)))
        labels = [metric.label for metric in app.metric]
        for label in ("Engine fault risk", "Windows differing from idle", "Signatures outside observed span"):
            self.assertIn(label, labels)

    def test_inspection_guidance_does_not_invent_risk(self):
        rows = pd.DataFrame({"prediction_available": [False, True, True],
                             "status": ["anomaly_detected", "anomaly_detected", "no_anomaly_detected"]})
        result = inspection_guidance(rows)
        self.assertEqual(result["alert_percentage"], 50.)
        self.assertIsNone(result["fault_risk_percentage"])
        self.assertIn("Inspection recommended", result["title"])
        self.assertIn("Demo only", inspection_guidance(rows, synthetic=True)["title"])
        self.assertIsNone(inspection_guidance(rows.iloc[:1])["alert_percentage"])
        self.assertIn("Unable to assess", inspection_guidance(rows.iloc[:1])["title"])
        clear = inspection_guidance(rows.iloc[2:])
        self.assertEqual(clear["alert_percentage"], 0.)
        self.assertIn("cannot establish", clear["message"])

    def test_quality_single_reading_and_gaps(self):
        frame = pd.DataFrame({c: [1., 1., 1.] for c in REQUIRED})
        frame["time"] = [0., .05, 1.]
        self.assertEqual(recording_quality(frame)["gaps_over_0_1_seconds"], 1)
        self.assertEqual(recording_quality(frame)["constant_signals"], REQUIRED[1:])
        self.assertIsNone(recording_quality(frame.iloc[:1])["median_interval_seconds"])

    def test_summary_excludes_unavailable_readings(self):
        import json
        rows = pd.DataFrame({"time": [0., .05, .1],
                             "prediction_available": [False, True, True],
                             "status": ["insufficient_history", "anomaly_detected", "no_anomaly_detected"],
                             "anomaly_score": [np.nan, 2., .5]})
        summary = analysis_summary(rows, "test.csv", "model-hash")
        self.assertEqual(summary["flagged_fraction_of_available"], .5)
        self.assertEqual(summary["peak_anomaly_score"], 2.)
        self.assertEqual(len(summary["alert_intervals"]), 1)
        empty = analysis_summary(rows.iloc[:1], "test.csv", "model-hash")
        self.assertEqual(empty["status"], "insufficient_history")
        self.assertIsNone(empty["flagged_fraction_of_available"])
        json.dumps(empty, allow_nan=False)

    def test_optional_invalid_values_are_rejected(self):
        data = pd.DataFrame({c: [1., 2.] for c in REQUIRED})
        data["ambient_pressure"] = [100000, "invalid"]
        with self.assertRaises(ValueError):
            read_telemetry(data.to_csv(index=False).encode())

    def test_gaps_and_unavailable_rows_break_alerts(self):
        rows = pd.DataFrame({"time": [0, .05, .1, .15, 1., 1.05],
                             "prediction_available": [True, True, False, True, True, True],
                             "status": ["anomaly_detected"] * 6,
                             "anomaly_score": [2., 3., np.nan, 4., 5., 6.]})
        result = alert_intervals(rows)
        self.assertEqual(result.start_seconds.tolist(), [0, .15, 1])
        self.assertEqual(result.max_anomaly_score.tolist(), [3, 4, 6])

    def test_wav_in_memory_and_partial_window(self):
        buffer = io.BytesIO()
        wavfile.write(buffer, 8000, np.zeros(12000, dtype=np.int16))
        buffer.seek(0)
        samples, rate, channels = load_audio(buffer)
        self.assertEqual(channels, 1)
        self.assertEqual(len(list(iter_windows(samples, rate, 1))), 1)

    def test_both_launchers_empty_state(self):
        from streamlit.testing.v1 import AppTest
        for entry in ("app.py", "dashboard.py"):
            app = AppTest.from_file(str(Path(__file__).with_name(entry))).run(timeout=30)
            self.assertFalse(list(app.exception), str(list(app.exception)))
            self.assertEqual([tab.label for tab in app.tabs], ["Engine telemetry", "Engine audio", "Vibration", "Model evaluation"])

    def test_upload_paths(self):
        from streamlit.testing.v1 import AppTest
        # Upload widgets are supplied through a mock; all UI/backend code executes.
        root = str(Path(__file__).resolve().parent)
        for kind in ("csv", "wav", "idle_wav", "invalid_csv", "short_wav"):
            source = f'''
import io, runpy
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from dashboard_logic import REQUIRED
kind = {kind!r}
if kind in ("csv", "invalid_csv"):
    frame = pd.DataFrame({{c: np.ones(40) for c in REQUIRED}})
    frame["time"] = np.arange(40) / 20
    if kind == "invalid_csv": frame = frame.drop(columns="engine_speed")
    upload = io.BytesIO(frame.to_csv(index=False).encode()); upload.name = "test.csv"
else:
    upload = io.BytesIO()
    rate = 44100 if kind == "idle_wav" else 8000
    wavfile.write(upload, rate, np.zeros(4000 if kind == "short_wav" else rate * 2, dtype=np.int16))
    upload.name = "test.wav"
def uploader(label, **kwargs):
    target = "telemetry_upload" if kind in ("csv", "invalid_csv") else "audio_upload"
    return upload if kwargs.get("key") == target else None
with patch("streamlit.file_uploader", uploader):
    runpy.run_path({root!r} + "/dashboard.py", run_name="__main__")
'''
            app = AppTest.from_string(source).run(timeout=30)
            self.assertFalse(list(app.exception), f"{kind}: {list(app.exception)}")
            if kind == "invalid_csv":
                self.assertTrue(any("Missing required" in e.value for e in app.error))
            elif kind == "short_wav":
                self.assertTrue(any("too short" in e.value for e in app.warning))
            else:
                self.assertFalse(list(app.error), f"{kind}: {list(app.error)}")
            if kind == "wav":
                app.checkbox[0].check().run(timeout=30)
                self.assertFalse(list(app.exception), str(list(app.exception)))
            if kind == "idle_wav":
                self.assertTrue(any("Differences from the idle baseline found" in e.value for e in app.warning))


if __name__ == "__main__":
    unittest.main()
