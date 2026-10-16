import unittest
import numpy as np
import pandas as pd
from audio_baseline import FEATURES, compare_idle


class IdleComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = pd.DataFrame({key: np.arange(1., 101.) for key in FEATURES})
        self.baseline["sample_rate"] = 44100
        self.baseline["start_seconds"] = 0.
        self.baseline["end_seconds"] = 1.

    def test_in_range_and_single_feature_difference(self):
        upload = self.baseline.iloc[[49, 50]].copy().reset_index(drop=True)
        upload.loc[1, "rms"] = 200
        table, windows, summary = compare_idle(upload, self.baseline)
        self.assertEqual(windows.differs_from_idle.tolist(), [False, True])
        self.assertEqual(windows.different_features.tolist(), ["", "rms"])
        self.assertEqual(summary["different_windows_percent"], 50)
        self.assertEqual(table.set_index("feature").loc["rms", "idle_median"], 50.5)

    def test_sample_rate_and_bad_data(self):
        for column, value in [("sample_rate", 8000), ("end_seconds", 2), ("rms", np.nan)]:
            upload = self.baseline.iloc[:1].copy()
            upload[column] = value
            with self.assertRaises(ValueError):
                compare_idle(upload, self.baseline)

    def test_constant_baseline_and_no_input_mutation(self):
        baseline = self.baseline.copy()
        baseline[FEATURES] = 0.
        before = baseline.copy()
        _, _, summary = compare_idle(baseline.iloc[:1], baseline)
        self.assertEqual(summary["different_windows_percent"], 0)
        pd.testing.assert_frame_equal(baseline, before)


if __name__ == "__main__":
    unittest.main()
