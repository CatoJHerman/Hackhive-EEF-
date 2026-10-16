"""Check that temporal features cannot see future samples or cross gaps."""
import unittest
import numpy as np
import pandas as pd
from liu_features import build_features, RAW


class CausalFeatureTests(unittest.TestCase):
    def recording(self):
        t = np.arange(100) / 20
        d = pd.DataFrame({c: 2 + t for c in RAW})
        d["time"] = t
        d["ambient_pressure"] = 100000.0
        return d

    def test_future_changes_do_not_change_past(self):
        original = self.recording()
        changed = original.copy()
        changed.loc[60:, RAW] *= 2
        a, ar = build_features(original, True)
        b, br = build_features(changed, True)
        pd.testing.assert_frame_equal(a.iloc[:60], b.iloc[:60])
        pd.testing.assert_series_equal(ar.iloc[:60], br.iloc[:60])

    def test_prefix_matches_full_recording(self):
        d = self.recording()
        a, ready_a = build_features(d.iloc[:50], True)
        b, ready_b = build_features(d, True)
        pd.testing.assert_frame_equal(a, b.iloc[:50])
        pd.testing.assert_series_equal(ready_a, ready_b.iloc[:50])
        self.assertFalse(ready_a.iloc[:20].any())
        self.assertTrue(ready_a.iloc[20:].all())

    def test_gap_requires_new_history(self):
        d = self.recording()
        d.loc[50:, "time"] += 5
        _, ready = build_features(d, True)
        self.assertFalse(ready.iloc[50:70].any())
        self.assertTrue(ready.iloc[70:].all())

    def test_invalid_ratio_input_rejected(self):
        d = self.recording()
        d.loc[30, "injected_fuel_mass"] = 0
        with self.assertRaises(ValueError):
            build_features(d, True)


if __name__ == "__main__":
    unittest.main()
