import unittest
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from fusion_schema import FEATURE_NAMES, SENSORS, PRESETS, input_frame, assign_fault


class FusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(Path(__file__).resolve().parent / "outputs/fusion_model/multimodal_svm.pkl")

    def test_schema_scaling_and_all_presets(self):
        self.assertIsInstance(self.model, Pipeline)
        self.assertEqual(list(self.model.feature_names_in_), FEATURE_NAMES)
        for expected, values in PRESETS.items():
            self.assertEqual(assign_fault(values), expected)
            frame = input_frame(dict(reversed(list(values.items()))))
            self.assertEqual(list(frame), FEATURE_NAMES)
            self.assertEqual(frame.shape, (1, 14))
            self.assertTrue(all(t == np.dtype("float64") for t in frame.dtypes))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(self.model.predict(frame)[0], expected)
                probabilities = self.model.predict_proba(frame)
            self.assertFalse(caught, str(caught))
            self.assertAlmostEqual(float(probabilities.sum()), 1.)
            scaler = self.model.named_steps['standardscaler']
            svc = self.model.named_steps['svc']
            self.assertEqual(svc.predict(scaler.transform(frame))[0], expected)

    def test_reachability_and_strict_boundaries(self):
        for values in PRESETS.values():
            for key, value in values.items():
                self.assertLessEqual(SENSORS[key][0], value)
                self.assertLessEqual(value, SENSORS[key][1])
        boundaries = [("Mechanical Knocking", "accel_z_std", 3.5),
                      ("Valve Train Wear", "engine_speed", 4500.),
                      ("Engine Misfire", "air_mass_flow", 50.),
                      ("Turbocharger Stress", "throttle_position", 80.)]
        for label, key, value in boundaries:
            self.assertEqual(assign_fault({**PRESETS[label], key: value}), "Nominal Operation")

    def test_dashboard_preset_buttons_and_controls(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=30)
        self.assertFalse(list(app.exception), str(list(app.exception)))
        for label, values in PRESETS.items():
            app.button(key="preset_" + label).click().run(timeout=30)
            self.assertFalse(list(app.exception), str(list(app.exception)))
            self.assertFalse(list(app.error), str(list(app.error)))
            self.assertEqual(next(m.value for m in app.metric if m.label == "SVM synthetic scenario"), label)
            for key, value in values.items():
                self.assertEqual(app.slider(key="fusion_" + key).value, value)


if __name__ == "__main__":
    unittest.main()
