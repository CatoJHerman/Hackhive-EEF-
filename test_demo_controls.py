from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class DemoControlsChecks(unittest.TestCase):
    def test_judging_buttons_and_clear(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=30)
        checks = [
            ("telemetry_load_liu_normal_training", "Analyzed readings with alerts"),
            ("telemetry_load_liu_pressure_fault_training", "Analyzed readings with alerts"),
            ("audio_load_synthetic_audio_comparison", "Windows differing from idle"),
            ("vibration_load_synthetic_vibration_changed", "Signatures outside observed span"),
            ("vibration_load_synthetic_vibration_matching", "Signatures outside observed span"),
        ]
        for button, label in checks:
            app.button(key=button).click().run(timeout=30)
            self.assertFalse(list(app.exception), str(list(app.exception)))
            self.assertFalse(list(app.error), str(list(app.error)))
            self.assertIn(label, [m.label for m in app.metric])
        self.assertEqual(next(m.value for m in app.metric if m.label == "Signatures outside observed span"), "0 / 24")
        for key in ("telemetry", "audio", "vibration"):
            app.button(key=key + "_clear").click().run(timeout=30)
        self.assertNotIn("Signatures outside observed span", [m.label for m in app.metric])
        self.assertNotIn("Windows differing from idle", [m.label for m in app.metric])


if __name__ == "__main__":
    unittest.main()
