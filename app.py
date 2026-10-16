"""Canonical launcher for the upload-driven LiU and audio dashboard."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("dashboard.py")), run_name="__main__")
