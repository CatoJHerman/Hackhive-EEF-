"""Load WAV recordings and export features for non-overlapping windows."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from features import extract_features


def load_audio(path):
    """Return mono full-scale floats, original sample rate and channel count."""
    sample_rate, raw = wavfile.read(path)
    if raw.ndim not in (1, 2) or raw.size == 0:
        raise ValueError("Recording must contain mono or multichannel audio.")
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    if raw.dtype == np.uint8:
        samples = (raw.astype(np.float64) - 128) / 128
    elif np.issubdtype(raw.dtype, np.signedinteger):
        samples = raw.astype(np.float64) / (2 ** (raw.dtype.itemsize * 8 - 1))
    elif np.issubdtype(raw.dtype, np.floating):
        samples = raw.astype(np.float64)
    else:
        raise ValueError(f"Unsupported audio type: {raw.dtype}")
    if sample_rate <= 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Invalid sampling rate or non-finite audio samples.")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples, sample_rate, channels


def iter_windows(samples, sample_rate, window_seconds=2.0):
    """Yield start/end seconds and samples; skip the incomplete final window."""
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window duration must be finite and positive.")
    size = round(window_seconds * sample_rate)
    if size < 4:
        raise ValueError("Each window must contain at least four samples.")
    if len(samples) < size:
        raise ValueError("Recording is shorter than one full window.")
    for start in range(0, len(samples) - size + 1, size):
        end = start + size
        yield start / sample_rate, end / sample_rate, samples[start:end]


def export_features(input_path, output_path, window_seconds=2.0):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from the original audio file.")
    samples, sample_rate, channels = load_audio(input_path)
    rows = []
    for start, end, window in iter_windows(samples, sample_rate, window_seconds):
        rows.append({
            "recording_id": input_path.stem,
            "start_seconds": start,
            "end_seconds": end,
            "sample_rate": sample_rate,
            **extract_features(window, sample_rate),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    duration = len(samples) / sample_rate
    print(f"Loaded {input_path.name}: {duration:.3f}s, {sample_rate} Hz, {channels} channel(s)")
    print(f"Exported {len(rows)} windows to {output_path}")
    print(f"Skipped final {duration - rows[-1]['end_seconds']:.3f}s")
    return rows


def create_demo(path):
    """Create a processing test only: 100 Hz, 400 Hz, then silence."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"Demo file already exists: {path}. Use --input to process it.")
    rate = 8000
    time = np.arange(2 * rate) / rate
    samples = np.concatenate([
        0.5 * np.sin(2 * np.pi * 100 * time),
        0.5 * np.sin(2 * np.pi * 400 * time),
        np.zeros(2 * rate),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, np.round(samples * 32767).astype(np.int16))
    print("Generated synthetic processing test; not engine training data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Existing WAV recording")
    source.add_argument("--demo", action="store_true", help="Generate and process a test WAV")
    parser.add_argument("--output", type=Path, default=Path("outputs/audio_features.csv"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    args = parser.parse_args()
    try:
        path = args.input
        if args.demo:
            path = Path("data/generated_test.wav")
            create_demo(path)
        export_features(path, args.output, args.window_seconds)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Audio processing error: {exc}\n")


if __name__ == "__main__":
    main()
