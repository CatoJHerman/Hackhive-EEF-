import numpy as np
import matplotlib.pyplot as plt


def frequency_spectrum(samples, sample_rate):
    """Return frequencies and FFT magnitudes after removing DC offset."""
    centered = samples - np.mean(samples)
    windowed = centered * np.hanning(len(centered))

    frequencies = np.fft.rfftfreq(len(samples), d=1 / sample_rate)
    magnitudes = np.abs(np.fft.rfft(windowed))

    return frequencies, magnitudes


def extract_features(samples, sample_rate):
    """Accept one window of mono floating-point audio."""
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 1 or len(samples) < 4:
        raise ValueError("Provide a mono audio window with at least 4 samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Audio contains missing or invalid values.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    crest_factor = peak / rms if rms > 0 else 0.0

    frequencies, magnitudes = frequency_spectrum(samples, sample_rate)

    # Silence or a constant signal has no meaningful dominant frequency.
    dominant_frequency = 0.0
    if np.max(np.abs(samples - np.mean(samples))) > 1e-12:
        peak_index = np.argmax(magnitudes[1:]) + 1
        dominant_frequency = float(frequencies[peak_index])

    return {
        "rms": rms,
        "peak_amplitude": peak,
        "crest_factor": crest_factor,
        "dominant_frequency_hz": dominant_frequency,
    }


def plot_signal(samples, sample_rate):
    """Create plots separately so training can run without opening charts."""
    samples = np.asarray(samples, dtype=float)
    time = np.arange(len(samples)) / sample_rate
    frequencies, magnitudes = frequency_spectrum(samples, sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(time, samples)
    axes[0].set_title("Waveform — first 0.05 seconds")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, min(0.05, time[-1]))

    axes[1].plot(frequencies, magnitudes)
    axes[1].set_title("Frequency spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("FFT magnitude")
    axes[1].set_xlim(0, sample_rate / 2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Two seconds of a known signal to check our calculations.
    sample_rate = 8000
    time = np.arange(2 * sample_rate) / sample_rate
    samples = 0.5 * np.sin(2 * np.pi * 100 * time)

    features = extract_features(samples, sample_rate)
    for name, value in features.items():
        print(f"{name}: {value:.4f}")

    assert abs(features["dominant_frequency_hz"] - 100) < 1
    assert np.isclose(features["rms"], 0.5 / np.sqrt(2))

    silence = extract_features(np.zeros(sample_rate), sample_rate)
    assert all(value == 0 for value in silence.values())
    print("Tone and silence checks passed.")

    plot_signal(samples, sample_rate)
    plt.show()