"""Upload-driven vibration comparison against the saved reference database."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import stft
from vibration_baseline import AXES, read_signal, window_features, compare_reference

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baselines/vibration_reference.csv"


def vibration_panel():
    st.subheader("Vibration analysis")
    st.caption("Upload a vibration recording. It is compared automatically with the saved baseline.")
    st.info("Current baseline: three synthetic reference recordings at 1,024 Hz. This tests comparison only; it is not a verified engine-idle baseline.")
    upload = st.file_uploader("Vibration recording CSV", type=["csv"], key="vibration_input")
    with st.expander("CSV format and test files"):
        st.code("time_seconds,sampling_rate_hz,accel_x,accel_y,accel_z", language="text")
        st.caption("Use matching sample rate, axes, units and acquisition settings. At least one complete second is required.")
        for scenario in ("matching", "changed"):
            path = ROOT / "demo_samples" / f"synthetic_vibration_{scenario}.csv"
            if path.is_file():
                st.download_button(f"Download {scenario} test CSV", path.read_bytes(), file_name=path.name, mime="text/csv")
    if upload is None:
        st.info("Upload a CSV to see whether its vibration signatures fall outside the baseline range.")
        return
    try:
        data, rate = read_signal(upload.getvalue())
        features = window_features(data, rate)
    except (ValueError, OSError) as exc:
        st.error(f"Vibration input incompatible: {exc}")
        return
    try:
        reference = pd.read_csv(BASELINE)
        result = compare_reference(features, reference)
        outside = result.comparison.isin(["below reference", "above reference"])
        comparable = result.comparison.isin(["below reference", "above reference", "within reference"])
        if outside.any():
            st.warning("Outside baseline range ? vibration differences detected")
        elif comparable.all() and len(result) == 24:
            st.info("Within baseline range ? no outlying signatures detected")
        else:
            st.warning("Unable to fully assess ? insufficient usable baseline or signal features")
        st.caption("This is a descriptive outlier observation, not a fault diagnosis or a probability. Matching the baseline does not establish engine health.")
        with st.expander("Comparison details"):
            st.metric("Signatures outside observed span", f"{int(outside.sum())} / {int(comparable.sum())}")
            st.caption("Each recording contributes one median feature vector. Comparison uses the observed minimum?maximum across reference recordings; brief changes can be hidden by a recording median.")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button("Download vibration comparison", result.to_csv(index=False), file_name=f"{Path(upload.name).stem}_vibration_comparison.csv", mime="text/csv")
    except (ValueError, OSError, KeyError) as exc:
        st.warning(f"Unable to compare with saved baseline: {exc}")
    with st.expander("Waveform, FFT, STFT and extracted features"):
        st.caption(f"{rate:,} Hz ? {len(features)} complete seconds ? {len(data) % rate / rate:.3f} trailing seconds skipped. Amplitude units unverified; features are mean-centered except the mean.")
        signal_details(data, rate, features, upload.name)


def signal_details(data, rate, features, filename):
    axis = st.selectbox("Vibration axis", AXES)
    signal = data[axis].to_numpy()
    stride = max(1, int(np.ceil(len(data) / 15000)))
    fig = go.Figure(go.Scatter(x=data.time_seconds.iloc[::stride], y=signal[::stride], name=axis))
    fig.update_layout(title="Vibration waveform", xaxis_title="Time (seconds)", yaxis_title="Recorded signal amplitude")
    st.plotly_chart(fig, width="stretch")
    index = st.selectbox("Vibration one-second window", range(len(features)))
    segment = signal[index*rate:(index+1)*rate]
    centered = segment - segment.mean()
    window = np.hanning(len(centered))
    magnitude = abs(np.fft.rfft(centered * window)) / window.sum()
    magnitude[1:-1] *= 2
    if len(centered) % 2:
        magnitude[-1] *= 2
    fig = go.Figure(go.Scatter(x=np.fft.rfftfreq(len(centered), 1/rate), y=magnitude))
    fig.update_layout(title="FFT · selected second, DC removed, Hann window", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (uncalibrated)")
    st.plotly_chart(fig, width="stretch")
    n = min(1024, len(centered))
    f, t, z = stft(centered, fs=rate, window="hann", nperseg=n, noverlap=n*3//4,
                   boundary=None, padded=False)
    fig = go.Figure(go.Heatmap(x=t + data.time_seconds.iloc[index*rate], y=f,
                              z=20*np.log10(np.maximum(abs(z), 1e-12)), colorscale="Viridis"))
    fig.update_layout(title="STFT · selected second (dB re 1 amplitude unit)",
                      xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"STFT: Hann window, {n} samples, hop {n-n*3//4}, no padding. Waveform may be subsampled for display; feature calculations use all complete-window samples.")
    st.dataframe(features, hide_index=True, width="stretch")
    st.download_button("Download vibration features", features.to_csv(index=False),
                       file_name=f"{Path(filename).stem}_vibration_features.csv", mime="text/csv")
