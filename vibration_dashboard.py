"""Independent vibration panel; no classifier or simulated telemetry is used."""
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import stft

from vibration_baseline import (PACKAGE, AXES, read_signal, window_features,
                                demo_reference, reference_recordings, compare_reference)


def vibration_panel():
    st.subheader("Vibration reference comparison")
    st.caption("Independent vibration analysis. Audio and telemetry are not synchronized with these recordings.")
    mode = st.radio("Reference source", ["Upload idle reference", "Simulated normal demo"],
                    key="vibration_reference_mode")
    reference_upload = None
    if mode == "Simulated normal demo":
        st.warning("Synthetic demonstration only. These normal examples span several operating speeds and are not a verified idle dataset. No fault classification is performed.")
        samples = sorted((PACKAGE / "samples").glob("*.csv"))
        if samples:
            sample = st.selectbox("Download a supplied demo recording", samples, format_func=lambda p: p.stem)
            st.download_button("Download vibration demo CSV", sample.read_bytes(),
                               file_name=sample.name, mime="text/csv")
    else:
        st.info("Use separate idle recordings with matching sensor axes, units, mounting and acquisition settings. A ZIP can contain several CSV recordings. Reference provenance is user supplied, not independently verified.")
        reference_upload = st.file_uploader("Idle vibration reference CSV or ZIP", type=["csv", "zip"],
                                            key="vibration_reference")
    st.code("time_seconds,sampling_rate_hz,accel_x,accel_y,accel_z", language="text")
    upload = st.file_uploader("Vibration recording CSV", type=["csv"], key="vibration_input")
    if upload is None:
        st.info("Upload a waveform recording to view its features and compare with the selected reference.")
        return
    try:
        data, rate = read_signal(upload.getvalue())
        features = window_features(data, rate)
    except (ValueError, OSError) as exc:
        st.error(f"Vibration input incompatible: {exc}")
        return
    st.caption(f"Analysed file: **{upload.name}** · {rate:,} Hz declared and checked against timestamps · "
               f"{len(data):,} samples · {len(features)} complete one-second windows · "
               f"{len(data) % rate / rate:.3f} trailing seconds skipped. "
               "Amplitude units are unverified. Features use mean-centered signals except the reported mean.")
    try:
        if mode == "Simulated normal demo":
            reference = demo_reference()
            source = "Synthetic normal examples (not measured idle)"
        elif reference_upload is not None:
            reference = reference_recordings(reference_upload.getvalue(), reference_upload.name)
            source = f"Uploaded idle reference: {reference_upload.name}"
        else:
            reference = None
            source = ""
        if reference is not None:
            result = compare_reference(features, reference)
            st.caption(f"Reference source: **{source}** · {len(reference)} reference vector(s). "
                       "The uploaded CSV and each uploaded reference CSV contribute one median feature vector. "
                       "Separate files are not necessarily independent engines or sessions.")
            changed = result.comparison.isin(["below reference", "above reference"])
            if len(reference) < 3:
                st.info("One or two reference recordings: feature values are displayed, but there is too little reference spread to state whether the recording differs from an idle range.")
            else:
                count = int(changed.sum())
                available = int(result.comparison.ne("unavailable").sum())
                if mode == "Simulated normal demo":
                    st.info(f"Synthetic demo: {count} of {available} available signatures outside its generated reference span.")
                elif count:
                    st.warning("Some vibration signatures are outside the supplied idle reference span")
                else:
                    st.info("No available recording-level signatures outside the supplied idle reference span")
                st.metric("Signatures outside observed span", f"{count} / {available}")
            st.caption("The reference span uses observed minimum–maximum values across separate recordings. "
                       "It is not a calibrated diagnostic threshold or fault probability. "
                       "Unavailable features do not establish a match. Operating-condition changes can cause differences, "
                       "and matching the span does not prove engine health.")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button("Download vibration comparison", result.to_csv(index=False),
                               file_name=f"{Path(upload.name).stem}_vibration_comparison.csv", mime="text/csv")
    except (ValueError, OSError, KeyError) as exc:
        st.warning(f"Reference comparison unavailable: {exc}")
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
                       file_name=f"{Path(upload.name).stem}_vibration_features.csv", mime="text/csv")
