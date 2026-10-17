"""Upload-driven experimental engine-signal dashboard. Run: streamlit run dashboard.py"""

from __future__ import annotations

import json
import io
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import stft

from audio_processing import iter_windows, load_audio
from audio_baseline import compare_idle
from dashboard_logic import (FEATURES, OPTIONAL, REQUIRED, alert_intervals,
                             read_telemetry, recording_quality, analysis_summary, inspection_guidance)
from features import extract_features, frequency_spectrum
from fusion_schema import SENSORS, FEATURE_NAMES, PRESETS, assign_fault, input_frame
from predict import predict_recording
from demo_controls import example_input
from vibration_dashboard import vibration_panel

ROOT = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("ENGINE_MODEL_PATH", str(ROOT / "outputs/liu_residuals/residual_model.joblib")))
BASELINE = ROOT / "baselines/normal_idle_features_1s.csv"

# Premium Corporate Color Palette
BLUE, ORANGE, RED, GREY = "#00d4ff", "#ff9d00", "#ff3366", "#6c7a89"
BG, PANEL, TEXT = "#090d14", "#131b2b", "#f0f4f8"
SUCCESS = "#00e676"

st.set_page_config(page_title="Engine Diagnostics Command Center", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# Executive Dashboard CSS
st.markdown(f"""<style>
.stApp {{background:{BG}; color:{TEXT}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}}
.block-container {{max-width:98%; padding-top:1.5rem; padding-bottom:3rem;}}
h1,h2,h3 {{font-weight: 700; letter-spacing: -0.5px;}}
h1 {{font-size:2.4rem!important; border-bottom: 2px solid #1a2538; padding-bottom: 10px; margin-bottom: 20px;}}
[data-testid="stMetric"] {{background:{PANEL}; border:1px solid #233147; border-radius:8px; padding:12px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);}}
[data-testid="stMetricLabel"] {{color:#8b9eb3; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.75rem;}}
[data-testid="stMetricValue"] {{color:{TEXT}; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 1.6rem; font-weight: 600;}}
[data-testid="stExpander"] {{background:{PANEL}; border: 1px solid #233147; border-radius: 8px;}}
.status-indicator {{font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;}}
button[data-baseweb="tab"] {{font-weight:700; font-size: 1.05rem;}}
button[data-baseweb="tab"][aria-selected="true"] {{color:{BLUE}; border-bottom-color: {BLUE};}}
</style>""", unsafe_allow_html=True)


def figure(title, y_label, height=300):
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=17, color='#8b9eb3')), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                      margin=dict(l=42, r=24, t=54, b=42), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(title="Time (seconds)", gridcolor="#1a2538", zeroline=False),
                      yaxis=dict(title=y_label, gridcolor="#1a2538", zeroline=False))
    return fig


def line(fig, x, y, name, color):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2),
                             connectgaps=False))


def chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def model_identity():
    manifest = MODEL.with_name("model_manifest.json")
    official = json.loads((ROOT / "reports/official_test.json").read_text())
    if manifest.is_file():
        identity = json.loads(manifest.read_text())
        if identity["model_sha256"] != official["model_sha256"]:
            return identity["model_sha256"], "Locally rebuilt model · official benchmark does not apply"
    return official["model_sha256"], "Frozen model · official benchmark artifact"


@st.cache_resource(show_spinner="Loading frozen model…")
def load_model(path: str, expected_hash: str, file_stamp: tuple):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("Model checksum does not match its manifest or official report. Restore the matching model artifact.")
    return joblib.load(io.BytesIO(content))


@st.cache_data(max_entries=3, show_spinner="Analysing uploaded telemetry…")
def analyse_telemetry(content: bytes, model_path: str, expected_hash: str, file_stamp: tuple):
    return predict_recording(read_telemetry(content), load_model(model_path, expected_hash, file_stamp))


def create_radar_chart(features_dict):
    categories = ['RPM', 'Temp', 'Vibration', 'Acoustic Freq', 'Intake Press', 'Air Mass']
    values = [
        min(100, (features_dict['engine_speed'] / 6000) * 100),
        min(100, ((features_dict['intercooler_temperature'] - 270) / 110) * 100),
        min(100, (features_dict['accel_z_std'] / 5.0) * 100),
        min(100, (features_dict['dominant_frequency_hz'] / 5000) * 100),
        min(100, ((features_dict['intake_manifold_pressure'] - 80000) / 70000) * 100),
        min(100, (features_dict['air_mass_flow'] / 500) * 100)
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(0, 212, 255, 0.2)', line=dict(color=BLUE, width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1a2538', tickfont=dict(color='#8b9eb3')),
                   angularaxis=dict(gridcolor='#1a2538', tickfont=dict(color=TEXT))),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20), height=350
    )
    return fig


@st.cache_resource(show_spinner="Loading synthetic fusion SVM...")
def load_fusion_model(path, revision):
    model = joblib.load(path)
    if getattr(model, "fusion_schema_version", None) != 1 or list(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("Fusion artifact is outdated. Run train_fusion_model.py to rebuild the scaled pipeline.")
    if "standardscaler" not in model.named_steps:
        raise ValueError("Fusion model must include StandardScaler.")
    return model


def apply_fusion_preset(label):
    for key, value in PRESETS[label].items():
        st.session_state["fusion_" + key] = float(value)


def multimodal_fusion():
    st.subheader("Multimodal fusion simulator")
    path = ROOT / "outputs/fusion_model/multimodal_svm.pkl"
    try:
        stat = path.stat()
        fusion_model = load_fusion_model(str(path), (stat.st_mtime_ns, stat.st_size))
    except Exception as exc:
        st.error(f"Fusion unavailable: {exc}. Run train_fusion_model.py to initialize.")
        return
    st.caption("Load a reproducible scenario, then adjust any of the 14 controls. All inference goes through StandardScaler and SVC; no rule overrides the model prediction.")
    for column, label in zip(st.columns(5), PRESETS):
        column.button(label, key="preset_" + label, on_click=apply_fusion_preset, args=(label,))
    features = {}
    for title, names in [("Acoustic features", FEATURE_NAMES[:2]),
                         ("Vibration features", FEATURE_NAMES[2:4]),
                         ("ECU features", FEATURE_NAMES[4:])]:
        with st.expander(title, expanded=True):
            columns = st.columns(2 if len(names) == 2 else 5)
            for i, key in enumerate(names):
                low, high, default, label = SENSORS[key]
                widget = "fusion_" + key
                if widget not in st.session_state:
                    st.session_state[widget] = default
                features[key] = columns[i % len(columns)].slider(label, min_value=low, max_value=high,
                                                                  key=widget, format="%.2f")
    try:
        frame = input_frame(features)
        prediction = str(fusion_model.predict(frame)[0])
        probabilities = fusion_model.predict_proba(frame)[0]
    except (ValueError, TypeError) as exc:
        st.error(f"Fusion inference failed: {exc}")
        return
    radar_col, diagnosis_col = st.columns([1, 2])
    with radar_col:
        chart(create_radar_chart(features))
        st.caption("Radar shows simulator range positions, not risk percentages.")
    with diagnosis_col:
        st.metric("SVM synthetic scenario", prediction)
        st.write("Nominal demo pattern; this does not establish engine health." if prediction == "Nominal Operation"
                 else "Pattern matches a synthetic fault example; no real fault or imminent failure is established.")
        expected = assign_fault(features)
        st.caption(f"Synthetic generator rule for this input: {expected}")
        if prediction != expected:
            st.warning("SVM and generator rule disagree here. Learned boundaries are approximate, especially near thresholds and overlapping scenarios.")
        st.dataframe(pd.DataFrame({"Synthetic class": fusion_model.classes_,
                                   "Model probability estimate": probabilities}), hide_index=True, width="stretch")
        st.caption("Estimates apply to the artificial training distribution. They are not calibrated real-engine fault probabilities; SVC probability ranking can differ from its predicted class.")
    with st.expander("Exact model input and rule thresholds"):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.write("Knocking: vibration std > 3.5, audio frequency < 1000 Hz, intake pressure > 120000. "
                 "Valve wear: audio frequency > 4000 Hz, speed > 4500, intercooler temperature > 350. "
                 "Misfire: speed < 1200, vibration std > 4, airflow < 50. "
                 "Turbo stress: throttle > 80, wastegate < 20, intercooler pressure > 140000. "
                 "Rules use strict inequalities and this priority order if several match. Five of the 14 inputs are not used by these illustrative rules.")


def telemetry():
    st.subheader("Historical Telemetry Analysis")
    st.caption("Upload one time-ordered LiU-scale CSV. Time is in seconds; other values remain in original dataset units. No unit conversion or missing-value filling is applied.")
    with st.expander("Required CSV columns and scale", expanded=False):
        st.code(", ".join(REQUIRED), language="text")
        st.write("Keep the original numeric scale and exact column names. The three pressure/airflow targets must be positive. Optional ambient readings are shown when present. Engine speed is in dataset units; its physical unit is unverified.")
        st.download_button("Download CSV column template", ",".join(REQUIRED) + "\n",
                           file_name="telemetry_template.csv", mime="text/csv")
    upload = st.file_uploader("Telemetry CSV", type=["csv"], key="telemetry_upload")
    upload = example_input(upload, "telemetry", [
        ("Try normal telemetry", ROOT / "outputs/judging_examples/liu_normal_training.csv", "LiU normal recording, first 30 seconds. Training-data excerpt, not an independent evaluation."),
        ("Try pressure-fault telemetry", ROOT / "outputs/judging_examples/liu_pressure_fault_training.csv", "LiU intercooler-pressure scenario, 150-180 seconds. Training-data excerpt with approximate fault timing; not independent validation."),
        ("Try synthetic telemetry", ROOT / "demo_samples/synthetic_telemetry_demo.csv", "Generated processing example; not measured engine data."),
    ])
    demo = ROOT / "demo_samples/synthetic_telemetry_demo.csv"
    if demo.is_file():
        st.download_button("Download demo telemetry CSV", demo.read_bytes(),
                           file_name=demo.name, mime="text/csv")
        st.caption("Synthetic 60-second sample for testing uploads and charts. Values are illustrative, not verified LiU measurements; model alerts on this sample do not validate fault detection.")
    if upload is None:
        st.info("Upload a CSV recording to see its sensor traces, model estimates and alert intervals.")
        return
    if Path(upload.name).suffix.lower() != ".csv":
        st.error("Unsupported file. Upload a telemetry .csv recording.")
        return
    try:
        data = read_telemetry(upload.getvalue())
    except (ValueError, OverflowError) as exc:
        st.error(f"CSV incompatible: {exc}")
        return
    st.caption(f"Analysed file: **{upload.name}** · {len(data):,} readings · {data.time.iloc[-1] - data.time.iloc[0]:.2f} seconds")
    quality = recording_quality(data)
    with st.expander("Recording quality and input preview"):
        a, b, c = st.columns(3)
        a.metric("Validated readings", f"{quality['readings']:,}")
        interval = quality["median_interval_seconds"]
        b.metric("Median sample interval", f"{interval:.4f} s" if interval is not None else "—")
        c.metric("Time gaps > 0.1 s", quality["gaps_over_0_1_seconds"])
        if quality["gaps_over_0_1_seconds"]:
            st.warning("Time gaps interrupt model history and can make predictions unavailable.")
        if quality["constant_signals"]:
            st.info("Constant signals: " + ", ".join(quality["constant_signals"]) + ". Check that these are expected for this recording.")
        st.dataframe(data.head(100), hide_index=True, width="stretch")
        st.caption("Preview shows up to 100 readings. Analysis uses the complete recording.")
    if not MODEL.is_file():
        st.warning("Model unavailable. Place the separately supplied frozen artifact at `outputs/liu_residuals/residual_model.joblib` and restart the dashboard. Predictions and alerts cannot be generated without it.")
        sensor_trends(data, list(REQUIRED[1:]), title="Uploaded sensor readings")
        return
    try:
        expected_hash, provenance = model_identity()
        stat = MODEL.stat()
        rows = analyse_telemetry(upload.getvalue(), str(MODEL), expected_hash,
                                 (stat.st_mtime_ns, stat.st_size))
    except Exception as exc:
        st.error(f"Model could not analyse this recording: {exc}")
        return
    available = rows.prediction_available.fillna(False).astype(bool)
    st.info(provenance)
    flagged = available & rows.prediction.eq(1).fillna(False)
    intervals = alert_intervals(rows)
    synthetic = demo.is_file() and upload.getvalue() == demo.read_bytes()
    guidance = inspection_guidance(rows, synthetic=synthetic)
    with st.container(border=True):
        st.subheader("Does this engine need a check?")
        if flagged.any() and not synthetic:
            st.warning(guidance["title"])
        else:
            st.info(guidance["title"])
        st.write(guidance["message"])
        a, b = st.columns(2)
        percentage = guidance["alert_percentage"]
        a.metric("Analyzed readings with alerts", f"{percentage:.1f}%" if percentage is not None else "Unavailable")
        b.metric("Engine fault risk", "Not established")
        if percentage is not None:
            st.progress(percentage / 100)
        st.caption(f"{guidance['flagged_readings']:,} flagged / {guidance['available_readings']:,} analyzed readings. "
                   f"{guidance['unavailable_readings']:,} readings could not be assessed and are excluded from this percentage.")
        st.caption("This percentage measures how often alerts occurred, not the chance of a fault, damage severity, or safety to operate. A reliable fault-risk percentage requires independent validation and probability calibration.")
    cols = st.columns(4)
    cols[0].metric("Readings", f"{len(data):,}")
    cols[1].metric("Duration", f"{data.time.iloc[-1] - data.time.iloc[0]:.1f} s")
    cols[2].metric("Prediction available", f"{available.mean():.1%}")
    cols[3].metric("Flagged readings · available", f"{flagged.sum() / available.sum():.1%}" if available.any() else "—")
    state = "Anomaly detected" if flagged.any() else ("No anomaly detected" if available.any() else "Waiting for sufficient history")
    st.markdown(f"### {state}")
    if flagged.any():
        st.write("What this means: some sensor readings disagree unusually with the relationships learned during normal operation.")
    elif available.any():
        st.write("What this means: the model found no alert-level disagreement in the readings it could assess; faults can still be missed.")
    st.caption("Alerts indicate unusual disagreement between sensor readings and learned normal relationships. They do not confirm a component fault. No anomaly detected does not establish engine health.")
    score = figure("Anomaly score · fixed alert threshold > 1", "Normalized disagreement (not probability)", 340)
    line(score, rows.time, rows.anomaly_score, "Anomaly score", BLUE)
    score.add_hline(y=1, line_dash="dash", line_color=ORANGE, annotation_text="Alert threshold 1")
    if len(intervals) <= 200:
        for item in intervals.itertuples(index=False):
            score.add_vrect(x0=item.start_seconds, x1=item.end_seconds, fillcolor=RED,
                            opacity=.26, line_width=0)
    score.add_trace(go.Scatter(x=rows.loc[flagged, "time"], y=rows.loc[flagged, "anomaly_score"],
                               mode="markers", marker=dict(color=RED, size=3), name="Flagged readings"))
    chart(score)
    st.caption("Gaps indicate unavailable predictions. Red points mark flagged readings; interval shading is shown when there are at most 200 intervals. All intervals remain in the table and download.")
    start, end = float(data.time.iloc[0]), float(data.time.iloc[-1])
    selected = st.slider("Inspect a time interval (seconds)", min_value=start, max_value=end,
                         value=(start, end), key="telemetry_range_" + hashlib.sha256(upload.getvalue()).hexdigest()) if end > start else (start, end)
    view = rows.time.between(*selected)
    st.subheader("Measured signals and model estimates")
    for target, title in [("intake_manifold_pressure", "Intake manifold pressure"),
                          ("Intercooler_pressure", "Intercooler pressure"),
                          ("air_mass_flow", "Air mass flow")]:
        fig = figure(title, "Dataset units")
        line(fig, rows.loc[view, "time"], rows.loc[view, target + "_actual"], "Measured", BLUE)
        line(fig, rows.loc[view, "time"], rows.loc[view, target + "_predicted"], "Model estimate", ORANGE)
        chart(fig)
    st.caption("Model estimates describe learned sensor relationships; they are not ground truth. The full recording is scored before selecting a chart interval so history remains intact.")
    sensor_trends(data.loc[data.time.between(*selected)],
                  [x for x in ["engine_speed", "throttle_position", "wastegate_position",
                               "injected_fuel_mass", "intercooler_temperature", *OPTIONAL] if x in data.columns],
                  title="Supporting sensor trends")
    st.subheader("Alert intervals")
    st.caption("Consecutive flagged readings; intervals break at unflagged readings, unavailable predictions or time gaps. These are not confirmed fault events.")
    st.dataframe(intervals, hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.download_button("Download predictions CSV", rows.to_csv(index=False).encode(),
                      file_name=f"{Path(upload.name).stem}_predictions.csv", mime="text/csv")
    b.download_button("Download alert intervals CSV", intervals.to_csv(index=False).encode(),
                      file_name=f"{Path(upload.name).stem}_alert_intervals.csv", mime="text/csv")
    summary = analysis_summary(rows, upload.name, expected_hash)
    summary["model_provenance"] = provenance
    summary["inspection_guidance"] = guidance
    summary["input_quality"] = quality
    st.download_button("Download analysis summary JSON", json.dumps(summary, indent=2, allow_nan=False),
                       file_name=f"{Path(upload.name).stem}_summary.json", mime="application/json")


def sensor_trends(data, choices, title):
    st.subheader(title)
    chosen = st.multiselect("Signals (each shown on its own scale)", choices,
                            default=choices[:3], key="sensor_selection")
    for column in chosen:
        values = data[column]
        fig = figure(column.replace("_", " ").capitalize(), "Dataset units", 220)
        line(fig, data.time, values, "Measured", BLUE)
        chart(fig)


def audio():
    st.subheader("Engine Audio Signatures")
    st.caption("WAV analysis in complete one-second windows. Audio and telemetry recordings are separate and are not synchronized.")
    upload = st.file_uploader("Engine audio WAV", type=["wav"], key="audio_upload")
    upload = example_input(upload, "audio", [
        ("Try audio example", ROOT / "demo_samples/synthetic_audio_comparison.wav", "Generated 12-second audio: six seconds within idle feature ranges, followed by six seconds with changed sound. Not an engine-fault recording."),
    ])
    demo_audio = ROOT / "demo_samples/synthetic_audio_comparison.wav"
    if demo_audio.is_file():
        st.download_button("Download demo audio WAV", demo_audio.read_bytes(),
                           file_name=demo_audio.name, mime="audio/wav")
        st.caption("Synthetic 12-second processing demo: 0–6 seconds within the saved idle feature ranges; "
                   "6–12 seconds louder and higher-frequency. Not an engine recording or evidence of fault-detection accuracy.")
    if upload is None:
        st.info("Upload a WAV recording to see its waveform, FFT and one-second features.")
        return
    if Path(upload.name).suffix.lower() != ".wav":
        st.error("Unsupported file. Upload a PCM or supported floating-point .wav recording.")
        return
    content = upload.getvalue()
    try:
        samples, rate, channels = load_audio(io.BytesIO(content))
        if rate < 4 or len(samples) < rate:
            st.warning(f"Recording is too short: {len(samples)/rate:.3f} seconds. At least one complete one-second window is required.")
            return
        windows = list(iter_windows(samples, rate, window_seconds=1.0))
        records = [{"start_seconds": start, "end_seconds": end, "sample_rate": rate,
                    **extract_features(segment, rate)} for start, end, segment in windows]
        frame = pd.DataFrame(records)
    except (ValueError, OSError, EOFError) as exc:
        st.error(f"WAV incompatible: {exc}")
        return
    st.caption(f"Analysed file: **{upload.name}**")
    st.audio(content, format="audio/wav")
    cols = st.columns(4)
    cols[0].metric("Duration", f"{len(samples)/rate:.2f} s")
    cols[1].metric("Sample rate", f"{rate:,} Hz")
    cols[2].metric("Original channels", str(channels))
    cols[3].metric("Complete windows", str(len(windows)))
    trailing = len(samples) - len(windows) * rate
    if trailing:
        st.caption(f"Incomplete trailing window skipped: {trailing/rate:.3f} seconds.")
    idle_comparison(frame, upload.name)
    waveform = figure("Complete recording waveform", "Amplitude (loader scale)", 310)
    stride = max(1, len(samples) // 15000)
    line(waveform, np.arange(0, len(samples), stride) / rate, samples[::stride], "Mono waveform", BLUE)
    chart(waveform)
    st.caption("Waveform is subsampled for display; short peaks may not be visible. Integer PCM is scaled by the loader; float WAV amplitudes are preserved.")
    index = st.selectbox("Complete one-second window", range(len(windows)), key="audio_window_" + hashlib.sha256(content).hexdigest(),
                         format_func=lambda n: f"{windows[n][0]:.1f}–{windows[n][1]:.1f} s")
    selected = frame.iloc[index]
    cards = st.columns(4)
    for col, key, label, fmt in zip(cards, FEATURES,
                                     ["RMS", "Peak amplitude", "Crest factor", "Dominant frequency"],
                                     [".4f", ".4f", ".2f", ".1f"]):
        col.metric(label, f"{selected[key]:{fmt}}" + (" Hz" if key == "dominant_frequency_hz" else ""))
    freq, magnitude = frequency_spectrum(windows[index][2], rate)
    spectrum = figure("FFT spectrum · selected window", "FFT magnitude", 310)
    line(spectrum, freq, magnitude, "Magnitude", ORANGE)
    spectrum.update_xaxes(title="Frequency (Hz)", range=[0, rate/2])
    chart(spectrum)
    st.subheader("Feature trends · one-second windows")
    for key, label in zip(FEATURES, ["RMS", "Peak amplitude", "Crest factor", "Dominant frequency (Hz)"]):
        fig = figure(label, label, 210)
        line(fig, frame.start_seconds, frame[key], label, BLUE)
        chart(fig)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("Download one-second feature CSV", frame.to_csv(index=False).encode(),
                       file_name=f"{Path(upload.name).stem}_audio_features_1s.csv", mime="text/csv")
    spectrogram(samples, rate)
    reference(frame)
    st.caption("Audio features and FFT are descriptive signatures. No audio fault classifier or diagnostic threshold has been trained.")


def spectrogram(samples, rate):
    with st.expander("STFT spectrogram · descriptive frequency view"):
        if not st.checkbox("Generate spectrogram", value=False):
            return
        display_samples = samples[:min(len(samples), 60 * rate)]
        if len(display_samples) < len(samples):
            st.caption("Spectrogram displays the first 60 seconds; feature extraction covers the complete recording.")
        nperseg = min(1024, len(display_samples))
        hop = min(nperseg, max(nperseg // 4, int(np.ceil(len(display_samples) / 1000))))
        frequencies, times, z = stft(display_samples, fs=rate, nperseg=nperseg,
                                      noverlap=nperseg - hop, boundary=None)
        power = 20 * np.log10(np.maximum(abs(z), 1e-10))
        time_stride = max(1, int(np.ceil(len(times) / 1000)))
        frequency_stride = max(1, int(np.ceil(len(frequencies) / 512)))
        fig = go.Figure(go.Heatmap(x=times[::time_stride], y=frequencies[::frequency_stride],
                                    z=power[::frequency_stride, ::time_stride], colorscale="Viridis",
                                    colorbar=dict(title="dB re 1 amplitude unit")))
        st.caption("STFT magnitude on the loader's amplitude scale, not calibrated sound pressure. Display grid may be subsampled.")
        fig.update_layout(title="Time–frequency signature", height=410, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                          xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
        chart(fig)


def reference(frame):
    with st.expander("Normal-idle reference · historical distribution"):
        try:
            baseline = pd.read_csv(BASELINE)
            if not set(FEATURES).issubset(baseline.columns):
                raise ValueError("Reference columns are incomplete.")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.warning(f"Normal-idle reference unavailable: {exc}")
            return
        st.caption(f"{len(baseline)} saved normal-idle rows. Rows are not assumed to represent independent vehicles or sessions. Differences are descriptive, not fault decisions.")
        feature = st.selectbox("Reference feature", FEATURES, format_func=lambda x: x.replace("_", " ").title())
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=baseline[feature], name="Normal-idle reference", marker_color=GREY))
        fig.add_vline(x=float(frame[feature].median()), line_color=BLUE, line_width=3,
                      annotation_text="Uploaded median")
        fig.update_layout(title=feature.replace("_", " ").title(), xaxis_title=feature,
                          yaxis_title="Reference rows", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=TEXT), height=310)
        chart(fig)


def idle_comparison(frame, filename):
    st.subheader("How does this sound compare with idle?")
    try:
        content = BASELINE.read_bytes()
        table, windows, summary = compare_idle(frame, pd.read_csv(io.BytesIO(content)))
    except (OSError, ValueError, TypeError) as exc:
        st.warning(f"Idle comparison unavailable: {exc}")
        return
    if summary["different_windows"]:
        st.warning("Differences from the idle baseline found")
    else:
        st.info("Selected audio features are within the idle reference range")
    st.write("This comparison is a diagnostic observation. Acceleration, microphone position, recording gain, "
             "background noise and engine differences can change the sound without indicating a defect. "
             "For a meaningful idle comparison, record at idle using comparable equipment and settings.")
    a, b, c = st.columns(3)
    a.metric("Windows differing from idle", f"{summary['different_windows_percent']:.1f}%")
    b.metric("Compared windows", summary["uploaded_windows"])
    c.metric("Idle reference windows", summary["baseline_windows"])
    st.write(f"What this means: {summary['different_windows']} of {summary['uploaded_windows']} one-second pieces differ from the idle reference in at least one sound feature. This does not identify the cause.")
    st.caption(f"A window differs if any feature is outside its idle 5th–95th percentile range. "
               f"Even the saved idle reference has {summary['idle_reference_outside_percent']:.1f}% of windows outside at least one range. "
               "These are descriptive bands, not validated diagnostic thresholds or a fault-risk percentage.")
    st.dataframe(table, hide_index=True, width="stretch")
    feature = st.selectbox("Compare feature against idle range", FEATURES,
                           format_func=lambda x: x.replace("_", " ").title(), key="idle_comparison_feature")
    selected = table.set_index("feature").loc[feature]
    fig = figure("Uploaded audio against idle reference", feature.replace("_", " "))
    fig.add_hrect(y0=selected.idle_low, y1=selected.idle_high, fillcolor=BLUE,
                  opacity=.15, line_width=0, annotation_text="Idle central 90% range")
    fig.add_hline(y=selected.idle_median, line_dash="dash", line_color=GREY)
    line(fig, frame.start_seconds, frame[feature], "Uploaded feature", ORANGE)
    chart(fig)
    with st.expander("Window-by-window differences"):
        st.dataframe(windows, hide_index=True, width="stretch")
    summary["baseline_sha256"] = hashlib.sha256(content).hexdigest()
    summary["recording"] = filename
    summary["feature_comparison"] = table.to_dict(orient="records")
    a, b = st.columns(2)
    a.download_button("Download idle comparison CSV", windows.to_csv(index=False),
                      file_name=f"{Path(filename).stem}_idle_comparison.csv", mime="text/csv")
    b.download_button("Download idle summary JSON", json.dumps(summary, indent=2, allow_nan=False),
                      file_name=f"{Path(filename).stem}_idle_summary.json", mime="application/json")


def evaluation():
    st.subheader("Model evaluation · historical LiU benchmark")
    try:
        with (ROOT / "reports/official_test.json").open() as handle:
            official = json.load(handle)
        with (ROOT / "reports/development_validation.json").open() as handle:
            development = json.load(handle)
    except (OSError, ValueError) as exc:
        st.warning(f"Saved evaluation reports unavailable: {exc}")
        return
    pooled = official["pooled"]
    intercooler = [r for r in official["recordings"] if r["recording_id"].startswith("wltp_f_pic_")]
    tp = sum(r["tp"] for r in intercooler)
    fn = sum(r["fn"] for r in intercooler)
    normal = next((r for r in official["recordings"] if r["recording_id"] == "wltp_NF_2"), None)
    values = [("Faulty samples detected", pooled["tp"] / (pooled["tp"] + pooled["fn"])),
              ("Normal samples falsely flagged", pooled["fp"] / (pooled["fp"] + pooled["tn"])),
              ("Full normal recording false alarms", normal["fp"] / (normal["fp"] + normal["tn"]) if normal else float("nan")),
              ("Intercooler-pressure scenario detection", tp / (tp + fn) if tp + fn else float("nan"))]
    for col, (label, value) in zip(st.columns(4), values):
        col.metric(label, f"{value:.1%}")
    st.caption("Official frozen-model test · sample-level results. The intercooler figure applies only to that scenario; it is not overall accuracy or component identification. The weak overall result includes many missed faulty samples.")
    st.write(f"Development validation: {development['pooled']['fault_recall']:.1%} faulty-sample detection and {development['pooled']['false_positive_rate']:.1%} normal-sample false flags across held-out recording folds.")
    st.info("Evaluation uses the LiU benchmark, not a fleet of independent vehicles. Fault onset is approximate; fault-recording evaluation excludes 110–130 seconds. Official test data has already been inspected, so further tuning requires fresh independent evaluation.")
    st.caption("Current telemetry model: gradient-boosted regression trees. CNN, SVM and ANN/Softmax approaches remain a brief-alignment gap. Vibration supports descriptive reference comparison; live ECU/OBD/CAN connection, validated wear prediction and audio fault classification are not implemented.")
    st.subheader("Results by test recording")
    results = pd.DataFrame(official["recordings"])
    st.dataframe(results[["recording_id", "fault_recall", "false_positive_rate", "tp", "fn", "fp", "tn"]],
                 hide_index=True, width="stretch",
                 column_config={
                     "fault_recall": st.column_config.NumberColumn("Fault detection fraction", format="%.3f"),
                     "false_positive_rate": st.column_config.NumberColumn("False alarm fraction", format="%.3f"),
                 })
    st.caption("Fractions range from 0 to 1. Blank detection values mean no faulty samples were evaluated in that recording.")
    st.download_button("Download benchmark results CSV", results.to_csv(index=False),
                       file_name="historical_benchmark.csv", mime="text/csv")


# Main Application Structure
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### SYSTEM DIAGNOSTICS")
    st.markdown(f"<div class='status-indicator' style='color:{SUCCESS};'>🟢 LOCAL DATA ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status-indicator' style='color:{SUCCESS};'>🟢 DSP AUDIO PROCESSING: ACTIVE</div>", unsafe_allow_html=True)

    if (ROOT / "outputs/fusion_model/multimodal_svm.pkl").is_file():
        st.markdown(f"<div class='status-indicator' style='color:{SUCCESS};'>🟢 SYNTHETIC SVM ARTIFACT: PRESENT</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='status-indicator' style='color:{RED};'>🔴 SVM FUSION MATRIX: OFFLINE</div>", unsafe_allow_html=True)
        st.caption("Run train_fusion_model.py to initialize.")

st.title("Multivariable Engine Analysis Command Center")
st.markdown("Centralized visualization of time-domain signatures, ECU telemetry, and high-frequency vibration arrays.", unsafe_allow_html=True)

with st.expander("Judging walkthrough and data sources"):
    st.write("1. ECU telemetry: try the normal and pressure-fault examples. These are LiU training excerpts, not independent performance tests.")
    st.write("2. Acoustic spectra: try the generated audio example. Its second half differs from the saved idle feature ranges.")
    st.write("3. Vibration array: try matching and changed examples against the synthetic baseline.")
    st.write("4. Multimodal simulator: choose a scenario, then adjust controls. The SVM learns synthetic labeling rules; it does not fuse the uploaded recordings.")
    st.caption("Alerts and differences support investigation. No module establishes engine safety, remaining life, or validated specific mechanical diagnoses.")

# Keep measured-data analysis first; show the synthetic fusion extension separately.
tab2, tab3, tab4, tab1, tab5 = st.tabs([
    "ECU TELEMETRY", "ACOUSTIC SPECTRA", "VIBRATION ARRAY",
    "MULTIMODAL SIMULATOR", "HISTORICAL BENCHMARKS"
])
with tab1: multimodal_fusion()
with tab2: telemetry()
with tab3: audio()
with tab4: vibration_panel()
with tab5: evaluation()
