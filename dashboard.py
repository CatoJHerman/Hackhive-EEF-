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
from predict import predict_recording

ROOT = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("ENGINE_MODEL_PATH", str(ROOT / "outputs/liu_residuals/residual_model.joblib")))
BASELINE = ROOT / "baselines/normal_idle_features_1s.csv"
BLUE, ORANGE, RED, GREY = "#2794ec", "#f5a23c", "#e75b68", "#8692a5"
BG, PANEL, TEXT = "#0d1423", "#172237", "#eaf2ff"

st.set_page_config(page_title="Engine signal studio", page_icon="◉", layout="wide")
st.markdown(f"""<style>
.stApp {{background:{BG}; color:{TEXT}}}
.block-container {{max-width:1350px; padding-top:2rem; padding-bottom:3rem}}
h1,h2,h3 {{letter-spacing:-.035em}}
h1 {{font-size:2.65rem!important}}
[data-testid="stMetric"] {{background:{PANEL};border:1px solid #28354c;border-radius:18px;padding:16px 20px}}
[data-testid="stMetricLabel"] {{color:#aab8cc}}
[data-testid="stMetricValue"] {{color:{TEXT}}}
[data-testid="stFileUploaderDropzone"] {{background:{PANEL};border:1px dashed #587092;border-radius:16px}}
[data-testid="stAlert"], [data-testid="stExpander"] {{border-radius:14px}}
button[data-baseweb="tab"] {{font-weight:700}}
button[data-baseweb="tab"][aria-selected="true"] {{color:#58b8fa}}
.eyebrow {{color:#68cce5;letter-spacing:.16em;font-size:.76rem;font-weight:700}}
.intro {{color:#abbcd0;max-width:850px;line-height:1.6}}
</style>""", unsafe_allow_html=True)


def figure(title, y_label, height=300):
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=17)), height=height,
                      paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
                      margin=dict(l=42, r=24, t=54, b=42), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(title="Time (seconds)", gridcolor="#304059", zeroline=False),
                      yaxis=dict(title=y_label, gridcolor="#304059", zeroline=False))
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


def telemetry():
    st.subheader("Engine telemetry")
    st.caption("Upload one time-ordered LiU-scale CSV. Time is in seconds; other values remain in original dataset units. No unit conversion or missing-value filling is applied.")
    with st.expander("Required CSV columns and scale", expanded=False):
        st.code(", ".join(REQUIRED), language="text")
        st.write("Keep the original numeric scale and exact column names. The three pressure/airflow targets must be positive. Optional ambient readings are shown when present. Engine speed is in dataset units; its physical unit is unverified.")
        st.download_button("Download CSV column template", ",".join(REQUIRED) + "\n",
                           file_name="telemetry_template.csv", mime="text/csv")
    upload = st.file_uploader("Telemetry CSV", type=["csv"], key="telemetry_upload")
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
    st.caption("Alerts indicate unusual disagreement between sensor readings and learned normal relationships. They do not confirm a component fault. No anomaly detected does not establish engine health.")
    score = figure("Anomaly score · fixed alert threshold > 1", "Normalized disagreement (not probability)", 340)
    line(score, rows.time, rows.anomaly_score, "Anomaly score", BLUE)
    score.add_hline(y=1, line_dash="dash", line_color=ORANGE, annotation_text="Alert threshold 1")
    if len(intervals) <= 200:
        for item in intervals.itertuples(index=False):
            score.add_vrect(x0=item.start_seconds, x1=item.end_seconds, fillcolor=RED,
                            opacity=.26, line_width=0)
    # Mark isolated flags too; thousands of Plotly rectangles can freeze the UI.
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
    st.subheader("Engine audio")
    st.caption("WAV analysis in complete one-second windows. Audio and telemetry recordings are separate and are not synchronized.")
    upload = st.file_uploader("Engine audio WAV", type=["wav"], key="audio_upload")
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
        # SciPy accepts a file-like object; avoids Windows temporary-file locks.
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
        # Bound memory for long uploads without changing the sample rate or extraction.
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
        fig.update_layout(title="Time–frequency signature", height=410, paper_bgcolor=PANEL,
                          plot_bgcolor=PANEL, font=dict(color=TEXT),
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
                          yaxis_title="Reference rows", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
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
    st.caption("Reference: saved normal_idle_features_1s.csv, 44.1 kHz one-second features. "
               "Raw idle WAVs are unavailable in this checkout, so extraction provenance and independent-session coverage have not been reverified.")


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
    st.caption("Current telemetry model: gradient-boosted regression trees. CNN, SVM and ANN/Softmax approaches remain a brief-alignment gap. No accelerometer/vibration pipeline, live ECU/OBD/CAN connection, validated wear prediction or audio fault classifier is present.")
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


def system_status():
    with st.sidebar:
        st.markdown("### Engine signal studio")
        st.caption("HACKATHON WORKSPACE")
        st.markdown("#### Model connection")
        if MODEL.is_file():
            try:
                expected, provenance = model_identity()
                stat = MODEL.stat()
                load_model(str(MODEL), expected, (stat.st_mtime_ns, stat.st_size))
                st.success("Model verified and loaded")
                st.caption(provenance)
            except Exception as exc:
                st.error(f"Model unavailable: {exc}")
        else:
            st.warning("Frozen model not connected")
            st.caption("Place the team's residual_model.joblib at the path below, then refresh. You can also set ENGINE_MODEL_PATH before launching.")
        st.code(str(MODEL), language="text")
        st.markdown("#### Analysis workflow")
        st.write("1. Upload a telemetry CSV or audio WAV.\n2. Inspect signals and available alerts.\n3. Export results for your presentation.")
        st.caption("Telemetry requires the frozen model for predictions. Audio feature analysis works independently.")


st.markdown('<div class="eyebrow">ENGINE SIGNAL STUDIO / EXPERIMENTAL</div>', unsafe_allow_html=True)
system_status()
st.title("See what the engine signals show.")
st.markdown('<div class="intro">An experimental engine-signal analysis and consistency-monitoring dashboard. Inspect uploaded telemetry and audio as separate recordings.</div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Engine telemetry", "Engine audio", "Model evaluation"])
with t1:
    telemetry()
with t2:
    audio()
with t3:
    evaluation()
