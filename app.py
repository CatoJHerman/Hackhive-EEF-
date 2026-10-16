"""Streamlit dashboard for the merged automobile-engine prototype.

Run with:  python -m streamlit run app.py

Two separate tabs, no combined score:
  * Engine sensor check - per-group OBD view. Expected MAP, the difference score and the
    "possible problem" headline appear only when map_model.score_vehicle is connected.
  * Sound analysis - normal-idle reference plus a WAV upload measured in 1-second windows.

Prototype view only: no diagnosis, component-wear or remaining-life claims.
"""

from __future__ import annotations

import importlib
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile

try:  # existing repo modules; import errors are reported inside the Sound tab
    from audio_processing import iter_windows, load_audio
    from features import extract_features, frequency_spectrum

    AUDIO_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    AUDIO_IMPORT_ERROR = exc

HERE = Path(__file__).resolve().parent
OBD_CSV = HERE / "diesel_obd_clean.csv"
BASELINE_CSV = HERE / "baselines" / "normal_idle_features_1s.csv"

# Plain-language names for the four measured audio features.
AUDIO_FEATURES = {
    "rms": "Average loudness (RMS)",
    "peak_amplitude": "Highest amplitude",
    "crest_factor": "Peak compared with average (crest factor)",
    "dominant_frequency_hz": "Strongest frequency (Hz)",
}
WINDOW_SECONDS = 1.0
IDLE_ONLY_GROUPS = {6}  # per project notes: group 6 was recorded at idle only
TREND_SENSORS = ["LOAD_PCT", "ECT", "RPM", "VSS", "IAT", "MAF", "FRP", "VPWR", "AAT"]
DEFAULT_TRENDS = ["RPM", "LOAD_PCT", "VSS", "ECT"]
REQUIRED_OBD_COLUMNS = {
    "MAP", "BARO", "RPM", "LOAD_PCT", "vehicle_group", "fault_family",
    "label_uncertain", *TREND_SENSORS,
}
TRIGGER = 2.0  # display only; the backend applies the fixed cutoff and returns the flag

# Palette: warm white ground, charcoal ink, one dark green reserved for focal data.
CREAM, INK, GREEN = "#F9F6F0", "#1A1A1A", "#1F4D3A"
GREY, RULE, TINT = "#8C8A84", "#D9D4C7", "#EFEBE1"

CAVEATS = [
    "Only two vehicle groups (1 and 7) are confirmed healthy.",
    "Group 3's healthy label is unconfirmed.",
    "Group 6 is idle-only, so its score is not meaningful.",
    "Audio and OBD recordings are not linked to the same vehicle.",
    "Prototype screening view: no confirmed diagnosis, component wear or remaining-life claims.",
]

CSS = f"""
<style>
.stApp {{ background:{CREAM}; }}
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stCaptionContainer"], .stApp li {{
  color:{INK};
  font-family:"Neue Haas Grotesk Display Pro","Neue Haas Grotesk Text Pro","Helvetica Neue",
              Helvetica,Arial,"Liberation Sans",sans-serif;
}}
.block-container {{ padding-top:1.6rem; padding-bottom:2rem; max-width:1180px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
button[data-baseweb="tab"] {{ font-size:1.05rem; font-weight:600; padding-left:0; padding-right:1.6rem; }}
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] p {{ color:{GREEN}; }}
div[data-baseweb="tab-highlight"] {{ background:{GREEN}; height:3px; }}
div[data-baseweb="tab-border"] {{ background:{RULE}; }}
div[data-baseweb="select"] > div {{ border-radius:0; background:{CREAM}; border:1px solid {INK}; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:0; background:{TINT}; border:1px dashed {INK}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:0 !important; box-shadow:none !important; }}
[data-testid="stExpander"] details {{ border-radius:0; border:1px solid {RULE}; }}
.pagehead {{ font-size:0.95rem; color:{GREY}; margin-bottom:0.4rem; letter-spacing:0.01em; }}
.status {{ border-top:3px solid {INK}; padding:0.9rem 0 1.3rem; }}
.status.flag {{ border-top-color:{GREEN}; }}
.status-title {{ font-size:2.3rem; font-weight:700; line-height:1.1; letter-spacing:-0.02em; }}
.status-body {{ font-size:1.08rem; line-height:1.45; max-width:44rem; margin-top:0.55rem; }}
.card-label {{ font-size:0.88rem; color:{GREY}; margin-bottom:0.35rem; }}
.card-big {{ font-size:3.6rem; font-weight:700; line-height:1; letter-spacing:-0.03em; color:{GREEN}; }}
.card-big.muted {{ color:{GREY}; }}
.card-trigger {{ font-size:1.05rem; margin-top:0.55rem; }}
.card-note {{ font-size:0.86rem; color:{GREY}; margin-top:0.6rem; line-height:1.35; }}
.sec-title {{ font-size:1.15rem; font-weight:700; letter-spacing:-0.01em; margin:1.4rem 0 0.2rem; }}
.meta {{ font-size:0.9rem; color:{GREY}; }}
.foot {{ border-top:1px solid {RULE}; margin-top:2rem; padding-top:0.8rem; font-size:0.85rem; color:{GREY}; line-height:1.5; }}
</style>
"""

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


# --------------------------------------------------------------------------------------
# MAP backend adapter: the ONLY place that touches the other teammate's code.
# To connect: put map_model.py next to app.py and the trained model at MAP_MODEL_PATH,
# and adjust the column / key names below if the backend uses different ones.
# --------------------------------------------------------------------------------------
MAP_MODEL_PATH = HERE / "models" / "map_consistency.joblib"
PRED_COL = "predicted_map"      # rows column: expected (predicted) MAP
GAP_COL = "abs_gap"             # rows column: |actual MAP - predicted MAP|
BELOW_COL = "map_lt_baro"       # rows column: MAP < BARO (bool)
SCORE_KEYS = ("normalized_score", "anomaly_ratio")  # summary: first key present is used
FLAG_KEY = "possible_anomaly"   # summary key: flag at the backend's fixed 2.0 cutoff


@st.cache_data(show_spinner="Scoring group...")
def _call_backend(vehicle_df: pd.DataFrame, model_path: Path, backend_stamp: tuple):
    # backend_stamp (file mtimes) is part of the cache key so edits to map_model.py or a
    # retrained model file never serve stale scores.
    from map_model import score_vehicle  # imported lazily so the dashboard runs without it

    rows, summary = score_vehicle(vehicle_df, model_path)
    return pd.DataFrame(rows).reset_index(drop=True), dict(summary)


def score_group(vehicle_df: pd.DataFrame) -> dict:
    """Return {status, message, rows, score, flag}; status is ok | not_connected | error.

    Never invents values: anything other than status "ok" carries no scores.
    """
    out = {"status": "not_connected", "message": "MAP model not connected",
           "rows": None, "score": None, "flag": None}
    try:
        module = importlib.import_module("map_model")
    except ModuleNotFoundError as exc:
        if exc.name == "map_model":
            out["message"] = "MAP model not connected (map_model.py not found)"
            return out
        return {**out, "status": "error", "message": f"map_model failed to import: {exc}"}
    except Exception as exc:  # backend broken at import time
        return {**out, "status": "error", "message": f"map_model failed to import: {exc}"}
    if not hasattr(module, "score_vehicle"):
        out["message"] = "MAP model not connected (score_vehicle not defined)"
        return out
    if not Path(MAP_MODEL_PATH).is_file():
        out["message"] = f"MAP model not connected (model file not found: {Path(MAP_MODEL_PATH).name})"
        return out
    try:
        stamp = (Path(module.__file__).stat().st_mtime_ns, Path(MAP_MODEL_PATH).stat().st_mtime_ns)
        rows, summary = _call_backend(vehicle_df, Path(MAP_MODEL_PATH), stamp)
    except Exception as exc:
        return {**out, "status": "error", "message": f"MAP model error: {type(exc).__name__}: {exc}"}

    missing = [c for c in (PRED_COL, GAP_COL, BELOW_COL) if c not in rows.columns]
    score = next((summary[k] for k in SCORE_KEYS if k in summary), None)
    flag = summary.get(FLAG_KEY)
    problems = []
    if missing:
        problems.append(f"rows missing {missing}")
    if len(rows) != len(vehicle_df):
        problems.append(f"expected {len(vehicle_df)} rows, got {len(rows)}")
    if not isinstance(score, (int, float, np.number)) or not math.isfinite(float(score)):
        problems.append(f"summary score ({' or '.join(SCORE_KEYS)}) missing or not finite")
    if flag is None:
        problems.append(f"summary['{FLAG_KEY}'] missing")
    if problems:
        return {**out, "status": "error", "message": "MAP model output not usable: " + "; ".join(problems)}
    return {"status": "ok", "message": "", "rows": rows, "score": float(score), "flag": bool(flag)}


# --------------------------------------------------------------------------------------
# Data loading and group labelling
# --------------------------------------------------------------------------------------
@st.cache_data
def load_obd() -> pd.DataFrame:
    df = pd.read_csv(OBD_CSV, encoding="utf-8-sig")
    missing = REQUIRED_OBD_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{OBD_CSV.name} is missing columns: {sorted(missing)}")
    return df


@st.cache_data
def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
    missing = set(AUDIO_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"{BASELINE_CSV.name} is missing columns: {sorted(missing)}")
    return df


def group_profiles(df: pd.DataFrame) -> dict:
    profiles = {}
    for g, x in df.groupby("vehicle_group"):
        healthy = bool((x["fault_family"] == "healthy").all())
        uncertain = bool((x["label_uncertain"] == 1).any())
        profiles[int(g)] = {
            "rows": len(x),
            "family": ", ".join(sorted(x["fault_family"].unique())),
            "confirmed_healthy": healthy and not uncertain,
            "healthy_unconfirmed": healthy and uncertain,
            "idle_only": int(g) in IDLE_ONLY_GROUPS,
            "rpm_range": (int(x["RPM"].min()), int(x["RPM"].max())),
        }
    return profiles


def group_label(g: int, p: dict) -> str:
    if p["confirmed_healthy"]:
        tag = "confirmed healthy"
    elif p["healthy_unconfirmed"]:
        tag = "healthy label unconfirmed"
    else:
        tag = f"recorded code: {p['family']}"
    if p["idle_only"]:
        tag += ", idle-only"
    return f"Group {g} ({tag})"


def group_note(p: dict) -> str:
    note = f"Recorded code (dataset information, not a model prediction): {p['family']}."
    if p["healthy_unconfirmed"]:
        note += " The healthy label is unconfirmed (label_uncertain = 1)."
    elif p["confirmed_healthy"]:
        note += " One of the two confirmed healthy groups."
    if p["idle_only"]:
        lo, hi = p["rpm_range"]
        note += f" Idle-only recording (RPM {lo} to {hi})."
    return note


# --------------------------------------------------------------------------------------
# Flat chart helpers (matplotlib, so colours and corners stay exactly on-palette)
# --------------------------------------------------------------------------------------
def new_fig(width: float, height: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    fig.patch.set_facecolor(CREAM)
    for ax in np.atleast_1d(axes):
        style_ax(ax)
    return fig, axes


def style_ax(ax) -> None:
    ax.set_facecolor(CREAM)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8, length=3, width=0.6)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_color(INK)


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def donut_fig(pct: float):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.patch.set_facecolor(CREAM)
    ax.pie([pct, max(100 - pct, 0)], colors=[GREEN, RULE], startangle=90, counterclock=False,
           wedgeprops={"width": 0.2, "edgecolor": CREAM, "linewidth": 1.5})
    ax.text(0, 0, f"{pct:.1f}%", ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    ax.set_aspect("equal")
    return fig


def bars_fig(labels: list[str], values: list[float], colors: list[str], ylabel: str):
    fig, ax = new_fig(3.1, 2.5)
    xs = np.arange(len(values))
    ax.bar(xs, values, color=colors, width=0.5)
    for x, v in zip(xs, values):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)
    fig.tight_layout()
    return fig


def line_fig(x, series: list[tuple[str, np.ndarray, str, str]], height: float, ylabel: str = "",
             width: float = 11, legend: bool = True):
    """series = (label, values, colour, linestyle)."""
    fig, ax = new_fig(width, height)
    for label, values, colour, style in series:
        ax.plot(x, values, color=colour, linewidth=1.1, linestyle=style, label=label)
    ax.set_xlabel("Reading number")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK, ncols=len(series))
    fig.tight_layout()
    return fig


def card_html(label: str, big: str, trigger: str, note: str, muted: bool = False) -> str:
    cls = "card-big muted" if muted else "card-big"
    return (f'<div class="card-label">{label}</div><div class="{cls}">{big}</div>'
            f'<div class="card-trigger">{trigger}</div><div class="card-note">{note}</div>')


# --------------------------------------------------------------------------------------
# Tab 1: Engine sensor check
# --------------------------------------------------------------------------------------
def headline(result: dict, prof: dict) -> tuple[str, str, bool]:
    """Plain-English status. Only claims a possible problem when the backend flags one."""
    if result["status"] == "not_connected":
        return ("MAP model not connected",
                "Recorded readings are shown below. The difference score and expected MAP "
                "appear once the model backend is connected.", False)
    if result["status"] == "error":
        return ("MAP model could not score this group", result["message"], False)
    if prof["idle_only"]:
        return ("Score not meaningful for this group",
                "This group was recorded at idle only, so its difference score should not be "
                "interpreted.", False)
    if result["flag"]:
        return ("Possible MAP reading problem",
                "The MAP sensor's readings are much farther from expected than those in the "
                "healthy reference recordings.", True)
    return ("No MAP anomaly flagged",
            "The MAP sensor's readings are not much farther from expected than those in the "
            "healthy reference recordings.", False)


def obd_tab(obd: pd.DataFrame) -> None:
    profiles = group_profiles(obd)
    groups = sorted(profiles)
    sel, _ = st.columns([2, 1])
    with sel:
        g = st.selectbox("Recorded group", groups, key="obd_group",
                         format_func=lambda k: group_label(k, profiles[k]))
    prof = profiles[g]
    vehicle = obd[obd["vehicle_group"] == g].reset_index(drop=True)
    n = len(vehicle)
    reading_no = np.arange(1, n + 1)
    result = score_group(vehicle)
    ok = result["status"] == "ok"
    meaningless = ok and prof["idle_only"]

    title, body, flagged = headline(result, prof)
    st.markdown(f'<div class="status{" flag" if flagged else ""}"><div class="status-title">{title}</div>'
                f'<div class="status-body">{body}</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        big = f"{result['score']:.2f}" if ok else "n/a"
        note = ("Times the healthy reference difference. Not a percentage or a fault probability."
                if ok else "Appears when the model backend is connected.")
        if meaningless:
            note = "Not meaningful: this group is idle-only. " + note
        st.markdown(card_html("MAP difference score", big, f"Trigger point {TRIGGER:.1f}", note,
                              muted=not ok or meaningless), unsafe_allow_html=True)
    if ok:
        below = int(result["rows"][BELOW_COL].astype(bool).sum())
        source = "model output"
    else:
        below = int((vehicle["MAP"] < vehicle["BARO"]).sum())
        source = "recorded columns"
    pct = 100 * below / n
    with c2, st.container(border=True):
        st.markdown('<div class="card-label">MAP below BARO</div>', unsafe_allow_html=True)
        show(donut_fig(pct))
        st.markdown(f'<div class="card-note">MAP was below BARO in {pct:.1f}% of this recording\'s '
                    f'rows ({source}). An observation, not a diagnosis by itself.</div>',
                    unsafe_allow_html=True)
    with c3, st.container(border=True):
        st.markdown('<div class="card-label">Typical MAP: recorded and expected</div>', unsafe_allow_html=True)
        labels, values, colors = ["Recorded"], [float(vehicle["MAP"].median())], [INK]
        if ok:
            labels.append("Expected")
            values.append(float(np.median(result["rows"][PRED_COL].to_numpy())))
            colors.append(GREEN)
        show(bars_fig(labels, values, colors, "MAP (as recorded)"))
        st.markdown('<div class="card-note">Median across the recording.'
                    + ("" if ok else " Expected MAP: model not connected.") + "</div>",
                    unsafe_allow_html=True)

    st.markdown(f'<div class="meta" style="margin-top:0.8rem">{group_note(prof)} '
                f'{n:,} readings.</div>', unsafe_allow_html=True)
    if result["status"] == "error":
        st.error(result["message"])

    with st.expander("See the readings"):
        st.markdown('<div class="meta">The OBD file has no timestamps, so the horizontal axis is '
                    'reading number within the group.</div>', unsafe_allow_html=True)
        series = [("MAP recorded", vehicle["MAP"].to_numpy(), INK, "-"),
                  ("BARO", vehicle["BARO"].to_numpy(), GREY, "--")]
        if ok:
            series.insert(1, ("MAP expected", result["rows"][PRED_COL].to_numpy(), GREEN, "-"))
        show(line_fig(reading_no, series, 2.6, "Pressure (as recorded)"))

        left, right = st.columns(2)
        with left:
            st.markdown("**RPM and engine load**")
            fig, ax = new_fig(5.4, 2.6)
            ax.scatter(vehicle["RPM"], vehicle["LOAD_PCT"], s=4, color=INK, alpha=0.35, linewidths=0)
            ax.set_xlabel("RPM")
            ax.set_ylabel("LOAD_PCT (%)")
            fig.tight_layout()
            show(fig)
        with right:
            if ok:
                st.markdown("**Gap between recorded and expected MAP**")
                show(line_fig(reading_no, [("Absolute gap", result["rows"][GAP_COL].to_numpy(), GREEN, "-")],
                              2.6, "Gap", width=5.4, legend=False))

        st.markdown("**Other sensors**")
        chosen = st.multiselect("Sensors", TREND_SENSORS, default=DEFAULT_TRENDS,
                                key="trend_sensors", label_visibility="collapsed")
        cols = st.columns(3)
        for i, sensor in enumerate(chosen):
            with cols[i % 3]:
                st.caption(sensor)
                show(line_fig(reading_no, [(sensor, vehicle[sensor].to_numpy(), INK, "-")], 1.7,
                              width=3.6, legend=False))
        if ok:
            st.markdown("**Scored rows**")
            st.dataframe(result["rows"], height=220)


# --------------------------------------------------------------------------------------
# Tab 2: Sound analysis
# --------------------------------------------------------------------------------------
def test_signal_wav() -> bytes:
    """In-memory software processing test (100 Hz, 400 Hz, silence); nothing is written."""
    rate = 8000
    t = np.arange(2 * rate) / rate
    x = np.concatenate([0.5 * np.sin(2 * np.pi * 100 * t), 0.5 * np.sin(2 * np.pi * 400 * t),
                        np.zeros(2 * rate)])
    buf = io.BytesIO()
    wavfile.write(buf, rate, np.round(x * 32767).astype(np.int16))
    return buf.getvalue()


@st.cache_data(show_spinner="Extracting 1-second windows...", max_entries=4)
def process_wav(data: bytes) -> dict:
    """Use the existing load/window/feature code; nothing is written to disk."""
    samples, sample_rate, channels = load_audio(io.BytesIO(data))
    rows, first = [], None
    for start, end, window in iter_windows(samples, sample_rate, WINDOW_SECONDS):
        rows.append({"start_s": start, "end_s": end, **extract_features(window, sample_rate)})
        if first is None:
            first = window.copy()
    duration = len(samples) / sample_rate
    return {"table": pd.DataFrame(rows), "sample_rate": sample_rate, "channels": channels,
            "duration": duration, "skipped": duration - rows[-1]["end_s"], "first_window": first}


def compare_fig(baseline: pd.DataFrame, table: pd.DataFrame):
    fig, axes = new_fig(11, 2.4, ncols=4)
    for ax, (col, title) in zip(axes, AUDIO_FEATURES.items()):
        ref, up = float(baseline[col].median()), float(table[col].median())
        ax.bar([0, 1], [ref, up], color=[RULE, GREEN], width=0.55)
        for x, v in zip([0, 1], [ref, up]):
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks([0, 1], ["Normal idle", "Uploaded"])
        ax.set_title(title.split(" (")[0] if "crest" not in title else "Crest factor", fontsize=9)
        ax.set_ylim(0, max(ref, up, 1e-9) * 1.2)
    fig.tight_layout()
    return fig


def distribution_fig(baseline: pd.DataFrame, table: pd.DataFrame | None):
    fig, axes = new_fig(11, 2.3, ncols=4)
    for ax, (col, title) in zip(axes, AUDIO_FEATURES.items()):
        ax.hist(baseline[col], bins=24, color=RULE, edgecolor=CREAM, linewidth=0.5)
        if table is not None:
            ax.plot(table[col], np.zeros(len(table)), "|", color=GREEN, markersize=16,
                    markeredgewidth=1.6, clip_on=False)
        ax.set_title(title.split(" (")[0] if "crest" not in title else "Crest factor", fontsize=9)
    axes[0].set_ylabel("Windows")
    fig.tight_layout()
    return fig


def signal_fig(window: np.ndarray, sample_rate: int):
    fig, axes = new_fig(11, 2.3, ncols=2)
    t = np.arange(len(window)) / sample_rate
    keep = t <= 0.05
    axes[0].plot(t[keep], window[keep], color=INK, linewidth=1)
    axes[0].set_xlabel("Seconds (first 0.05 s of the first window)")
    axes[0].set_ylabel("Amplitude")
    freqs, mags = frequency_spectrum(window, sample_rate)
    top = min(sample_rate / 2, 2000)
    axes[1].plot(freqs[freqs <= top], mags[freqs <= top], color=GREEN, linewidth=1)
    axes[1].set_xlabel("Frequency (Hz), first window")
    axes[1].set_ylabel("FFT magnitude")
    fig.tight_layout()
    return fig


def sound_tab(baseline: pd.DataFrame) -> None:
    if AUDIO_IMPORT_ERROR is not None:
        st.error(f"Could not import audio_processing / features: {AUDIO_IMPORT_ERROR}")
        return

    st.markdown(f'<div class="status"><div class="status-title">Normal idle reference</div>'
                f'<div class="status-body">{len(baseline)} one-second windows of normal engine idle. '
                f'Upload a WAV to see how its measurements compare. These are measurements, not an '
                f'audio fault diagnosis.</div></div>', unsafe_allow_html=True)

    up_col, tog_col = st.columns([2, 1], gap="medium")
    with up_col:
        upload = st.file_uploader("Upload a WAV recording", type=["wav"], key="wav_upload")
    with tog_col:
        use_test = st.toggle("Use the software processing test signal", key="use_test_signal",
                             help="100 Hz, 400 Hz, then silence, generated in memory.")
    data, name, is_test = None, None, False
    if upload is not None:
        data, name = upload.getvalue(), upload.name
    elif use_test:
        data, name, is_test = test_signal_wav(), "software processing test signal", True

    result = None
    if data is not None:
        try:
            result = process_wav(data)
        except (ValueError, OSError, EOFError) as exc:
            st.error(f"Could not process {name}: {exc}")
    if result is None:
        if data is None:
            st.info("Upload a WAV file, or switch on the test signal, to see measurements for each "
                    "1-second window.")
        st.markdown('<div class="sec-title">Normal-idle distributions</div>', unsafe_allow_html=True)
        show(distribution_fig(baseline, None))
        return

    table = result["table"]
    if is_test:
        st.warning("Software processing test signal, not an engine recording. It only shows that "
                   "feature extraction responds to changes in the sound.")
    st.markdown(f'<div class="sec-title">{name}</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{result['duration']:.2f} s")
    m2.metric("Full 1-second windows processed", len(table))
    m3.metric("Sample rate", f"{result['sample_rate']:,} Hz")
    if result["skipped"] > 1e-9:
        st.caption(f"The incomplete final {result['skipped']:.2f} s was skipped. Audio is mono-mixed and "
                   "not level-normalized, so loudness depends on recording gain.")

    st.markdown('<div class="sec-title">Measurements: uploaded vs normal idle (median of windows)</div>',
                unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (feat, title) in zip(cols, AUDIO_FEATURES.items()):
        with col, st.container(border=True):
            val, ref = float(table[feat].median()), float(baseline[feat].median())
            st.markdown(card_html(title, f"{val:.3g}", f"Normal idle {ref:.3g}", ""),
                        unsafe_allow_html=True)
    show(compare_fig(baseline, table))

    with st.expander("See the distributions"):
        st.markdown('<div class="meta">Grey bars: normal-idle reference. Green ticks: each uploaded '
                    'window.</div>', unsafe_allow_html=True)
        show(distribution_fig(baseline, table))
    with st.expander("See the signal"):
        show(signal_fig(result["first_window"], result["sample_rate"]))
    with st.expander("Window-by-window table"):
        shown = table.rename(columns={"start_s": "Start (s)", "end_s": "End (s)", **AUDIO_FEATURES})
        st.dataframe(shown, hide_index=True, height=min(38 + 35 * len(shown), 280))


# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Engine prototype dashboard", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="pagehead">Engine prototype dashboard. OBD and audio come from separate '
                'recordings and are not combined.</div>', unsafe_allow_html=True)

    try:
        obd, baseline = load_obd(), load_baseline()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load input data: {exc}")
        st.stop()

    tab_obd, tab_sound = st.tabs(["Engine sensor check", "Sound analysis"])
    with tab_obd:
        obd_tab(obd)
    with tab_sound:
        sound_tab(baseline)

    st.markdown('<div class="foot">' + "<br>".join(CAVEATS) + "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
