# Engine signal dashboard

## Judging walkthrough

Use the one-click example buttons in each analysis tab. Uploaded files take
priority; remove an upload to enable examples, or use Clear example to reset.
Telemetry examples under `outputs/judging_examples/` are unmodified 30-second
excerpts from the prepared LiU training recordings: wltp_NF at 0-30 seconds and
wltp_f_pic_090 at 150-180 seconds. They are walkthrough inputs, not held-out
performance evidence, and remain local/ignored with the original data.
Audio and vibration examples are generated and visibly labeled synthetic.

The on-page walkthrough explains the demonstration sequence. Existing model
settings and thresholds were not changed for these presentation controls.

## Synthetic multimodal SVM simulator

The first tab simulates one 14-feature row using the schema in `fusion_schema.py`.
Training and sliders share feature names, order, numeric types and ranges.
`python train_fusion_model.py` generates 5,000 rule-labeled synthetic rows (1,000
per class), fits StandardScaler + SVC on 4,000 rows and evaluates on 1,000 held-out
rows. It saves `outputs/fusion_model/multimodal_svm.pkl` and
`outputs/fusion_model/synthetic_validation.json`. The previous model is backed up
as `multimodal_svm.before_fix.pkl` if present. Importing the script does not train.

Use the five scenario buttons, then adjust sliders. The displayed prediction is
always the SVM output; the generator rule is shown separately for comparison.
Rules use strict thresholds and priority order: knocking, valve wear, misfire,
turbo stress, otherwise nominal. Inputs near boundaries may disagree with rules.
Five inputs are not used by the illustrative labeling rules. Balanced synthetic
classes do not represent real-world fault prevalence. Neither the synthetic
validation score nor model probability estimates establish engine diagnostic
accuracy. The existing file-upload branches remain independent and are not fused.

Run `python -m unittest discover` after generating the model to check inference,
schema, reachable scenarios, dashboard buttons and the existing upload flows.

## Vibration integration

Upload one raw CSV with `time_seconds,sampling_rate_hz,accel_x,accel_y,accel_z`.
The saved database in `baselines/vibration_reference.csv` is loaded automatically.
It currently contains three synthetic reference recording vectors at 1,024 Hz,
created from `demo_samples/synthetic_vibration_reference.zip`; it is not measured
idle data. Replacement with verified comparable idle measurements is future work.

Each uploaded recording is summarized by median one-second features per axis.
The dashboard reports outside/within the observed reference range, or unable to
assess. This is a descriptive observation, not a calibrated threshold or diagnosis.
Recording medians can hide brief changes. Plots, feature tables and exports are
available in expandable details. No reference upload or demo selector is needed.

Run `python -m unittest discover` in the installed environment to check all branches.

## Audio comparison with idle

Upload a 44,100 Hz WAV in Engine audio to compare complete one-second windows
against the 264 saved idle feature rows. RMS, peak amplitude, crest factor and
dominant frequency are compared with each feature's 5th–95th percentile band.
A window is marked different if any feature is outside its band. The dashboard
shows the affected fraction, feature details, time traces and CSV/JSON exports.
The saved idle baseline's own outside-band fraction is shown for context.

This is descriptive comparison, not a trained fault classifier or risk estimate.
Use comparable idle operating conditions, microphone placement and recording gain.
Other sample rates still support audio analysis but cannot use this baseline.
Raw idle WAVs are absent locally; the saved CSV's extraction provenance has not
been reverified. Baseline rows are not assumed to be independent recordings.

## Local model recovery

The missing model was rebuilt using the existing training pipeline and the LiU
training archive in Downloads. The local artifact is saved at
`outputs/liu_residuals/residual_model.joblib`, with a checksum and training-data
identity in `model_manifest.json` beside it. The dashboard verifies this manifest
and labels the model as locally rebuilt. The historical official benchmark does
not apply to this replacement artifact; no official test data was used to rebuild it.
Local development validation is in `outputs/liu_residuals/validation_report.json`.
These generated files remain excluded from Git; share model and manifest together.

The sidebar verifies and loads the frozen model immediately, so connection problems
are visible before uploading. Recording quality includes time gaps, constant signals,
and a preview. Successful predictions can be exported as CSV plus a JSON summary
with the model hash, prediction coverage, alert intervals and input quality.
The evaluation tab includes individual recording results and a benchmark CSV export.

For the existing Windows environment:

```powershell
.venv/Scripts/python.exe -m streamlit run app.py
```

To use the team's frozen artifact from another folder, set `ENGINE_MODEL_PATH`
before launching. Its hash must still match the official evaluation report:

```powershell
$env:ENGINE_MODEL_PATH = 'C:\path\to\residual_model.joblib'
.venv/Scripts/python.exe -m streamlit run app.py
```

Run from this directory:

```bash
python -m pip install -r requirements-dashboard.txt
python -m streamlit run app.py
```

The dashboard starts with empty telemetry and audio uploads and an empty vibration upload. Upload a time-ordered
LiU-scale CSV or a WAV file in its respective tab. The required CSV names are listed
in the app. Unsupported formats, invalid numeric values and nonincreasing time are
rejected, including invalid optional ambient columns. Uploaded files are processed
in memory, including WAV on Windows, and no uploaded data retrains the model.

The model is **not included in Git**. For a fresh checkout, obtain the separately shared frozen
`residual_model.joblib` from the team and place it at
`outputs/liu_residuals/residual_model.joblib`. Without it, uploaded telemetry can
show measured trends but cannot produce predictions or alerts. The model may require
the versions in `requirements-liu.txt` to load and score reliably. Do not replace it
with a model trained on uploaded or official test data.

Both `app.py` and `dashboard.py` launch the current LiU dashboard. The obsolete
diesel dashboard has been replaced. Model loading verifies the SHA-256 from the
official evaluation report. Upload results are cached by complete file content,
model identity and file revision. Audio, vibration
and telemetry are separate analyses with no shared vehicle or clock. The evaluation
tab shows saved historical benchmark reports, not results from the upload.
