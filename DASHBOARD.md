# Engine signal dashboard

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

The dashboard starts with empty telemetry and audio uploads. Upload a time-ordered
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
model identity and file revision. Audio
and telemetry are separate analyses with no shared vehicle or clock. The evaluation
tab shows saved historical benchmark reports, not results from the upload.
