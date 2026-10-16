# Engine signal dashboard

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

The model is **not included in this archive**. Obtain the separately shared frozen
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
