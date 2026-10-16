# Engine signal dashboard

Run from this directory:

```bash
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard.py
```

The dashboard starts with empty telemetry and audio uploads. Upload a time-ordered
LiU-scale CSV or a WAV file in its respective tab. The required CSV names are listed
in the app. Unsupported formats, invalid numeric values and nonincreasing time are
rejected. Uploaded files are processed in memory (WAV is briefly copied into a
temporary file for the existing loader), and no uploaded data retrains the model.

The model is **not included in this archive**. Obtain the separately shared frozen
`residual_model.joblib` from the team and place it at
`outputs/liu_residuals/residual_model.joblib`. Without it, uploaded telemetry can
show measured trends but cannot produce predictions or alerts. The model may require
the versions in `requirements-liu.txt` to load and score reliably. Do not replace it
with a model trained on uploaded or official test data.

This separate entry point leaves the existing `app.py` and its teammate's changes
untouched. Use `dashboard.py` for the LiU telemetry and WAV upload experience. Audio
and telemetry are separate analyses with no shared vehicle or clock. The evaluation
tab shows saved historical benchmark reports, not results from the upload.
