# Hackhive-EEF-

## Current scope

Two separate demonstrations: engine-audio feature extraction and an experimental
LiU engine sensor-consistency monitor. The recordings are unmatched; no sensor
fusion or audio fault classification is claimed. The monitor flags disagreement
between measured and predicted signals. It does not identify the faulty component
or establish engine health when no anomaly is detected.

The frozen residual model detected 32.3% of faulty samples on the official test
set, with 4.2% false alarms across normal samples (7.6% on the full normal run).
Detection was strongest in intercooler-pressure fault scenarios: 87.5% pooled
recall. This is a scenario-specific result, not a fault-isolation accuracy.
See `reports/official_test.json` for all recordings, limitations and model hash.
The test set has now been inspected; further tuning needs fresh independent
evaluation before claiming a new final-test result.

## Sensor monitor setup and prediction

Use Python 3.13, matching the training environment:

```bash
python3.13 -m venv .venv-liu
./.venv-liu/bin/python -m pip install -r requirements-liu.txt
./.venv-liu/bin/python predict.py --input data/recording.csv --output outputs/prediction.csv
```

On Windows create the environment with `py -3.13 -m venv .venv-liu`, then use
`.venv-liu\Scripts\python.exe` instead of `./.venv-liu/bin/python`.

The default model is `outputs/liu_residuals/residual_model.joblib`. It is ignored
by Git. Share this frozen artifact with the dashboard teammate separately and
verify its SHA-256 against `reports/official_test.json`; do not retrain merely
to run a demo. Only load model artifacts from trusted sources.

Input is one time-ordered recording in original LiU numeric scale. Required
columns: `time`, `Intercooler_pressure`, `intake_manifold_pressure`,
`intercooler_temperature`, `air_mass_flow`, `engine_speed`, `throttle_position`,
`wastegate_position`, `injected_fuel_mass`. Complete physical units remain
unverified, so do not relabel engine speed as RPM or silently convert inputs.

## Dashboard handoff

The upload-driven dashboard is implemented in `dashboard.py`; `app.py` launches
the same dashboard. Install and run in the model environment:

```bash
./.venv-liu/bin/python -m pip install -r requirements-dashboard.txt
./.venv-liu/bin/python -m streamlit run app.py
```

See `DASHBOARD.md` for setup. Both entry points support CSV/WAV uploads and a
historical evaluation tab. WAV processing is in memory for Windows compatibility.
An optional STFT spectrogram is included. Backend interface:

```python
import joblib
from predict import predict_recording

bundle = joblib.load("outputs/liu_residuals/residual_model.joblib")
rows = predict_recording(uploaded_dataframe, bundle)
```

Load the model once. Pass a whole recording, not individual isolated rows.
Outputs preserve input row order and include:

- `time`: elapsed seconds.
- `anomaly_score`: normalized disagreement, not a probability. Alert when > 1.
- `prediction_available`: false for initial history or after a time gap.
- `prediction`: 0/1 when available, otherwise blank.
- `status`: `insufficient_history`, `no_anomaly_detected`, `anomaly_detected`.
- Actual and predicted columns for manifold pressure, intercooler pressure and airflow.

Show actual/predicted traces, score with a threshold line at 1, and status.
Do not label a zero prediction as confirmed healthy, display a diagnosis, or
turn the score into confidence percentages. Display validation limitations.
Catch `ValueError` for invalid uploads and show its message.
Keep the audio view separate and process WAV uploads with one-second windows.

## Reproducing development (not needed for prediction)

```bash
./.venv-liu/bin/python prepare_liu_data.py --input /path/to/trainingdata.zip
./.venv-liu/bin/python define_liu_evaluation.py
OMP_NUM_THREADS=2 ./.venv-liu/bin/python train_model.py
```

Existing output directories are protected from overwriting. Training now means
the selected residual model, not the superseded binary classifier. It performs
eight recording-level folds, then fits and calibrates the final artifact using
normal training data. Calibration uses reserved time blocks, not test data.
Labels assume normal before 110 seconds and faulty after 130 seconds in fault
recordings; the transition is excluded. These are approximate timing assumptions.

`evaluate_liu_test.py --test-zip /path/to/testdata.zip` evaluates a frozen artifact
without fitting and refuses to overwrite existing official evaluation outputs.
`liu_metrics.py` contains shared metrics, `liu_residual_model.py` the model logic.
Reports are retained in `reports/`; raw recordings, models and large predictions
remain under ignored `data/` and `outputs/`.

Superseded diesel models, their old audit, and the LiU classifier/history trials
were removed from the source tree. Their generated local reports remain under
ignored `outputs/` for audit. A recovery archive was saved outside the repository.

## Remaining project work

1. Rehearse the connected dashboard on the presentation computer.
2. Share the frozen model and representative recordings outside Git.
3. Rehearse normal, detectable-fault, missed-fault and invalid-upload cases.
4. Present the full test results, including weak fault families and false alarms.
5. For broader reliability, obtain additional independent normal/fault recordings
   and verified units; reserve new test data before any further model changes.

## Audio pipeline

Install dependencies with `python -m pip install -r requirements.txt`.

Generate a six-second processing test (100 Hz, 400 Hz, silence) and export features:

```bash
python audio_processing.py --demo --output outputs/demo_features.csv
```

For subsequent runs or real WAV recordings:

```bash
python audio_processing.py --input data/generated_test.wav --output outputs/demo_features.csv
python audio_processing.py --input data/engine.wav --output outputs/engine_features.csv --window-seconds 1
```

Audio is converted to mono full-scale floats without per-window volume normalization.
The original sample rate is preserved. Windows default to two seconds, without overlap.
The incomplete final window is skipped and reported. Each CSV row includes the file
stem as recording ID, start/end times, sampling rate, and four shared audio features.
Use unique filenames across recordings. Labels are not inferred or invented.
The generated recording tests processing only; it is not engine training data.
Data, output files and virtual environments are excluded from Git.
