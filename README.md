# Hackhive-EEF-

## Audio pipeline

Install dependencies with `python -m pip install -r requirements.txt`.

Generate a six-second processing test (100 Hz, 400 Hz, silence) and export features:

```bash
python audio_processing.py --demo --output outputs/demo_features.csv
```

For subsequent runs or real WAV recordings:

```bash
python audio_processing.py --input data/generated_test.wav --output outputs/demo_features.csv
python audio_processing.py --input data/engine.wav --output outputs/engine_features.csv
```

Audio is converted to mono full-scale floats without per-window volume normalization.
The original sample rate is preserved. Windows default to two seconds, without overlap.
The incomplete final window is skipped and reported. Each CSV row includes the file
stem as recording ID, start/end times, sampling rate, and four shared audio features.
Use unique filenames across recordings. Labels are not inferred or invented.
The generated recording tests processing only; it is not engine training data.
Data, output files and virtual environments are excluded from Git.
