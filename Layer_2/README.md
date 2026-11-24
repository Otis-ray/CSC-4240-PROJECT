# Layer 2 — Ardour-event multi-label refactor

This replaces the previous single-label, whole-recording classifier with four independent presence classifiers (`low_frequency`, `high_frequency`, `broadband`, `60hz_hum`). Each 5-second window can have zero, one, or multiple labels. `lf+hf` counts as two independent positives, not a fifth class. The original 15 spectral features and Random Forest design are retained.

## Requirements

From the project root on Ubuntu:

```bash
sudo apt install python3-numpy python3-scipy python3-sklearn python3-joblib
```

The audio input must be **48,000 Hz, mono, 16-bit PCM WAV**. The input dataset is the original noisy mixed recordings (e.g. `Filters/dataset/noisy/Speaker1.wav`), **not the clean speech-only recordings**. The JSON event manifest generated from the Ardour projects is also required. The scripts do **not** synthesize new audio and do **not** duplicate WAVs.

## 1. Make the audio index

Copy `layer2_noise_events.json` into your project root, then:

```bash
python3 -m Layer_2.prepare_dataset \
  --events layer2_noise_events.json \
  --audio-dir Filters/dataset/noisy \
  --output Layer_2/data/audio_index.json \
  --strict
```

The index resolves each `SpeakerN` from the Ardour event labels to `SpeakerN.wav`, verifies WAV format, and warns about events extending beyond the WAV duration. `--strict` refuses missing recordings and duration mismatches. If the export length intentionally differs from the Ardour timeline, **investigate and correct timing alignment** before using labels. Timestamps are assumed to be relative to export sample 0, which should correspond to Ardour timeline 0. Checking Speaker1 is a good initial validation but not proof for every session.

## 2. Train and evaluate

```bash
python3 -u -m Layer_2.training_model \
  --manifest Layer_2/data/audio_index.json \
  --output Layer_2/models/noise_multilabel.joblib
```

Saves `noise_multilabel.joblib` and `noise_multilabel_metrics.json`. Train/test split is by **speaker ID**, never by window, and evaluation reports per-noise precision/recall/F1, micro/macro F1 and exact-match accuracy. Source audio is streamed rather than loaded in its entirety. Each window uses existing 15 audio features. Windows with 10–80% label overlap for any class are omitted during training, to avoid uncertain boundaries; all other windows use >=80% coverage as positive and <=10% as negative. Prediction uses all windows and provides approximate 5-second resolution. The last shorter window is included for prediction but excluded from training.

For a fast software smoke test, you can append `--max-windows-per-speaker 30`, but **do not use those metrics as final validation** (it only includes the beginning of each recording and may exclude some labels). Use the complete recording for evaluation.

## 3. Predict on a single recording

```bash
python3 -m Layer_2.predict \
  --model Layer_2/models/noise_multilabel.joblib \
  --input results/analysis-speaker1/reconstructed.wav \
  --output results/analysis-speaker1/layer2_prediction.json
```

The output JSON contains `noise_types` (independent flags and peak scores), `windows` (time interval, four scores/flags), and merged `events` with start/stop time and mean score. Overlapping noise types are supported; the `0.5` default threshold is a starting point, **not a calibrated probability**. Use `--threshold 0.6` to experiment.

**Important performance caveat:** This first model trains on your *noisy mixed recordings*, whereas the final pipeline predicts on *inverse-filtered recordings*. Performance may change due to reconstruction artifacts. Evaluate end-to-end on held-out reconstructed recordings before claiming forensic detection accuracy. Likewise, project-region labels record placement, not certainty that the noise was audible above speech in every window. The 15 whole-window spectral features may be inadequate for low-level noise; improving them can happen after integration.

**Not included:** frequency-band event localization or calibrated probabilities, automated alignment detection, reconstruct-then-train dataset augmentation, a dedicated event-presence model, or integration into the Offline_app controller. Window event boundaries are approximate (5-second grid). This refactor delivers a compatible standalone Layer 2 training and prediction API.

## Run tests

```bash
python3 -m unittest discover -s Layer_2/tests -v
```

Use `python3 -m ...` from the project root rather than running scripts by path, because they use package imports.
