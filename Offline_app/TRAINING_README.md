# CSC-4240 Offline App — Training (including dedicated LF)

This is a drop-in `Offline_app` update. Keep your existing `Layer_1`, `Layer_2`, `foundational`, `Filters` and already trained models untouched. Requires Python dependencies already used by the working pipeline, and compiled `Filters/filter`.

## Interactive menu

From the repository root:

```bash
python3 -m Offline_app
```

Choose **Train models / generate full-length dataset** or **Analyze WAV recording**. `python3 -m Offline_app menu` also works.

## Unified training (defaults: 60 seconds, 10 examples/source, realistic voice)

```bash
python3 -u -m Offline_app train \
  --audio-dir Filters/dataset/noisy \
  --events layer2_noise_events.json \
  --profile realistic \
  --clip-seconds 60 \
  --examples-per-source 10 \
  --out results/train-realistic-v3
```

Use **a NEW output directory** per run. Layer 1 trains on short windows (temporary PCM WAVs deleted); Layer 2 trains on 5-second event-labelled windows from the full original mixed recordings. Layer 2 does not train on inverse-reconstructed WAVs yet. No synthetic noises are generated.

Cutoff choices:
- `realistic` (default): HP 80–600 Hz, LP 2000–8000 Hz; BP lower 80–600 Hz and upper 2000–8000 Hz. Nominal central passband 600–2000 Hz, **not** guaranteed perfectly flat.
- `unrestricted`: uses frequency ranges from the Layer 1 base settings JSON unchanged.
- `custom`: use `--custom-ranges ranges.json`, a JSON object with all four keys: `highpass_cutoff_hz`, `lowpass_cutoff_hz`, `bandpass_low_hz`, `bandpass_high_hz`; each value is `[min,max]` in Hz. For BP the upper HP bound must be less than the lower LP bound.

The *Layer 1 prediction interface is unchanged*. Restriction applies only to generation of new training examples and optionally exported WAV pairs. The Layer 2 event manifest and speaker split remain as before.

## Preserve the full dataset (optional; disk-intensive)

Add `--save-full-dataset` to the training command. This mode requires **exactly** `Speaker1.wav` to `Speaker109.wav` in the selected directory, checks their PCM 16-bit/48kHz/mono format, and saves **one full-length forward-filtered and one full-length inverse-filtered WAV per recording**. It does **not** save 60-second clips or the ten random Layer 1 examples per source.

The application writes `full_length_dataset/forward/SpeakerN.wav`, `full_length_dataset/recovered/SpeakerN.wav` and `full_length_dataset/manifest.json`. For each source, the forward and reverse passes use the **same known generated filter parameters** (ground truth), not Layer 1's predicted parameters. This makes it a useful reference dataset; it is not an end-to-end predicted-reconstruction assessment. For actual model performance, use the separate analysis mode.

To *only* export a full-length paired dataset without retraining the models:

```bash
python3 -u -m Offline_app train \
  --audio-dir Filters/dataset/noisy \
  --profile realistic \
  --skip-layer1 --skip-layer2 --save-full-dataset \
  --out results/full-dataset-v3
```

Exports can be large (twice the total input audio size). If interrupted, the output directory remains with an **incomplete** manifest; never confuse it for a finished dataset. Existing outputs are never overwritten.

## Outputs of a unified train run

```
results/train-realistic-v3/
  training_manifest.json
  layer1_settings.json
  layer1/model.joblib
  layer1/metrics.json
  layer2_audio_index.json
  layer2/noise_multilabel.joblib
  layer2/noise_multilabel_metrics.json
  full_length_dataset/      # only if explicitly requested
```

No changes to the existing `analyze` CLI. For example:

```bash
python3 -m Offline_app analyze Filters/dataset/noisy/Speaker1.wav \
  --model results/c-filter-run/model.joblib \
  --layer2-model Layer_2/models/noise_multilabel.joblib \
  --out results/analysis-new
```

### Important

- The 109-source restriction applies **only** to the permanent full-length dataset export. Smaller training runs are permitted.
- Existing `foundational/settings.json` is read but **not modified**. The effective settings are copied to each new training run directory.
- Forward and reverse filtering of band-pass recordings each use **two independent C stages** in a private working directory to avoid the shared `int.wav` name.
- Outputs preserve source frame counts; spectral correctness is not asserted by the basic file-validation checks.
- Training currently operates on known mixed noise and trains Layer 2 separately from Layer 1. Its predictive performance on reconstructed audio remains an evaluation question.

## Dedicated low-frequency model training

The unified interactive training menu now asks **Train specialized LF model?**.
Choose **standard** for a faster baseline (second-order HP300 -> matching inverse
HP300 -> fourth-order LP300 -> 5-second Random Forest classification). The
standard model predicts `low_frequency`, `60hz_hum`, and `broadband`. Choose
**generalized** for the slower eight-variant filter-robust model, which predicts
only `low_frequency` and `60hz_hum`. Both use the actual noise-region CSV
`layer2_noise_events.csv` and the recorded noisy WAV dataset; neither creates
synthetic noise sources. These LF training options are separate from the Layer 1
cutoff profile and Layer 1 sample duration.

To train **only LF** after Layer 1 and full-band Layer 2 have finished, use a
new output directory and do not disturb the existing training run:

```bash
python3 -m Offline_app train \
  --audio-dir Filters/dataset/noisy \
  --skip-layer1 --skip-layer2 --train-lf --lf-mode standard \
  --lf-events layer2_noise_events.csv \
  --out results/lf-reproduction
```

For the generalized LF model, change `--lf-mode standard` to
`--lf-mode generalized`. The generalized run can be long (8 filter variants
per recording). Neither LF training path saves the intermediate full-length
WAV files by default. The generalized trainer does save reusable compressed
`.npz` feature caches under its output directory.

LF model outputs appear in `OUT/lf/model.joblib` and
`OUT/lf/evaluation.json`, and the consolidated run logs them in
`OUT/training_manifest.json`. Standard output also includes
`OUT/lf/heldout_windows.json` and generalized output includes
`OUT/lf/experiment.json` and `OUT/lf/heldout_windows.csv`.

After validation, activate the LF model explicitly:

```bash
mkdir -p models/lf
cp results/lf-reproduction/lf/model.joblib models/lf/model.joblib
```

The interactive **Analyze WAV** menu uses `models/lf/model.joblib`.
Training does *not* silently overwrite the active inference model.
