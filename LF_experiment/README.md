# Generalized low-frequency classifier (standalone experiment)

This is a **new optional training script**, not a replacement for `lf_experiment.py`. Your existing HP300 model and results remain unchanged.

## Installation

Copy `lf_generalized.py` into the existing `LF_experiment` folder next to `lf_experiment.py`. Run commands from your `CSC-4240-PROJECT` root. Use your current environment with `numpy`, `soundfile`, `scipy`, `scikit-learn`, and `joblib`. The C filter must be compiled at `Filters/filter`.

## 1. Smoke test (6 speakers, 8 configurations)

```bash
python3 LF_experiment/lf_generalized.py \
  --data-root Filters/dataset/noisy \
  --events-csv layer2_noise_events.csv \
  --filter-exe Filters/filter \
  --max-speakers 6 \
  --trees 40 \
  --out results/lf-rf-generalized-smoke
```

A small sample may not have every class represented in the held-out set. **Do not interpret the smoke-test metrics as research results**. It only verifies the command, filter, cache, and output workflow.

## 2. Complete focused experiment (recommended first)

```bash
python3 LF_experiment/lf_generalized.py \
  --data-root Filters/dataset/noisy \
  --events-csv layer2_noise_events.csv \
  --filter-exe Filters/filter \
  --preset focused \
  --out results/lf-rf-generalized-v1
```

Focused configurations: HP orders **1 and 2** at **200, 300, 400 Hz** with exact inverses, plus **300 -> 335 Hz** inverse/cutoff mismatches for each order (8 processing configurations in total). LP order = 4 and LP cutoff = the inverse cutoff. Each training sample has exactly the same physically captured noise as the existing dataset: **no fabricated noise is added**. Two independent binary RF models classify `low_frequency` and `60hz_hum`. Broadband is not modeled in this branch.

## 3. Optional extended training

```bash
python3 LF_experiment/lf_generalized.py \
  --preset full \
  --cache-dir results/lf-rf-generalized-v1/feature_cache \
  --out results/lf-rf-generalized-full
```

`--preset full` adds **150 Hz and 600 Hz** for both HP orders (12 total configurations). The `--cache-dir` option points to the focused experiment's saved feature cache. Existing configuration features are reused, and only four additional configurations need preprocessing. The **full** experiment writes a separate new model and evaluation under `results/lf-rf-generalized-full/`, preserving both the focused model and the original HP300 baseline.

**Practical caution:** At 600 Hz, low-frequency recovery may be physically limited by attenuation and 16-bit quantization. Scores there should not be interpreted as an ML-only failure.

## 4. Predict on already-reconstructed WAV

The generated model package is compatible with the original `predict` entry point:

```bash
python3 LF_experiment/lf_experiment.py predict \
  --input results/hp300-order2/automatic-inverse/reconstructed.wav \
  --model results/lf-rf-generalized-v1/model.joblib \
  --filter-exe Filters/filter \
  --lp-cutoff 335 \
  --out results/lf-generalized-speaker1-automatic
```

No Layer 1 source or existing Layer 2 code is modified. The new model is **independent** of the original `results/lf-rf-hp300-order2/model.joblib`.

## Outputs

- `model.joblib`: new two-label Random Forest model package.
- `evaluation.json`: metrics per filter configuration, plus pooled metrics.
- `experiment.json`: exact train/test speakers and filter conditions.
- `heldout_windows.csv`: scores, labels and speaker IDs per evaluation window.
- `feature_cache/SpeakerN/<config>.npz`: compressed, persistent feature cache.

### Cache behavior

Caches are tied to audio input's file modification time and size, filter executable's modification time and size, events CSV content, feature version, and processing configuration. A rerun with an unchanged `--out` folder reuses valid caches; `--cache-dir` also allows different experiments to share one cache. Use `--recompute` to deliberately force regeneration. This does not reuse features from the previous *single-configuration* HP300 training, because that previous run did not write a feature cache.

Cached feature arrays are not original audio and do not contain newly synthesized noise. Intermediate processed WAVs are deleted by default; `--keep-audio` saves WAVs at substantial disk cost.

### Train/test leakage caveats

Speaker IDs are split *before* creating filter variants. The same speaker never appears in training and test across configurations. Nevertheless, underlying physically recorded noise samples may repeat in several speakers, so scores measure recognition of known noise signatures across new speakers, not generalization to new noise sources. Timing labels come from Ardour and require consistent WAV offsets.

### Choosing the LP cutoff

For exact and 300->335 cases, the LP cutoff follows the inverse HP cutoff, matching the original experimental setup. Changing LP cutoff also changes the feature distribution. Use the `--variant` option for explicit custom conditions:

```bash
python3 LF_experiment/lf_generalized.py \
  --variant my_o2_fc250:2:250:250:250 \
  --variant my_o2_fc250_mismatch:2:250:275:275 \
  --out results/lf-custom
```

Each `--variant` has `NAME:HP_ORDER:FORWARD_HP_HZ:INVERSE_HP_HZ:LP_CUTOFF_HZ` and replaces the default preset. The script intentionally does not use fourth-order HP.

### Interpreting results

Evaluate **each individual configuration**, not just pooled accuracy. A model can have good pooled scores while failing for higher cutoffs. Missing positive examples in a small holdout can make precision/recall undefined (shown as null). The baseline experiment's speaker split is preserved when all 109 speakers are used with the same seed and fraction.
