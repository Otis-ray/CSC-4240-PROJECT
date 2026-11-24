# Offline_app: canonical inference models and LF isolation override

This patch changes **only** `Offline_app/`; trained models and C filters are untouched.

The application now defaults to one active model per subsystem:

    models/layer1/model.joblib
    models/layer2/model.joblib
    models/lf/model.joblib

From project root, set up once (copies shown to avoid symlink issues on other machines):

```bash
mkdir -p models/{layer1,layer2,lf}
cp results/train-new/layer1/model.joblib models/layer1/model.joblib
cp results/train-new/layer2/noise_multilabel.joblib models/layer2/model.joblib
cp results/lf-rf-generalized-v1/model.joblib models/lf/model.joblib
```

These are *inference* defaults; training still writes versioned runs to `results/` and **does not automatically replace active models**. Explicitly copy a new trained model into `models/` after evaluating it. The CLI retains `--model`, `--layer2-model`, and `--lf-model` overrides for diagnostics and backward compatibility.

## CLI

```bash
python3 -m Offline_app analyze Filters/processed/forward/48kBounce_filt_4th.wav \
  --lf-cutoff 400 --out results/new-recording-lp400
```

Omit `--lf-cutoff` to use Layer 1's predicted HP cutoff. `--no-lf` disables the LF branch. `--skip-inverse` runs the original direct Layer 2 diagnostic, automatically disabling LF. All previous output safeguards still apply (output dir must not exist).

## Menu

```bash
python3 -m Offline_app menu
```

Select option 2, enter WAV file, then respond to the LF isolation cutoff prompt: leave blank for automatic cutoff or type `400` to override. The menu no longer asks for per-layer model paths. Default model paths are printed for transparency.

Both `manifest.json` and `report.json` record `lf.isolation_cutoff_hz` and `lf.isolation_cutoff_source` (`manual` / `layer1`). The manifest additionally records `lf.predicted_hp_cutoff_hz` even with a manual cutoff.

Note: for a Layer 1 predicted *low-pass-only* filter, the LF branch remains skipped, regardless of cutoff override; no HP suppression was identified to reverse.

## Tests

```bash
python3 -m unittest discover -s Offline_app/tests -v
```
