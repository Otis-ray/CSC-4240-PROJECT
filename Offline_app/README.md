# Offline application — Analysis integration v2

This update replaces **only `Offline_app/`**. It uses your existing `Layer_1`, `Layer_2`, `Filters/filter`, and trained `.joblib` files without changing them.

## Requirements

From the project root, verify that these exist:

- `Filters/filter` — compiled C DSP executable
- `results/c-filter-run/model.joblib` — trained Layer 1
- `Layer_2/models/noise_multilabel.joblib` — trained Layer 2 multi-label

Input WAV must be mono, 48 kHz, 16-bit PCM. Run commands from the project root.

## Normal analysis (both layers)

```bash
python3 -m Offline_app analyze \
    Filters/dataset/noisy/Speaker1.wav \
    --model results/c-filter-run/model.joblib \
    --layer2-model Layer_2/models/noise_multilabel.joblib \
    --out results/analysis-speaker1-complete
```

Output:

```
results/analysis-speaker1-complete/
    layer1_prediction.json    # original Layer 1 JSON, unchanged
    reconstructed.wav         # inverse-filtered WAV
    layer2_prediction.json    # all five-second windows, class scores and events
    manifest.json             # processing parameters, hashes and status
    report.json               # compact combined overview and event summary
```

`report.json` references `layer2_prediction.json` for per-window data, to avoid duplicating the large window array. The input WAV is referenced by absolute path and SHA-256 in the manifest, **not copied**. Output folders must be new.

## Diagnostic mode: Layer 2 directly on original audio

```bash
python3 -m Offline_app analyze \
    Filters/dataset/noisy/Speaker1.wav \
    --layer2-model Layer_2/models/noise_multilabel.joblib \
    --skip-inverse \
    --out results/analysis-speaker1-direct
```

This skips **both Layer 1 and inverse filtering**, and runs Layer 2 on the original WAV. The output contains `layer2_prediction.json`, `manifest.json`, `report.json` only. Use it to compare classification before and after reconstruction.

## Other arguments

- `--threshold 0.5`: optionally override Layer 2's default detection threshold; strict interval (0, 1).
- `--timeout 600`: maximum seconds for each launched subprocess, not the entire pipeline.
- `--filter-bin path/to/filter`: alternate compiled C binary.
- `--model path/to/model.joblib`: alternate Layer 1 trained model.
- Without `--layer2-model`, the original Layer 1 → inverse filter workflow still operates for backwards compatibility.

## Test

```bash
python3 -m unittest discover -s Offline_app/tests -v
```

## Notes

- Layer 2 model expects the **multilabel schema v2** model created by the refactor, not the older single-label model.
- Layer 2's event timestamps are quantized to five-second window edges; class scores are not calibrated probabilities.
- For valid *reconstruction* testing, forward-filter a known recording first. Directly inverse-filtering an unfiltered recording changes its spectrum and can distort Layer 2 outputs.
- A failed operation preserves `manifest.json` with status `failed` and an error. Previously completed artifacts stay in the output folder for debugging.
- The interface is command-line-only at this stage; interactive menu and training orchestration remain future stages.
