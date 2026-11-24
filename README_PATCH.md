# LF timestamped-event reporting patch

Updates only `Offline_app/analysis.py` and `Offline_app/tests/test_lf_integration.py`.

- Merges contiguous LF-positive five-second windows into timestamped per-label events.
- Adds `lf.events` in `report.json` with `type`, `start_seconds`, `end_seconds`, `mean_confidence`, `max_confidence`, and `window_count`.
- Adds `lf.event_count` to `report.json` and `manifest.json`.
- Keeps `lf/lf_prediction.json` unchanged and does not merge or overwrite full-band Layer 2 events.
- Handles no detections (`events: []`) and maintains independent overlapping labels.

Extract into your project root (backup the two files first):

```bash
unzip -o lf_timestamp_events_patch.zip -d .
python3 -m unittest discover -s Offline_app/tests -v
```

To obtain an updated `report.json`, run analysis again with a **new** `--out` directory (the application intentionally refuses to overwrite prior analysis results). Example:

```bash
python3 -m Offline_app analyze \
  results/hp300-order2/audio/Speaker1_forward.wav \
  --model results/train-new/layer1/model.joblib \
  --layer2-model results/train-new/layer2/noise_multilabel.joblib \
  --lf-model results/lf-rf-generalized-v1/model.joblib \
  --out results/analysis-lf-integrated-speaker1-events
```

The test suite checks merge behavior, temporal gaps, overlapping label events, mean/max confidence, empty detections and failures.
