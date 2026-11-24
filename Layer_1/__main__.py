from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from foundational.records import load_settings
from Layer_1.dataset import build_dataset
from Layer_1.features import spectrum_features
from Layer_1.model import (
    FilterIdModel,
    _pick_rows,
    evaluate,
    load_model,
    save_model,
    split_by_recording,
)

DEFAULT_SETTINGS = Path("Foundational/settings.json")

def _print_progress(done: int, total: int) -> None:
    print(f"built {done}/{total} examples", flush = True)


def train(args) -> None:
    settings = load_settings(args.settings)
    out = Path(args.out)

    if out.exists():
        sys.exit(f"{out} already exists; pick a new folder for --out")

    started = time.perf_counter()

    try:
        data = build_dataset(settings, progress=_print_progress)
    except ValueError as error:
        sys.exit(f"can't build the training examples: {error}")

    train_rows, test_rows = split_by_recording(data.groups, settings["test_fraction"], settings["seed"])

    model = FilterIdModel()
    model.fit(data.features[train_rows], _pick_rows(data.labels, train_rows))

    test_labels = _pick_rows(data.labels, test_rows)
    test_records = _pick_rows(data.records, test_rows)
    metrics = evaluate(model, data.features[test_rows], test_labels, test_records)

    test_recordings = sorted(set(data.groups[test_rows].tolist()))
    metrics["train_examples"] = len(train_rows)
    metrics["train_recordings"] = len(set(data.groups[train_rows].tolist()))
    metrics["test_recordings"] = len(test_recordings)
    metrics["test_recording_ids"] = test_recordings
    metrics["seconds"] = round(time.perf_counter() - started, 1)

    out.mkdir(parents=True)
    save_model(out / "model.joblib", model, settings)
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"type accuracy {metrics['type_accuracy']:.3f}, type + order {metrics['type_and_order_accuracy']:.3f}")
    print(f"saved to {out}")


def _read_mono_wav(path: str, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float64")
    if audio.ndim != 1:
        sys.exit(f"{path}: expected a mono WAV file")
    if rate != sample_rate:
        sys.exit(f"{path}: {rate} Hz, but the model was trained at {sample_rate} Hz")
    return audio


def predict(args) -> None:
    model, settings = load_model(args.model)
    for path in args.wavs:
        audio = _read_mono_wav(path, settings["sample_rate"])
        features = spectrum_features(audio, settings["sample_rate"], **settings["features"])
        result = model.predict(features)[0]
        print(json.dumps({"file": str(path), **result}))


def _build_parser() -> argparse.ArgumentParser:
    description = (__doc__ or "").strip().split("\n\n")[0]
    parser = argparse.ArgumentParser(prog="python -m Layer_1", description=description)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a model on the WAV files in the settings' source_dir")
    train_parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="a settings JSON file")
    train_parser.add_argument("--out", required=True, help="a new folder for the model, settings, and scores")
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict", help="predict the filter on WAV files")
    predict_parser.add_argument("--model", required=True, help="a trained model.joblib")
    predict_parser.add_argument("wavs", nargs="+", help="mono WAV files at the model's sample rate")
    predict_parser.set_defaults(run=predict)

    return parser


def main(argv = None) -> None:
    args = _build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
