from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# core dsp operations, provenance/seed helpers, and spectral feature extraction
from foundational.dsp import FILTER_TYPES, apply_filter, quantize
from foundational.records import build_record, example_seed
from Layer_1.features import spectrum_features

# target RMS level for normalizing source audio before filtering
SOURCE_RMS = 0.1
# Print/invoke the progress callback every N examples.
PROGRESS_EVERY = 500


@dataclass
class Dataset:
    # Store feature rows, labels, recording groups, and provenance.
    features: np.ndarray
    labels: list[dict]
    groups: np.ndarray
    records: list[dict]

 # Required top-level training settings.

_REQUIRED_SETTINGS = (
    "seed",
    "sample_rate",
    "source_dir",
    "clip_seconds",
    "examples_per_source",
    "test_fraction",
    "bits",
    "filters",
    "features",
)

# Required filter-configuration keys.
_REQUIRED_FILTER_SETTINGS = (
    "types",
    "orders",
    "highpass_cutoff_hz",
    "lowpass_cutoff_hz",
    "bandpass_low_hz",
    "bandpass_high_hz",
)

def check_settings(settings: dict) -> None:
    # Verify required top-level settings.
    for key in _REQUIRED_SETTINGS:
        if key not in settings:
            raise ValueError(f'settings are missing "{key}"')

    if not isinstance(settings["source_dir"], str) or not settings["source_dir"]:
        raise ValueError('"source_dir" must name a folder of mono WAV files')

    if settings["examples_per_source"] < 1:
        raise ValueError(f"examples_per_source must be at least 1, got {settings['examples_per_source']}")

    if not 0 < settings["test_fraction"] < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {settings['test_fraction']}")

    # validate filter sub dict structure and bounds
    if settings["sample_rate"] != 48000 or settings["bits"] != 16:
        raise ValueError("C filters require sample_rate=48000 and bits=16")

    filters = settings["filters"]
    for key in _REQUIRED_FILTER_SETTINGS:
        if key not in filters:
            raise ValueError(f'settings["filters"] is missing "{key}"')

    for bp_key in ("bandpass_low_hz", "bandpass_high_hz"):
        val = filters[bp_key]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f'filters["{bp_key}"] must be a [low, high] list')

    for kind in filters["types"]:
        if kind not in FILTER_TYPES:
            raise ValueError(f"unknown filter type in settings: {kind!r}")
        if not filters["orders"].get(kind):
            raise ValueError(f'settings["filters"]["orders"] has no orders for {kind!r}')

    # Ensure the band-pass cutoff ranges do not overlap.
    if filters["bandpass_low_hz"][1] >= filters["bandpass_high_hz"][0]:
        raise ValueError("bandpass_low_hz has to end below where bandpass_high_hz starts")


def wav_sources(source_dir, sample_rate: int, clip_seconds: float | None) -> list[dict]:
    paths = sorted(Path(source_dir).glob("*.wav"))
    if len(paths) == 0:
        raise ValueError(f"no .wav files in {source_dir}")

    sources = []
    for path in paths:
        # Inspect the WAV header without loading the full recording.
        info = sf.info(path)
        if info.samplerate != sample_rate:
            raise ValueError(f"{path.name} is {info.samplerate} Hz, but this run uses {sample_rate} Hz")
        if info.channels != 1:
            raise ValueError(f"{path.name} has {info.channels} channels; sources must be mono")

        if clip_seconds is not None:
            clip_samples = round(clip_seconds * sample_rate)
            if info.frames < clip_samples:
                raise ValueError(f"{path.name} is shorter than one {clip_seconds} s clip")

        # keep _path internal so provenance records only expose public field
        sources.append({"recording_id": path.stem, "kind": "wav", "file": path.name, "_path": str(path)})

    return sources


def make_sources(settings: dict) -> list[dict]:
    return wav_sources(settings["source_dir"], settings["sample_rate"], settings["clip_seconds"])


def render_source(source: dict, seconds: float | None, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    if seconds is None:
        audio, _ = sf.read(source["_path"], dtype="float64")
    else:
        # pick random valid start offset for fixed length clip
        n_samples = round(seconds * sample_rate)
        total_frames = sf.info(source["_path"]).frames
        start = int(rng.integers(0, total_frames - n_samples + 1))
        audio, _ = sf.read(source["_path"], start=start, frames=n_samples, dtype="float64")

    # guard against silent segments before RMS normalization
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        raise ValueError(f"source {source['recording_id']} gave a silent clip")

    return audio * (SOURCE_RMS / rms)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    # Sample frequencies uniformly in log space across octaves.
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_filter(filters: dict, rng: np.random.Generator) -> dict:
    kind = str(rng.choice(filters["types"]))
    order = int(rng.choice(filters["orders"][kind]))

    # Band-pass filters use both low and high cutoff frequencies.
    if kind == "bandpass":
        low = _log_uniform(rng, *filters["bandpass_low_hz"])
        high = _log_uniform(rng, *filters["bandpass_high_hz"])
        cutoff = [low, high]
    else:
        cutoff = _log_uniform(rng, *filters[f"{kind}_cutoff_hz"])

    return {"type": kind, "family": "butter", "order": order, "cutoff": cutoff}


def _public(source: dict) -> dict:
    # strip private/internal keys prefixed with _
    return {key: value for key, value in source.items() if not key.startswith("_")}


def make_example(index: int, sources: list[dict], settings: dict) -> tuple[np.ndarray, dict]:
    # derive deterministic per example seed for reproducible generation
    seed = example_seed(settings["seed"], index)
    rng = np.random.default_rng(seed)

    source = sources[index // settings["examples_per_source"]]
    sample_rate = settings["sample_rate"]

    # Normalize audio, apply the sampled filter, and simulate quantization.
    audio = render_source(source, settings["clip_seconds"], sample_rate, rng)
    spec = sample_filter(settings["filters"], rng)

    audio, filter_record = apply_filter(audio, sample_rate, spec)
    audio, rounding = quantize(audio, bits=settings["bits"], stage="after_filter")

    example_id = f"ex-{index:06d}"
    record = build_record(example_id, seed, sample_rate, [_public(source)], filter_record, [rounding])

    return audio, record


def label_of(record: dict) -> dict:
    # extract ground truth classification/regression targets from applied filter
    applied = record["filter"]
    return {"type": applied["type"], "order": applied["order"], "cutoff": applied["cutoff"]}


def build_dataset(settings: dict, progress=None) -> Dataset:
    check_settings(settings)
    sources = make_sources(settings)
    total = len(sources) * settings["examples_per_source"]

    features, labels, groups, records = [], [], [], []

    for index in range(total):
        audio, record = make_example(index, sources, settings)
        features.append(spectrum_features(audio, settings["sample_rate"], **settings["features"]))
        labels.append(label_of(record))
        # use recording id as the grouping key for recording level train/test splitting
        groups.append(record["sources"][0]["recording_id"])
        records.append(record)

        examples_done = index + 1
        if progress is not None and examples_done % PROGRESS_EVERY == 0:
            progress(examples_done, total)

    return Dataset(np.array(features), labels, np.array(groups), records)
