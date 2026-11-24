from __future__ import annotations
import json
from numbers import Integral, Real
from pathlib import Path
import numpy as np
from foundational.dsp import STAGES 
from typing import Any, cast

SCHEMA_VERSION = 2

#!ADD COMMENTS EXPLAINING CODE
def load_settings(path: str | Path) -> dict:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    seed = settings.get("seed")

    is_whole_number = isinstance(seed, Integral) and not isinstance(seed, bool)
    if seed is None or not is_whole_number or seed < 0:
        raise ValueError(f'settings need a non-negative integer "seed", got {seed!r}')

    return settings


def example_seed(base_seed: int, index: int) -> int:
    if base_seed < 0 or index < 0:
        raise ValueError("base_seed and index must be non-negative")

    mixer = np.random.SeedSequence([base_seed, index])
    state = mixer.generate_state(1, dtype=np.uint32)
    return int(state[0])


def _check_example_id(example_id: str) -> None:
    if not example_id:
        raise ValueError("example_id can't be empty")


def _is_list_of_dicts(value) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
    return True


def _check_sources(sources) -> None:
    if not _is_list_of_dicts(sources):
        raise ValueError("sources must be a list of dicts")
    if len(sources) == 0:
        raise ValueError("an example needs at least one source")

    for source in sources:
        recording_id = source.get("recording_id")
        if not isinstance(recording_id, str) or recording_id == "":
            raise ValueError(f'every source needs a "recording_id" string, got {recording_id!r}')


def _check_damage(damage) -> None:
    if not _is_list_of_dicts(damage):
        raise ValueError("damage must be a list of dicts, in the order applied")

    stages = [step.get("stage") for step in damage]
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"every damage step needs a stage from {STAGES}, got {stages}")

    seen_after_filter = False
    for stage in stages:
        if stage == "after_filter":
            seen_after_filter = True
        elif seen_after_filter:
            raise ValueError(f"damage steps must be in the order applied, before_filter first, got {stages}")


def _check_clip_events(clip, events) -> None:
    if clip is None:
        if events is not None:
            raise ValueError("events needs clip")
        return

    if not isinstance(clip, dict) or not clip:
        raise ValueError("clip must be a dict")

    for key in ("start_s", "duration_s", "gain"):
        value = clip.get(key)

        if value is None:
            raise ValueError(f"clip {key} cannot be None")

        is_number = isinstance(value, Real) and not isinstance(value, bool)
        if not is_number:
            raise ValueError(f"clip {key} must be a number")

        val = float(value)  # Now the type checker knows it's a valid numeric type

        if not np.isfinite(val):
            raise ValueError(f"clip {key} must be a finite number")

        if val < 0 or (key != "start_s" and val == 0):
            raise ValueError(f"clip {key} is out of range")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_record(
        example_id: str,
    seed: int,
    sample_rate: float,
    sources: list[dict],
    filter_record: dict,
    damage: list[dict],
    *,
    clip=None,
    events=None,
) -> dict:
    _check_example_id(example_id)
    _check_sources(sources)
    _check_damage(damage)
    _check_clip_events(clip, events)

    record = {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "seed": seed,
        "sample_rate": sample_rate,
        "sources": sources,
        "filter": filter_record,
        "damage": damage,
    }

    if clip is not None:
        record["clip"] = clip
    if events is not None:
        record["events"] = events

    return cast(dict[str, Any], to_jsonable(record))


def write_record(record: dict, path) -> None:
    text = json.dumps(to_jsonable(record), indent=2, sort_keys=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_record(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
