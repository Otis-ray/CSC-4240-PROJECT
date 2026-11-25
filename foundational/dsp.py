"""Project Butterworth C executable adapter and PCM quantization helpers.

No SciPy DSP is used. The filter CLI requires 48 kHz, mono, PCM_16 WAV.
Band-pass `order` is the *total* pole count (2/4/8), while the C CLI
expects the per-stage order (1/2/4).
"""
from __future__ import annotations

import os
from numbers import Integral, Real
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

FILTER_TYPES = ("lowpass", "highpass", "bandpass")
FAMILIES = ("butter",)
STAGES = ("before_filter", "after_filter")
MIN_BITS = 2
MAX_BITS = 32
SAMPLE_RATE = 48000
PCM_BITS = 16
_DEFAULT_EXECUTABLE = Path(__file__).resolve().parents[1] / "Filters" / ("filter.exe" if os.name == "nt" else "filter")


def normalize_spec(spec: dict, sample_rate: float) -> dict:
    if set(spec) - {"type", "family", "order", "cutoff"}:
        raise ValueError(f"unexpected filter spec fields: {set(spec) - {'type', 'family', 'order', 'cutoff'}}")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"C filters require {SAMPLE_RATE} Hz, got {sample_rate}")
    kind = spec.get("type")
    if kind not in FILTER_TYPES:
        raise ValueError(f"unsupported filter type: {kind!r}")
    family = spec.get("family", "butter")
    if family != "butter":
        raise ValueError(f"C filter only supports butter, got {family!r}")
    order = spec.get("order")
    if not isinstance(order, Integral) or isinstance(order, bool):
        raise ValueError("order must be an integer")
    allowed = (2, 4, 8) if kind == "bandpass" else (1, 2, 4)
    if order not in allowed:
        raise ValueError(f"{kind} order must be one of {allowed}, got {order}")
    cutoff = spec.get("cutoff")
    if kind == "bandpass":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise ValueError("bandpass cutoff must be [low, high]")
        if any(not isinstance(v, Real) or isinstance(v, bool) for v in cutoff):
            raise ValueError("cutoff must contain numeric frequencies")
        lo, hi = map(float, cutoff)
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi < SAMPLE_RATE / 2):
            raise ValueError(f"invalid bandpass cutoff {cutoff}")
        cutoff = [lo, hi]
    else:
        if not isinstance(cutoff, Real) or isinstance(cutoff, bool):
            raise ValueError("cutoff must be numeric")
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or not (0 < cutoff < SAMPLE_RATE / 2):
            raise ValueError(f"invalid cutoff {cutoff}")
    return {"type": kind, "family": family, "order": int(order), "cutoff": cutoff}


def _run_stage(executable: Path, kind: str, order: int, cutoff: float, input_wav: Path, output_wav: Path, expected_frames: int):
    # C CLI requires *both* frequency arguments > 0, even for a single filter.
    hp = cutoff if kind == "highpass" else 1000.0
    lp = cutoff if kind == "lowpass" else 1000.0
    choice = 2 if kind == "highpass" else 1
    args = [str(executable), kind, "butter", str(order), f"{hp:.12g}", f"{lp:.12g}", str(input_wav), str(output_wav), str(choice)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"C filter exited {result.returncode}: {' '.join(args)}\n{result.stderr}\n{result.stdout}")
    # C filter routines return void and main can return success even when WAV I/O fails.
    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"C filter produced no output: {output_wav}\n{result.stderr}\n{result.stdout}")
    try:
        info = sf.info(output_wav)
    except RuntimeError as exc:
        raise RuntimeError(f"C filter produced unreadable WAV: {output_wav}") from exc
    if (info.samplerate, info.channels, info.frames, info.subtype) != (SAMPLE_RATE, 1, expected_frames, "PCM_16"):
        raise RuntimeError(f"C filter output mismatch: {info}; expected {expected_frames} frames of 48k mono PCM_16")


def apply_filter(audio, sample_rate: float, spec: dict, *, executable: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Apply C forward Butterworth filter; return filtered samples and metadata.

    Each example uses a private temporary directory. No shared int.wav path is
    used, so concurrent runs cannot overwrite one another's intermediate audio.
    """
    full = normalize_spec(spec, sample_rate)
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono 1-D array")
    if len(samples) == 0:
        raise ValueError("audio is empty")
    binary = Path(executable or os.environ.get("CSC4240_FILTER_BIN", _DEFAULT_EXECUTABLE)).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"C filter executable missing: {binary}. Build it using 'make -C Filters'.")
    record = {**full, "sample_rate": float(sample_rate), "processing": "project C Butterworth forward filter; PCM_16 input/output", "executable": str(binary)}
    with tempfile.TemporaryDirectory(prefix="csc4240_filter_") as tmp:
        temp = Path(tmp)
        source = temp / "source.wav"
        filtered = temp / "filtered.wav"
        sf.write(source, samples, SAMPLE_RATE, subtype="PCM_16")
        if full["type"] == "bandpass":
            # The C CLI's option 3 uses a shared 'int.wav' in the working
            # directory. Run both independent stages instead to avoid races.
            intermediate = temp / "highpass.wav"
            stage_order = full["order"] // 2
            low, high = full["cutoff"]
            _run_stage(binary, "highpass", stage_order, low, source, intermediate, len(samples))
            _run_stage(binary, "lowpass", stage_order, high, intermediate, filtered, len(samples))
            record["c_stage_order"] = stage_order
        else:
            _run_stage(binary, full["type"], full["order"], full["cutoff"], source, filtered, len(samples))
            record["c_stage_order"] = full["order"]
        result, _ = sf.read(filtered, dtype="float64")
    return result, record


def mono(audio) -> np.ndarray:
    # Convert input to a mono, float64 waveform.
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D), got {samples.shape}")
    return samples


def quantize(audio, bits: int = PCM_BITS, full_scale: float = 1.0, *, stage: str) -> tuple[np.ndarray, dict]:
    if not isinstance(bits, Integral) or isinstance(bits, bool) or not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be integer between {MIN_BITS} and {MAX_BITS}")
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    samples = mono(audio)
    # Map normalized amplitudes to signed integer quantization levels.
    step = full_scale / 2 ** (bits - 1)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    codes = np.round(samples / step)
    # Count out-of-range codes before clipping and dequantization.
    clipped = int(np.count_nonzero((codes < lo) | (codes > hi)))
    return np.clip(codes, lo, hi) * step, {"bits": int(bits), "stage": stage, "full_scale": float(full_scale), "step": step, "clipped_samples": clipped}
