from __future__ import annotations
from numbers import Integral, Real
import numpy as np
from scipy import signal

##Filter Config and Constants

FILTER_TYPES = ("lowpass", "highpass", "bandpass")
FAMILIES = ("butter", "cheby1", "cheby2", "ellip")

DEFAULT_PASSBAND_RIPPLE_DB = 1.0
DEFAULT_STOPBAND_ATTEN_DB = 40.0

_SPEC_KEYS = {"type", "family", "order", "cutoff", "passband_ripple_db", "stopband_atten_db"}

_FAMILIES_WITH_RIPPLE = ("cheby1", "ellip")
_FAMILIES_WITH_ATTENUATION = ("cheby2", "ellip")

_PROCESSING_NOTE = "causal sosfilt, single pass, zero initial state"

#!COMMENT DESCRIBING FUNCTIONS FOR CHECKING
#!ENSURE CHECKS ARE TRULY NEEDED 
def _check_keys(spec: dict) -> None:
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown filter spec keys: {sorted(unknown)}")


def _check_type(spec: dict) -> str:
    kind = spec.get("type")
    if kind not in FILTER_TYPES:
        raise ValueError(f"filter type must be one of {FILTER_TYPES}, got {kind!r}")
    return kind


def _check_family(spec: dict) -> str:
    family = spec.get("family", "butter")
    if family not in FAMILIES:
        raise ValueError(f"filter must be one of {FAMILIES}, got {family!r}")
    return family


def _check_order(spec: dict, kind: str) -> int:
    order = spec.get("order")
    if order is None:
        raise ValueError("filter spec is missing an 'order'")

    is_whole_number = isinstance(order, Integral) and not isinstance(order, bool)
    if not is_whole_number or order <= 0:
        raise ValueError(f"order must be a positive integer, got {order!r}")

    order_int = int(order)
    if kind == "bandpass" and order_int % 2 != 0:
        raise ValueError(f"band-pass order must be even (total poles), got {order_int}")

    return order_int


def _check_bandpass_cutoff(cutoff, nyquist: float) -> list[float]:
    if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
        raise ValueError(f"band-pass cutoff must be [low, high] in Hz, got {cutoff!r}")

    low, high = cutoff[0], cutoff[1]
    is_low_num = isinstance(low, Real) and not isinstance(low, bool)
    is_high_num = isinstance(high, Real) and not isinstance(high, bool)

    if not (is_low_num and is_high_num):
        raise ValueError(f"band-pass cutoffs must be numbers, got {cutoff!r}")

    low_f, high_f = float(low), float(high)
    if not 0 < low_f < high_f < nyquist:
        raise ValueError(f"band-pass cutoffs must satisfy 0 < low < high < {nyquist} Hz, got {cutoff!r}")

    return [low_f, high_f]


def _check_single_cutoff(cutoff, kind: str, nyquist: float) -> float:
    is_number = isinstance(cutoff, Real) and not isinstance(cutoff, bool)
    if not is_number:
        raise ValueError(f"{kind} cutoff must be a number in Hz, got {cutoff!r}")

    # Cast to float here so the type checker knows it's a float
    val = float(cutoff)

    if val <= 0 or val >= nyquist:
        raise ValueError(f"cutoff must be between 0 and {nyquist} Hz, got {cutoff}")
    return val

def normalize_spec(spec: dict, sample_rate: float) -> dict:
    _check_keys(spec)
    kind = _check_type(spec)
    family = _check_family(spec)
    order = _check_order(spec, kind)

    nyquist = sample_rate / 2
    if kind == "bandpass":
        cutoff_hz = _check_bandpass_cutoff(spec.get("cutoff"), nyquist)
    else:
        cutoff_hz = _check_single_cutoff(spec.get("cutoff"), kind, nyquist)

    full = {"type": kind, "family": family, "order": order, "cutoff": cutoff_hz}

    if family in _FAMILIES_WITH_RIPPLE:
        full["passband_ripple_db"] = float(spec.get("passband_ripple_db", DEFAULT_PASSBAND_RIPPLE_DB))
    if family in _FAMILIES_WITH_ATTENUATION:
        full["stopband_atten_db"] = float(spec.get("stopband_atten_db", DEFAULT_STOPBAND_ATTEN_DB))

    return full


def _design_one_side(full_spec: dict, kind: str, order: int, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    sos = signal.iirfilter(
        order,
        cutoff_hz,
        rp=full_spec.get("passband_ripple_db"),
        rs=full_spec.get("stopband_atten_db"),
        btype=kind,
        ftype=full_spec["family"],
        output="sos",
        fs=sample_rate,
    )
    return np.asarray(sos)


def design_filter(spec: dict, sample_rate: float) -> np.ndarray:
    full = normalize_spec(spec, sample_rate)
    if full["type"] != "bandpass":
        return _design_one_side(full, full["type"], full["order"], full["cutoff"], sample_rate)

    low_hz, high_hz = full["cutoff"]
    poles_per_side = full["order"] // 2

    highpass = _design_one_side(full, "highpass", poles_per_side, low_hz, sample_rate)
    lowpass = _design_one_side(full, "lowpass", poles_per_side, high_hz, sample_rate)
    return np.vstack([highpass, lowpass])


def apply_filter(audio, sample_rate: float, spec: dict) -> tuple[np.ndarray, dict]:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")

    full = normalize_spec(spec, sample_rate)
    sos = design_filter(full, sample_rate)
    filtered = signal.sosfilt(sos, samples)

    record = dict(full)
    record["sample_rate"] = float(sample_rate)
    record["processing"] = _PROCESSING_NOTE
    record["sos"] = sos.tolist()

    return np.asarray(filtered), record


def response_db(sos: np.ndarray, sample_rate: float, freqs) -> np.ndarray:
    freqs_hz = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    _, response = signal.sosfreqz(sos, worN=freqs_hz, fs=sample_rate)
    gain = np.maximum(np.abs(response), 1e-300)
    return 20 * np.log10(gain)


#!QUANTIZATION UTLITIES 
#!ADD COMMENTS ABOUT THESE THINGS

STAGES = ("before_filter", "after_filter")
MIN_BITS = 2
MAX_BITS = 32

def mono(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")
    return samples


def _check_bits(bits) -> int:
    is_whole_number = isinstance(bits, Integral) and not isinstance(bits, bool)
    if not is_whole_number:
        raise ValueError(f"bits must be an integer, got {bits!r}")
        
    val = int(bits)
    if val < MIN_BITS or val > MAX_BITS:
        raise ValueError(f"bits must be an integer from {MIN_BITS} to {MAX_BITS}, got {bits!r}")
        
    return val

def quantize(audio, bits: int = 24, full_scale: float = 1.0, *, stage: str) -> tuple[np.ndarray, dict]:
    bits = _check_bits(bits)
    if full_scale <= 0:
        raise ValueError(f"full_scale must be positive, got {full_scale}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    samples = mono(audio)
    step = full_scale / 2 ** (bits - 1)
    lowest_code = -(2 ** (bits - 1))
    highest_code = 2 ** (bits - 1) - 1

    codes = np.round(samples / step)
    too_low = codes < lowest_code
    too_high = codes > highest_code
    clipped = int(np.count_nonzero(too_low | too_high))
    codes = np.clip(codes, lowest_code, highest_code)

    record = {
        "bits": bits,
        "stage": stage,
        "full_scale": float(full_scale),
        "step": step,
        "clipped_samples": clipped,
    }

    return codes * step, record