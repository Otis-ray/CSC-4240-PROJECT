from __future__ import annotations

import numpy as np
from scipy import signal

SMALLEST_POWER = 1e-30

def band_edges(n_bands: int, f_min_hz: float, f_max_hz: float) -> np.ndarray:
    return np.geomspace(f_min_hz, f_max_hz, n_bands + 1)


def band_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def _check_inputs(samples: np.ndarray, sample_rate: float, f_min_hz: float, f_max_hz: float) -> None:
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")
    nyquist = sample_rate / 2
    if not 0 < f_min_hz < f_max_hz <= nyquist:
        raise ValueError(f"need 0 < f_min_hz < f_max_hz <= {nyquist}, got {f_min_hz} and {f_max_hz}")


def _power_per_band(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bands = len(edges) - 1
    centers = band_centers(edges)

    first_bin = np.searchsorted(freqs, edges[:-1])
    end_bin = np.searchsorted(freqs, edges[1:])

    running_total = np.concatenate([[0.0], np.cumsum(power)])
    band_power = np.zeros(n_bands)

    for band in range(n_bands):
        first = first_bin[band]
        end = end_bin[band]
        if (end - first) > 0:
            band_power[band] = (running_total[end] - running_total[first]) / (end - first)
        else:
            band_power[band] = np.interp(centers[band], freqs, power)

    return band_power


def spectrum_features(
        audio,
        sample_rate: float,
        n_bands: int = 96,
        f_min_hz: float = 20.0,
        f_max_hz: float = 24_000.0,
        segment: int = 8_192,
) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)
    _check_inputs(samples, sample_rate, f_min_hz, f_max_hz)

    segment_length = min(segment, len(samples))
    freqs, power = signal.welch(samples, fs=sample_rate, nperseg=segment_length)

    edges = band_edges(n_bands, f_min_hz, f_max_hz)
    band_power = _power_per_band(freqs, power, edges)

    db = 10 * np.log10(np.maximum(band_power, SMALLEST_POWER))
    return db - np.median(db)