from __future__ import annotations

import numpy as np
from scipy import signal

#floor value to avoid log10(0)
SMALLEST_POWER = 1e-30

def band_edges(n_bands: int, f_min_hz: float, f_max_hz: float) -> np.ndarray:
    #return n_bands + 1 log wise spaced freq boundaries
    return np.geomspace(f_min_hz, f_max_hz, n_bands + 1)


def band_centers(edges: np.ndarray) -> np.ndarray:
    #geometric mean gives the center freq in log freq space
    return np.sqrt(edges[:-1] * edges[1:])


def _check_inputs(samples: np.ndarray, sample_rate: float, f_min_hz: float, f_max_hz: float) -> None:
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")
    #freq bands cannot exceed nyquist limit
    nyquist = sample_rate / 2
    if not 0 < f_min_hz < f_max_hz <= nyquist:
        raise ValueError(f"need 0 < f_min_hz < f_max_hz <= {nyquist}, got {f_min_hz} and {f_max_hz}")


def _power_per_band(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bands = len(edges) - 1
    centers = band_centers(edges)

    #map band edges onto FFT bin indices for O(1) slice sums via prefix totals
    first_bin = np.searchsorted(freqs, edges[:-1])
    end_bin = np.searchsorted(freqs, edges[1:])
# Prepend 0.0 so sum of bins is running_total[end] - running_total[first]
    running_total = np.concatenate([[0.0], np.cumsum(power)])
    band_power = np.zeros(n_bands)

    for band in range(n_bands):
        first = first_bin[band]
        end = end_bin[band]
        if (end - first) > 0:
            #avg power across all bins falling inside the band
            band_power[band] = (running_total[end] - running_total[first]) / (end - first)
        else:
            #interpolate when a narrow band falls strictly between FTT freq
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
    #convert input audio to float64 for numerical stability
    samples = np.asarray(audio, dtype=np.float64)
    _check_inputs(samples, sample_rate, f_min_hz, f_max_hz)

    #clamp segment length to signal length for short clips
    segment_length = min(segment, len(samples))
    freqs, power = signal.welch(samples, fs=sample_rate, nperseg=segment_length)

    edges = band_edges(n_bands, f_min_hz, f_max_hz)
    band_power = _power_per_band(freqs, power, edges)

    #convert power to relative deciabls and center around the median
    db = 10 * np.log10(np.maximum(band_power, SMALLEST_POWER))
    return db - np.median(db)