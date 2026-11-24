"""Audio features for Layer 2; preserves the original 15-feature extractor.

New: iter_audio_windows() provides temporal feature vectors without averaging away
where an event occurred. Audio is restricted to 48 kHz mono 16-bit PCM.
"""
import wave
import numpy as np
from scipy.signal import welch

SAMPLE_RATE = 48000
WINDOW_SECONDS = 5
FREQUENCY_BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 24000)]
FEATURE_NAMES = [
    "rms", "peak_amplitude", "zero_crossing_rate", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
    "spectral_entropy", "peak_frequency"
] + [f"energy_{low}_{high}" for low, high in FREQUENCY_BANDS]


def calculate_features(audio):
    if len(audio) < 256:
        raise ValueError("Audio segment is too short.")
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    zero_crossings = np.mean(np.signbit(audio[:-1]) != np.signbit(audio[1:]))
    frequencies, power = welch(audio, fs=SAMPLE_RATE, nperseg=min(4096, len(audio)))
    total_power = np.sum(power)
    if total_power <= 1e-20:
        return np.array([rms, peak, zero_crossings] + [0.0] * (len(FEATURE_NAMES) - 3), dtype=np.float64)
    normalized_power = power / total_power
    centroid = np.sum(frequencies * normalized_power)
    bandwidth = np.sqrt(np.sum((frequencies - centroid) ** 2 * normalized_power))
    rolloff_index = min(np.searchsorted(np.cumsum(normalized_power), 0.85), len(frequencies) - 1)
    rolloff = frequencies[rolloff_index]
    flatness = np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20)
    entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-20))
    peak_frequency = frequencies[np.argmax(power)]
    band_energies = []
    for low, high in FREQUENCY_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        band_energies.append(np.sum(normalized_power[mask]))
    return np.array([
        rms, peak, zero_crossings, centroid, bandwidth, rolloff,
        flatness, entropy, peak_frequency, *band_energies
    ], dtype=np.float64)


def validate_wav(wav):
    if (wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1
            or wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE'):
        raise ValueError('Audio must be 48 kHz, mono, 16-bit PCM WAV.')


def iter_audio_windows(file_path, window_seconds=WINDOW_SECONDS, include_partial=False):
    """Yield (start_s, end_s, feature_vector) for nonoverlapping windows.

    Opens audio once and streams in bounded chunks. Partial tail windows are
    optional; training defaults to complete windows only.
    """
    frames_per_window = round(SAMPLE_RATE * float(window_seconds))
    if frames_per_window < 256:
        raise ValueError('Window duration is too short.')
    with wave.open(str(file_path), 'rb') as wav:
        validate_wav(wav)
        position = 0
        while position < wav.getnframes():
            remaining = wav.getnframes() - position
            if remaining < frames_per_window and not include_partial:
                break
            count = min(remaining, frames_per_window)
            data = wav.readframes(count)
            audio = np.frombuffer(data, dtype='<i2').astype(np.float64) / 32768.0
            if len(audio) < 256:
                break
            end = position + len(audio)
            yield position / SAMPLE_RATE, end / SAMPLE_RATE, calculate_features(audio)
            position = end


def extract_features(file_path):
    """Legacy whole-file feature interface retained for compatibility."""
    vectors, weights = [], []
    for start, end, features in iter_audio_windows(file_path, window_seconds=10, include_partial=True):
        vectors.append(features)
        weights.append(end - start)
    if not vectors:
        raise ValueError('No valid audio data found.')
    return np.average(vectors, axis=0, weights=weights)
