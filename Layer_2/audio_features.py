
import wave
import numpy as np
from scipy.signal import welch

SAMPLE_RATE = 48000
WINDOW_SECONDS = 10
FREQUENCY_BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 24000)]

FEATURE_NAMES = [
    "rms", "peak_amplitude", "zero_crossing_rate", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
    "spectral_entropy", "peak_frequency"
] + [f"energy_{low}_{high}" for low, high in FREQUENCY_BANDS]


def calculate_features(audio):
    if len(audio) < 256:
        raise ValueError("Audio segment is too short.")

    rms = np.sqrt(np.mean(audio ** 2)) # Root mean square amplitude
    peak = np.max(np.abs(audio)) # Highest amplitude in the segment
    zero_crossings = np.mean(np.signbit(audio[:-1]) != np.signbit(audio[1:])) # How often the signal crosses zero

    frequencies, power = welch(audio, fs=SAMPLE_RATE, nperseg=min(4096, len(audio))) # Estimate power at each frequency
    total_power = np.sum(power)

    if total_power <= 1e-20:
        return np.array([rms, peak, zero_crossings] + [0.0] * (len(FEATURE_NAMES) - 3))

    normalized_power = power / total_power # Normalize power distribution

    centroid = np.sum(frequencies * normalized_power) # Center of the frequency distribution
    bandwidth = np.sqrt(np.sum((frequencies - centroid) ** 2 * normalized_power)) # Spread of frequencies
    rolloff_index = min(np.searchsorted(np.cumsum(normalized_power), 0.85), len(frequencies) - 1)
    rolloff = frequencies[rolloff_index] # Frequency containing 85% of the signal's power

    flatness = np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20) # Helps distinguish tonal and broadband noise
    entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-20)) # Measures how spread out the power is
    peak_frequency = frequencies[np.argmax(power)] # Frequency with the highest power

    band_energies = []
    for low, high in FREQUENCY_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        band_energies.append(np.sum(normalized_power[mask])) # Relative power within each frequency band

    return np.array([
        rms, peak, zero_crossings, centroid, bandwidth, rolloff,
        flatness, entropy, peak_frequency, *band_energies
    ], dtype=np.float64)


def extract_features(file_path):
    feature_vectors = []
    weights = []

    with wave.open(str(file_path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Audio must be 48 kHz, mono, 16-bit PCM WAV.") # Matches Adam's filtering layer

        while True:
            frames = wav.readframes(SAMPLE_RATE * WINDOW_SECONDS) # Reads 10 seconds at a time
            if not frames:
                break

            audio = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0 # Convert PCM to normalized audio
            if len(audio) < 256:
                continue

            feature_vectors.append(calculate_features(audio))
            weights.append(len(audio))

    if not feature_vectors:
        raise ValueError("No valid audio data found.")

    return np.average(feature_vectors, axis=0, weights=weights) # Combine all windows into one feature vector
