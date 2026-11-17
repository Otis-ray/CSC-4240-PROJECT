import csv
import json
import wave
import joblib
import unittest
import tempfile
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from audio_features import extract_features, FEATURE_NAMES, SAMPLE_RATE, WINDOW_SECONDS
from prepare_dataset import prepare_dataset
from predict import predict_noise

class TestLayer2(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup() # Removes temporary testing files

    def make_wav(self, path, sample_rate=48000):
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(sample_rate // 5) / sample_rate
        audio = (np.sin(2 * np.pi * 440 * time) * 12000).astype("<i2") # Generates a 440 Hz test signal
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())
        return path

    def test_feature_extraction(self):
        path = self.make_wav(self.folder / "test.wav")
        features = extract_features(path)
        self.assertEqual(len(features), len(FEATURE_NAMES)) # Verifies all 15 features are extracted
        self.assertTrue(np.all(np.isfinite(features))) # Checks for invalid numerical values

    def test_invalid_wav(self):
        path = self.make_wav(self.folder / "invalid.wav", 44100)
        with self.assertRaises(ValueError):
            extract_features(path) # Rejects audio that is not 48 kHz

    def test_dataset_preparation(self):
        data = self.folder / "labeled"
        labels = ["noise_a", "noise_b", "noise_c", "noise_d"]
        for label in labels:
            for source in ["source_001", "source_002"]:
                self.make_wav(data / label / source / "test.wav")
        output = self.folder / "manifest.csv"
        prepare_dataset(data, output)
        with open(output, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(row["label"] for row in rows), set(labels))

    def test_prediction(self):
        path = self.make_wav(self.folder / "test.wav")
        features = extract_features(path)
        labels = ["noise_a", "noise_b", "noise_c", "noise_d"]
        model = DummyClassifier(strategy="prior") # Tests prediction output without training a real classifier
        model.fit(np.tile(features, (4, 1)), labels)
        model_path = self.folder / "test_model.joblib"
        joblib.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "classes": labels
        }, model_path)
        output = self.folder / "results.json"
        results = predict_noise(model_path, path, output)
        self.assertTrue(output.exists())
        self.assertIn(results["noise_type"], labels)
        self.assertEqual(len(results["class_scores"]), 4)
        with open(output, encoding="utf-8") as file:
            saved = json.load(file)
        self.assertEqual(saved, results) # Verifies saved JSON matches the prediction

if __name__ == "__main__":
    unittest.main()
