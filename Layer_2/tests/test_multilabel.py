"""Integration tests use artificial *test fixtures*, never project training data."""
import json
import tempfile
import unittest
import wave
from pathlib import Path

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from Layer_2.audio_features import FEATURE_NAMES, SAMPLE_RATE, iter_audio_windows
from Layer_2.prepare_dataset import CLASSES, prepare_dataset
from Layer_2.training_model import label_coverage, window_label
from Layer_2.predict import predict_noise, consolidate_events


def write_fixture(path, seconds=11):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(SAMPLE_RATE)
        samples = np.zeros(int(SAMPLE_RATE * seconds), dtype='<i2')
        wav.writeframes(samples.tobytes())


class TestMultiLabel(unittest.TestCase):
    def test_combined_labels_and_boundaries(self):
        events = [{'labels': ['low_frequency', 'high_frequency'], 'start_seconds': 0, 'end_seconds': 4},
                  {'labels': ['broadband'], 'start_seconds': 4, 'end_seconds': 9}]
        cov = label_coverage(events, 0, 5)
        self.assertAlmostEqual(cov[0], .8)
        self.assertAlmostEqual(cov[1], .8)
        self.assertIsNone(window_label(cov))  # broadband occupies ambiguous 20% of this window
        self.assertEqual(window_label(label_coverage(events, 5, 9)).tolist(), [0,0,1,0])
        self.assertEqual(window_label(label_coverage(events, 10, 15)).tolist(), [0,0,0,0])

    def test_index_and_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(1, 3):
                write_fixture(root / 'audio' / f'Speaker{n}.wav')
            entries = {'sessions': {f'Speaker{n}': {'events': [
                {'start_seconds': 0, 'end_seconds': 5,
                 'labels': ['low_frequency', 'high_frequency'], 'region': 'lf+hf_0s'}]}
                for n in range(1, 3)}}
            events_path = root / 'events.json'
            events_path.write_text(json.dumps(entries))
            output = root / 'index.json'
            idx = prepare_dataset(events_path, root / 'audio', output, strict=True)
            self.assertEqual(len(idx['sessions']), 2)
            windows = list(iter_audio_windows(root / 'audio/Speaker1.wav', window_seconds=5))
            self.assertEqual(len(windows), 2)
            self.assertEqual(len(windows[0][2]), len(FEATURE_NAMES))

    def test_prediction_multilabel_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / 'single.wav'
            write_fixture(wav)
            X = np.array([np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))]*3)
            y = np.array([[0,0,0,0], [1,1,0,0]]*3)
            model = MultiOutputClassifier(RandomForestClassifier(n_estimators=8, random_state=0))
            model.fit(X, y)
            artifact = root / 'model.joblib'
            joblib.dump({'schema_version': 2, 'model': model, 'feature_names': FEATURE_NAMES,
                         'sample_rate': SAMPLE_RATE, 'window_seconds': 5,
                         'classes': list(CLASSES), 'threshold': 0.5}, artifact)
            result = predict_noise(artifact, wav, root / 'out.json')
            self.assertEqual(len(result['windows']), 3)  # includes final 1s
            self.assertEqual(set(result['noise_types']), set(CLASSES))
            self.assertEqual(result['windows'][-1]['end_seconds'], 11)
            self.assertEqual(json.loads((root / 'out.json').read_text()), result)
            self.assertEqual(len(consolidate_events([
                {'start_seconds': 0, 'end_seconds': 5, 'detected': {'low_frequency': True}, 'scores': {'low_frequency': .8}},
                {'start_seconds': 5, 'end_seconds': 10, 'detected': {'low_frequency': True}, 'scores': {'low_frequency': .9}},
            ], labels=['low_frequency'])), 1)


if __name__ == '__main__':
    unittest.main()
