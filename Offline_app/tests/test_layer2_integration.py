import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import shutil

from Offline_app.analysis import run_analysis, AnalysisError


class Layer2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.wav = self.root / 'sample.wav'
        with wave.open(str(self.wav), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(b'\0\0' * 480)
        self.l1 = self.root / 'l1.joblib'
        self.l2 = self.root / 'l2.joblib'
        self.filter = self.root / 'filter'
        for f in [self.l1, self.l2, self.filter]:
            f.write_text('mock')
        self.output = self.root / 'output'
        self.commands = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_process(self, command, **kwargs):
        self.commands.append(command)
        if '-m' in command and 'Layer_1' in command:
            answer = {'file': str(self.wav), 'spec': {'type': 'highpass', 'family': 'butter', 'order': 2, 'cutoff': 300}, 'confidence': {'type': .9}}
            return subprocess.CompletedProcess(command, 0, json.dumps(answer), '')
        if '-m' in command and 'Layer_2.predict' in command:
            output = Path(command[command.index('--output')+1])
            inp = command[command.index('--input')+1]
            answer = {'schema_version': 2, 'file': inp, 'threshold': .5, 'window_seconds': 5.0,
                      'noise_types': {'low_frequency': {'detected': True}},
                      'windows': [{'start_seconds': 0, 'end_seconds': .01}],
                      'events': [{'type': 'low_frequency', 'start_seconds': 0, 'end_seconds': .01}]}
            output.write_text(json.dumps(answer))
            return subprocess.CompletedProcess(command, 0, 'classified', '')
        shutil.copyfile(self.wav, self.output/'reconstructed.wav')
        return subprocess.CompletedProcess(command, 0, 'filtered', '')

    def test_integrated(self):
        with patch('Offline_app.analysis.subprocess.run', side_effect=self.fake_process):
            path = run_analysis(self.wav, self.l1, self.output, self.filter, layer2_model=self.l2)
        data = json.loads(path.read_text())
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['layer2']['event_count'], 1)
        self.assertEqual(len(self.commands), 3)
        self.assertEqual(self.commands[2][self.commands[2].index('--input')+1], str(self.output/'reconstructed.wav'))
        report = json.loads((self.output/'report.json').read_text())
        self.assertEqual(report['layer2']['events'][0]['type'], 'low_frequency')
        self.assertTrue((self.output/'layer2_prediction.json').exists())

    def test_skip_inverse(self):
        with patch('Offline_app.analysis.subprocess.run', side_effect=self.fake_process):
            path = run_analysis(self.wav, self.l1, self.output, self.filter, layer2_model=self.l2, skip_inverse=True)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.commands[0][self.commands[0].index('--input')+1], str(self.wav))
        self.assertFalse((self.output/'reconstructed.wav').exists())
        self.assertEqual(json.loads(path.read_text())['mode'], 'direct_layer2')

    def test_missing_layer2_output_fails(self):
        def no_output(command, **kwargs):
            if '-m' in command and 'Layer_2.predict' in command:
                return subprocess.CompletedProcess(command, 0, 'success', '')
            return self.fake_process(command, **kwargs)
        with patch('Offline_app.analysis.subprocess.run', side_effect=no_output):
            with self.assertRaisesRegex(AnalysisError, 'did not save'):
                run_analysis(self.wav, self.l1, self.output, self.filter, layer2_model=self.l2)
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['status'], 'failed')

    def test_missing_model_prevents_directory_creation(self):
        with self.assertRaises(AnalysisError):
            run_analysis(self.wav, self.l1, self.output, self.filter, layer2_model=self.root/'missing.joblib')
        self.assertFalse(self.output.exists())

if __name__ == '__main__':
    unittest.main()
