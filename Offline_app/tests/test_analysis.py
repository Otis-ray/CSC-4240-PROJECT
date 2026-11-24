import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from Offline_app.analysis import AnalysisError, run_analysis, validate_spec, wav_info


class SpecTests(unittest.TestCase):
    def test_hp_lp_bp_mapping(self):
        self.assertEqual(validate_spec({"type":"lowpass","family":"butter","order":4,"cutoff":5000})[1],
                         ["lowpass", "butter", "4", "1000", "5000", "4"])
        self.assertEqual(validate_spec({"type":"highpass","family":"butter","order":2,"cutoff":300})[1],
                         ["highpass", "butter", "2", "300", "1000", "5"])
        self.assertEqual(validate_spec({"type":"bandpass","family":"butter","order":8,"cutoff":[300,5000]})[1],
                         ["bandpass", "butter", "4", "300", "5000", "6"])

    def test_reject_invalid_spec(self):
        for spec in (
            {"type":"bandpass","order":4,"cutoff":[6000,300]},
            {"type":"highpass","order":8,"cutoff":500},
            {"type":"highpass","order":2,"cutoff":0},
            {"type":"highpass","order":2,"cutoff":float('nan')},
            {"type":"lowpass","order":1,"cutoff":24000},
            {"type":"lowpass","order":True,"cutoff":300},
            {"type":"lowpass","order":2,"cutoff":300,"family":"ellip"},
        ):
            with self.subTest(spec=spec):
                with self.assertRaises(AnalysisError):
                    validate_spec(spec)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / 'source.wav'
        with wave.open(str(self.input), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(b'\x00\x00' * 480)
        self.model = self.root / 'model.joblib'
        self.model.write_bytes(b'mock')
        self.binary = self.root / 'filter'
        self.binary.write_bytes(b'mock')
        self.output = self.root / 'run1'

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_and_manifest(self):
        prediction = {"file": str(self.input), "spec": {"type": "bandpass", "family": "butter", "order": 8, "cutoff": [300, 5000]}, "confidence": {"type": .8, "order": .9}}
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                return subprocess.CompletedProcess(command, 0, json.dumps(prediction) + '\n', '')
            shutil.copyfile(self.input, self.output / 'reconstructed.wav')
            return subprocess.CompletedProcess(command, 0, 'Automated task finished', '')
        with patch('Offline_app.analysis.subprocess.run', side_effect=fake_run):
            manifest_path = run_analysis(self.input, self.model, self.output, self.binary)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest['status'], 'success')
        self.assertEqual(commands[1][1:6], ['bandpass', 'butter', '4', '300', '5000'])
        self.assertEqual(commands[1][-1], '6')
        self.assertEqual(json.loads((self.output / 'layer1_prediction.json').read_text()), prediction)
        self.assertEqual(wav_info(self.output / 'reconstructed.wav')['frames'], 480)
        self.assertEqual(manifest['inverse_filter']['c_stage_order'], 4)
        with self.assertRaises(AnalysisError):
            run_analysis(self.input, self.model, self.output, self.binary)

    def test_detect_missing_c_output_despite_zero_exit(self):
        prediction = {"file": str(self.input), "spec": {"type": "highpass", "family": "butter", "order": 1, "cutoff": 300}}
        def fake_run(command, **kwargs):
            if command[1:3] == ['-m', 'Layer_1']:
                return subprocess.CompletedProcess(command, 0, json.dumps(prediction), '')
            return subprocess.CompletedProcess(command, 0, 'success!', '')
        with patch('Offline_app.analysis.subprocess.run', side_effect=fake_run):
            with self.assertRaises(AnalysisError):
                run_analysis(self.input, self.model, self.output, self.binary)
        self.assertEqual(json.loads((self.output / 'manifest.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
