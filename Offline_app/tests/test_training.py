import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from Offline_app.training import (TrainingError, REALISTIC, resolve_ranges,
                                  _validate_sources, _apply_c, train_models)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.base={"types":["highpass","lowpass","bandpass"],
                   "orders":{"highpass":[1,2,4],"lowpass":[1,2,4],"bandpass":[2,4,8]},
                   "highpass_cutoff_hz":[80,1000],"lowpass_cutoff_hz":[2000,16000],
                   "bandpass_low_hz":[80,1000],"bandpass_high_hz":[2000,16000]}

    def test_realistic_and_unrestricted(self):
        practical=resolve_ranges(self.base,'realistic')
        self.assertEqual(practical['highpass_cutoff_hz'],REALISTIC['highpass_cutoff_hz'])
        self.assertEqual(practical['lowpass_cutoff_hz'],[2000,8000])
        self.assertEqual(self.base['lowpass_cutoff_hz'],[2000,16000])
        self.assertEqual(resolve_ranges(self.base,'unrestricted')['lowpass_cutoff_hz'],[2000,16000])

    def test_invalid_custom_bp_overlap(self):
        bad={name:list(bounds) for name,bounds in REALISTIC.items()}
        bad['bandpass_low_hz']=[80,2600]
        with self.assertRaises(TrainingError):resolve_ranges(self.base,'custom',bad)

    def test_full_dataset_needs_109(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TrainingError): _validate_sources(Path(d),require_complete=True)
            with wave.open(str(Path(d)/'Speaker1.wav'),'wb') as out:
                out.setnchannels(1);out.setsampwidth(2);out.setframerate(48000);out.writeframes(b'\0\0'*10)
            with self.assertRaises(TrainingError):_validate_sources(Path(d),require_complete=True)
            self.assertEqual(len(_validate_sources(Path(d),require_complete=False)),1)

    def test_does_not_overwrite_existing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TrainingError):
                train_models(audio_dir=d,out=d,train_layer1=False,train_layer2=False,save_full_dataset=True)


if __name__=='__main__': unittest.main()
