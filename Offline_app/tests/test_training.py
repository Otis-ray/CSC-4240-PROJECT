import json
from pathlib import Path
import tempfile
import subprocess
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

    def test_lf_only_standard_uses_labeled_hp300_pipeline(self):
        self._run_lf_only('standard')

    def test_lf_only_generalized_uses_multiple_variants(self):
        self._run_lf_only('generalized')

    def _run_lf_only(self, mode):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'audio'; source.mkdir()
            with wave.open(str(source/'Speaker1.wav'),'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
                wav.writeframes(b'\0\0'*480)
            binary=root/'filter'; binary.write_text('stub')
            events=root/'events.csv'; events.write_text('speaker,start,end\n')
            commands=[]
            def fake_run(command, *, timeout=None):
                commands.append(command)
                model_dir=Path(command[command.index('--out')+1])
                model_dir.mkdir(parents=True)
                (model_dir/'model.joblib').write_text('stub model')
                (model_dir/'evaluation.json').write_text('{}')
            out=root/'train'
            with patch('Offline_app.training._run',side_effect=fake_run):
                result=train_models(audio_dir=source,out=out,train_layer1=False,
                                    train_layer2=False,train_lf=True,lf_mode=mode,
                                    lf_events=events,filter_bin=binary)
            manifest=json.loads(result.read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['lf']['status'],'complete')
            self.assertEqual(manifest['lf']['mode'],mode)
            self.assertEqual(len(commands),1)
            cmd=commands[0]
            self.assertEqual(cmd[cmd.index('--events-csv')+1],str(events.resolve()))
            if mode=='standard':
                self.assertIn('lf_experiment.py',cmd[1]); self.assertIn('train',cmd)
                self.assertEqual(cmd[cmd.index('--cutoff')+1],'300')
            else:
                self.assertIn('lf_generalized.py',cmd[1]);self.assertIn('--preset',cmd)

    def test_lf_only_requires_events(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'audio';source.mkdir()
            with wave.open(str(source/'Speaker1.wav'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000)
                wav.writeframes(b'\0\0'*480)
            binary=root/'filter';binary.write_text('stub')
            with self.assertRaisesRegex(TrainingError,'LF noise-event CSV missing'):
                train_models(audio_dir=source,out=root/'train',train_layer1=False,
                             train_layer2=False,train_lf=True,filter_bin=binary,
                             lf_events=root/'missing.csv')

    def test_does_not_overwrite_existing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TrainingError):
                train_models(audio_dir=d,out=d,train_layer1=False,train_layer2=False,save_full_dataset=True)


if __name__=='__main__': unittest.main()
