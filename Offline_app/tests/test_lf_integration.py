import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from Offline_app.analysis import AnalysisError, run_analysis, lf_events_from_windows


class LFIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input.wav'
        with wave.open(str(self.source), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
            wav.writeframes(b'\0\0' * 480)
        self.l1 = self.root/'l1.joblib'; self.l2 = self.root/'l2.joblib'
        self.lf = self.root/'lf.joblib'; self.bin = self.root/'filter'
        self.script = self.root/'lf_experiment.py'
        for path in (self.l1, self.l2, self.lf, self.bin, self.script):
            path.write_text('mock')
        self.commands = []
        self.kind = 'highpass'
        self.bad_lf = False

    def tearDown(self):
        self.tmp.cleanup()

    def fake_run(self, cmd, **kwargs):
        self.commands.append(cmd)
        if cmd[1:3] == ['-m', 'Layer_1']:
            spec = {'type': self.kind, 'family': 'butter', 'order': 2,
                    'cutoff': [280, 5000] if self.kind == 'bandpass' else 300 if self.kind == 'highpass' else 5000}
            return subprocess.CompletedProcess(cmd, 0, json.dumps({'file': str(self.source), 'spec': spec}), '')
        if cmd[1:3] == ['-m', 'Layer_2.predict']:
            out=Path(cmd[cmd.index('--output')+1]); inp=cmd[cmd.index('--input')+1]
            out.write_text(json.dumps({'schema_version':2,'file':inp,'noise_types':{},'events':[],
                                       'windows':[{}],'threshold':0.5}))
            return subprocess.CompletedProcess(cmd, 0, '', '')
        if 'predict' in cmd and str(self.script) in cmd:
            if self.bad_lf:
                return subprocess.CompletedProcess(cmd, 2, '', 'LF error')
            out = Path(cmd[cmd.index('--out')+1]); out.mkdir(exist_ok=True)
            cutoff = float(cmd[cmd.index('--lp-cutoff')+1])
            (out/'lf_isolated.wav').write_bytes(b'test')
            (out/'lf_prediction.json').write_text(json.dumps({
                'input':cmd[cmd.index('--input')+1], 'cutoff_hz':cutoff,
                'windows':[{'start_seconds':0,'end_seconds':5,'scores':{'low_frequency':0.91,'60hz_hum':0.01},'detected':{'low_frequency':True,'60hz_hum':False}}]}))
            return subprocess.CompletedProcess(cmd, 0, '', '')
        # C inverse
        shutil.copyfile(self.source, Path(cmd[-2]))
        return subprocess.CompletedProcess(cmd, 0, '', '')

    def test_two_branches(self):
        out=self.root/'out'
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            p=run_analysis(self.source,self.l1,out,self.bin,layer2_model=self.l2,
                           lf_model=self.lf,lf_script=self.script)
        manifest=json.loads(p.read_text()); report=json.loads((out/'report.json').read_text())
        self.assertEqual(manifest['status'],'success')
        self.assertEqual(manifest['lf']['isolation_cutoff_hz'],300)
        self.assertEqual(manifest['lf']['positive_windows']['low_frequency'],1)
        self.assertEqual(report['lf']['prediction_file'],'lf/lf_prediction.json')
        self.assertEqual(report['lf']['event_count'],1)
        self.assertEqual(report['lf']['events'][0]['start_seconds'],0.0)
        self.assertEqual(report['lf']['events'][0]['end_seconds'],5.0)
        self.assertEqual(report['lf']['events'][0]['mean_confidence'],0.91)
        self.assertIn('layer2_prediction.json',str(report['layer2']))
        self.assertTrue((out/'lf/lf_isolated.wav').is_file())
        self.assertTrue((out/'layer2_prediction.json').is_file())
        self.assertEqual(len(self.commands),4)

    def test_merge_adjacent_windows_and_score(self):
        windows = [
            {'start_seconds':0,'end_seconds':5,'scores':{'low_frequency':0.7,'60hz_hum':0.1},
             'detected':{'low_frequency':True,'60hz_hum':False}},
            {'start_seconds':5,'end_seconds':10,'scores':{'low_frequency':0.9,'60hz_hum':0.6},
             'detected':{'low_frequency':True,'60hz_hum':True}},
            {'start_seconds':10,'end_seconds':15,'scores':{'low_frequency':0.2,'60hz_hum':0.7},
             'detected':{'low_frequency':False,'60hz_hum':True}},
            {'start_seconds':20,'end_seconds':25,'scores':{'low_frequency':0.8,'60hz_hum':0.1},
             'detected':{'low_frequency':True,'60hz_hum':False}},
        ]
        events = lf_events_from_windows(windows, ['low_frequency','60hz_hum'])
        self.assertEqual([(e['type'], e['start_seconds'], e['end_seconds']) for e in events],
            [('low_frequency',0,10),('60hz_hum',5,15),('low_frequency',20,25)])
        self.assertEqual(events[0]['mean_confidence'],0.8)
        self.assertEqual(events[0]['max_confidence'],0.9)
        self.assertEqual(events[0]['window_count'],2)
        self.assertEqual(events[1]['mean_confidence'],0.65)

    def test_no_lf_events(self):
        windows = [{'start_seconds':0,'end_seconds':5,
                    'scores':{'low_frequency':0.1},'detected':{'low_frequency':False}}]
        self.assertEqual(lf_events_from_windows(windows,['low_frequency']),[])

    def test_missing_score_rejected(self):
        windows = [{'start_seconds':0,'end_seconds':5,
                    'scores':{},'detected':{'low_frequency':True}}]
        with self.assertRaisesRegex(AnalysisError,'missing valid timing/score'):
            lf_events_from_windows(windows,['low_frequency'])

    def test_manual_cutoff_overrides_layer1(self):
        self.kind='bandpass'
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            manifest_path=run_analysis(self.source,self.l1,self.root/'manual',self.bin,
                      lf_model=self.lf,lf_script=self.script,lf_cutoff=400)
        data=json.loads(manifest_path.read_text())
        self.assertEqual(data['lf']['isolation_cutoff_hz'],400)
        self.assertEqual(data['lf']['predicted_hp_cutoff_hz'],280)
        self.assertEqual(data['lf']['isolation_cutoff_source'],'manual')
        self.assertEqual(self.commands[-1][self.commands[-1].index('--lp-cutoff')+1], '400')
        report=json.loads((self.root/'manual/report.json').read_text())
        self.assertEqual(report['lf']['isolation_cutoff_source'],'manual')

    def test_automatic_cutoff_records_source(self):
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            path=run_analysis(self.source,self.l1,self.root/'auto',self.bin,
                              lf_model=self.lf,lf_script=self.script)
        self.assertEqual(json.loads(path.read_text())['lf']['isolation_cutoff_source'],'layer1')

    def test_invalid_cutoff_rejected_before_writing(self):
        for cutoff in (-3, 0, float('nan'), 24000):
            with self.subTest(cutoff=cutoff):
                with self.assertRaisesRegex(AnalysisError,'--lf-cutoff'):
                    run_analysis(self.source,self.l1,self.root/'invalid',self.bin,
                                 lf_model=self.lf,lf_script=self.script,lf_cutoff=cutoff)
        self.assertFalse((self.root/'invalid').exists())

    def test_bandpass_lower_edge(self):
        self.kind='bandpass'
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            manifest=run_analysis(self.source,self.l1,self.root/'bp',self.bin,lf_model=self.lf,lf_script=self.script)
        self.assertEqual(json.loads(manifest.read_text())['lf']['isolation_cutoff_hz'],280)

    def test_lowpass_skips_lf(self):
        self.kind='lowpass'
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            manifest=run_analysis(self.source,self.l1,self.root/'lp',self.bin,lf_model=self.lf,lf_script=self.script)
        data=json.loads(manifest.read_text())
        self.assertEqual(data['status'],'success')
        self.assertEqual(data['lf']['status'],'skipped')
        self.assertEqual(len(self.commands),2)

    def test_lf_failure_is_logged(self):
        self.bad_lf=True
        out=self.root/'bad'
        with patch('Offline_app.analysis.subprocess.run',side_effect=self.fake_run):
            with self.assertRaisesRegex(AnalysisError,'LF predictor failed'):
                run_analysis(self.source,self.l1,out,self.bin,lf_model=self.lf,lf_script=self.script)
        self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'failed')

    def test_direct_mode_disallows_lf(self):
        with self.assertRaisesRegex(AnalysisError,'requires reconstruction'):
            run_analysis(self.source,self.l1,self.root/'direct',self.bin,lf_model=self.lf,
                         lf_script=self.script,skip_inverse=True,layer2_model=self.l2)

if __name__=='__main__': unittest.main()
