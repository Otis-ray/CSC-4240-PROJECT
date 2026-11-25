"""Orchestrate existing Layer 1/2 trainers and optional reusable 109-WAV DSP dataset."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .analysis import ROOT, DEFAULT_FILTER, wav_info, sha256

EXPECTED_DATASET_COUNT = 109
DEFAULT_SETTINGS = ROOT / 'foundational' / 'settings.json'
DEFAULT_EVENTS = ROOT / 'layer2_noise_events.json'
DEFAULT_LF_EVENTS = ROOT / 'layer2_noise_events.csv'
REALISTIC = {'highpass_cutoff_hz':[80.0,600.0], 'lowpass_cutoff_hz':[2000.0,8000.0],
             'bandpass_low_hz':[80.0,600.0], 'bandpass_high_hz':[2000.0,8000.0]}

class TrainingError(RuntimeError):
    pass


def resolve_ranges(base: dict, profile: str, custom: dict | None = None) -> dict:
    """Return a copy of Layer1 filters with constrained ranges; never modify model schema."""
    filters = copy.deepcopy(base)
    if profile == 'realistic':
        filters.update(copy.deepcopy(REALISTIC))
    elif profile == 'custom':
        required = {'highpass_cutoff_hz', 'lowpass_cutoff_hz', 'bandpass_low_hz', 'bandpass_high_hz'}
        if not isinstance(custom, dict) or set(custom) != required:
            raise TrainingError('Custom profile requires exactly four frequency ranges: ' + ', '.join(sorted(required)))
        filters.update(custom)
    elif profile != 'unrestricted':
        raise TrainingError('Profile must be realistic, unrestricted or custom')
    keys = ('highpass_cutoff_hz','lowpass_cutoff_hz','bandpass_low_hz','bandpass_high_hz')
    for name in keys:
        bounds = filters.get(name)
        if not isinstance(bounds, (list,tuple)) or len(bounds)!=2:
            raise TrainingError(f'{name}: expected [minimum, maximum]')
        a,b = bounds
        if not all(isinstance(x, (float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in bounds) or not 0 < a < b < 24000:
            raise TrainingError(f'{name}: invalid positive ordered 48kHz cutoff bounds: {bounds}')
    if filters['bandpass_low_hz'][1] >= filters['bandpass_high_hz'][0]:
        raise TrainingError('Band-pass HP maximum must be less than LP minimum')
    return filters


def _run(command: list[str], *, timeout: int | None = None) -> None:
    print('Executing: ' + ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False, timeout=timeout)
    if result.returncode:
        raise TrainingError(f'Command failed with exit code {result.returncode}: {command}')


def _validate_sources(audio_dir: Path, *, require_complete: bool) -> list[Path]:
    paths = sorted(audio_dir.glob('*.wav'))
    if not paths:
        raise TrainingError(f'No WAV files found in {audio_dir}')
    if require_complete:
        wanted = {f'Speaker{i}' for i in range(1, EXPECTED_DATASET_COUNT+1)}
        actual = {p.stem for p in paths}
        if actual != wanted:
            raise TrainingError(f'Full dataset needs Speaker1..Speaker109 WAVs: missing {sorted(wanted-actual)}, unexpected {sorted(actual-wanted)}')
    for p in paths:
        wav_info(p)
    return paths


def _apply_c(binary: Path, source: Path, target: Path, spec: dict, *, reverse: bool, work: Path, timeout: int) -> list[list[str]]:
    kind = spec['type']; order = spec['order']; cutoff = spec['cutoff']
    if kind=='bandpass':
        # Use separate stages; do not use C's shared int.wav intermediates.
        stage=order//2
        if reverse:
            operations=[('lowpass', float(cutoff[1]), 4), ('highpass', float(cutoff[0]), 5)]
        else:
            operations=[('highpass', float(cutoff[0]), 2), ('lowpass', float(cutoff[1]), 1)]
    else:
        stage=order
        operations=[(kind, float(cutoff), (5 if kind=='highpass' else 4) if reverse else (2 if kind=='highpass' else 1))]
    seen=[]; previous=source
    for index,(typ,cut,choice) in enumerate(operations):
        destination=target if index==len(operations)-1 else work / f'intermediate_{index}.wav'
        hp=cut if typ=='highpass' else 1000.0
        lp=cut if typ=='lowpass' else 1000.0
        cmd=[str(binary),typ,'butter',str(stage),f'{hp:.12g}',f'{lp:.12g}',str(previous),str(destination),str(choice)]
        result=subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False, timeout=timeout)
        if result.returncode or not destination.is_file():
            raise TrainingError(f'C filter failed: {cmd}\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}')
        if wav_info(destination)['frames'] != wav_info(previous)['frames']:
            raise TrainingError(f'C filter altered WAV frame count: {destination}')
        seen.append(cmd); previous=destination
    return seen


def generate_full_dataset(audio_dir: Path, output: Path, filters: dict, seed: int, binary: Path, timeout: int = 1800) -> Path:
    """Save only full-length filtered and recovered WAVs; never training clips."""
    sources=_validate_sources(audio_dir, require_complete=True)
    if not binary.is_file():
        raise TrainingError(f'Missing C filter executable: {binary}')
    output.mkdir(parents=True, exist_ok=False)
    forward=output/'forward'; recovered=output/'recovered'
    forward.mkdir(); recovered.mkdir()
    manifest={'schema_version':1,'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'source_dir':str(audio_dir),'source_count':len(sources),'seed':seed,
              'reconstruction_parameters':'ground_truth_forward_parameters (not Layer 1 predictions)',
              'filter_ranges':filters,'entries':[]}
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    save()
    from Layer_1.dataset import sample_filter
    rng=np.random.default_rng(seed)
    try:
        for idx,p in enumerate(sources,1):
            spec=sample_filter(filters,rng)
            out_f=forward/p.name; out_r=recovered/p.name
            with tempfile.TemporaryDirectory(prefix='full_filter_') as temporary:
                t=Path(temporary)
                forward_commands=_apply_c(binary,p,out_f,spec,reverse=False,work=t,timeout=timeout)
                reverse_commands=_apply_c(binary,out_f,out_r,spec,reverse=True,work=t,timeout=timeout)
            entry={'source':str(p),'source_sha256':sha256(p),'spec':spec,
                   'forward':str(out_f.relative_to(output)),'forward_sha256':sha256(out_f),
                   'recovered':str(out_r.relative_to(output)),'recovered_sha256':sha256(out_r),
                   'duration_seconds':wav_info(p)['duration_seconds'],
                   'forward_commands':forward_commands,'reverse_commands':reverse_commands}
            manifest['entries'].append(entry); save()
            print(f'Full WAV pair {idx}/{len(sources)}: {p.name} ({spec["type"]})',flush=True)
        manifest['status']='complete'; save()
    except Exception as exc:
        manifest['status']='incomplete'; manifest['error']=str(exc); save()
        raise
    return output/'manifest.json'


def train_models(*, audio_dir: str|Path, out: str|Path, profile: str='realistic',
                 clip_seconds: float|None=60, examples_per_source: int=10,
                 events: str|Path=DEFAULT_EVENTS, train_layer1: bool=True,
                 train_layer2: bool=True, save_full_dataset: bool=False,
                 train_lf: bool=False, lf_mode: str='standard',
                 lf_events: str|Path=DEFAULT_LF_EVENTS, lf_window_seconds: float=5.0,
                 lf_trees: int=200, lf_max_speakers: int=0,
                 custom_ranges:dict|None=None, filter_bin: str|Path=DEFAULT_FILTER,
                 timeout: int=1800, layer1_settings: str|Path=DEFAULT_SETTINGS) -> Path:
    audio=Path(audio_dir).expanduser().resolve(); dest=Path(out).expanduser().resolve()
    if dest.exists():
        raise TrainingError(f'Output directory exists; refusing overwrite: {dest}')
    if not audio.is_dir():
        raise TrainingError(f'Dataset folder not found: {audio}')
    if not (train_layer1 or train_layer2 or train_lf or save_full_dataset):
        raise TrainingError('At least one training or export option must be selected')
    if not isinstance(examples_per_source,int) or examples_per_source<1:
        raise TrainingError('examples_per_source must be positive integer')
    if clip_seconds is not None and (not math.isfinite(clip_seconds) or clip_seconds<=0):
        raise TrainingError('clip_seconds must be positive or null')
    if lf_mode not in ('standard','generalized'):
        raise TrainingError('lf_mode must be standard or generalized')
    if not math.isfinite(lf_window_seconds) or lf_window_seconds <= 0:
        raise TrainingError('lf_window_seconds must be positive')
    if lf_trees < 1 or lf_max_speakers < 0:
        raise TrainingError('lf_trees must be positive and lf_max_speakers nonnegative')
    if timeout<1:
        raise TrainingError('timeout must be positive')
    base=json.loads(Path(layer1_settings).expanduser().resolve().read_text())
    base['source_dir']=str(audio); base['clip_seconds']=clip_seconds; base['examples_per_source']=examples_per_source
    base['bits']=16; base['sample_rate']=48000
    base['filters']=resolve_ranges(base['filters'],profile,custom_ranges)
    binary=Path(filter_bin).expanduser().resolve()
    if train_layer1 or train_lf or save_full_dataset:
        if not binary.is_file():
            raise TrainingError(f'Missing C filter executable: {binary}')
        _validate_sources(audio,require_complete=save_full_dataset)
    eventfile=Path(events).expanduser().resolve()
    if train_layer2 and not eventfile.is_file():
        raise TrainingError(f'Noise event manifest missing: {eventfile}')
    lf_eventfile=Path(lf_events).expanduser().resolve()
    if train_lf and not lf_eventfile.is_file():
        raise TrainingError(f'LF noise-event CSV missing: {lf_eventfile}')
    dest.mkdir(parents=True)
    manifest={'schema_version':1,'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'dataset':str(audio),'profile':profile,'filters':base['filters'],
              'clip_seconds':clip_seconds,'examples_per_source':examples_per_source,
              'layer1':{'status':'pending' if train_layer1 else 'skipped'},
              'layer2':{'status':'pending' if train_layer2 else 'skipped'},
              'lf':{'status':'pending' if train_lf else 'skipped'},
              'full_dataset':{'status':'pending' if save_full_dataset else 'skipped'}}
    def save(): (dest/'training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        settingsfile=dest/'layer1_settings.json'
        settingsfile.write_text(json.dumps(base,indent=2)+'\n')
        if train_layer1:
            # Layer1 CLI refuses to overwrite its model output, as expected.
            _run([sys.executable,'-m','Layer_1','train','--settings',str(settingsfile),'--out',str(dest/'layer1')])
            manifest['layer1']={'status':'complete','model':'layer1/model.joblib','metrics':'layer1/metrics.json'};save()
        if train_layer2:
            index=dest/'layer2_audio_index.json'
            _run([sys.executable,'-m','Layer_2.prepare_dataset','--events',str(eventfile), '--audio-dir',str(audio),'--output',str(index),'--strict'])
            l2model=dest/'layer2'/'noise_multilabel.joblib'
            _run([sys.executable,'-m','Layer_2.training_model','--manifest',str(index),'--output',str(l2model)])
            manifest['layer2']={'status':'complete','model':'layer2/noise_multilabel.joblib','metrics':'layer2/noise_multilabel_metrics.json','audio_index':'layer2_audio_index.json'};save()
        if train_lf:
            # These scripts train on event-labelled recordings with the same
            # dataset as Layer 2. Their own speaker-level split is retained.
            lf_out=dest/'lf'
            if lf_mode=='standard':
                cmd=[sys.executable,str(ROOT/'LF_experiment'/'lf_experiment.py'),'train',
                     '--data-root',str(audio),'--events-csv',str(lf_eventfile),
                     '--filter-exe',str(binary),'--out',str(lf_out),
                     '--hp-order','2','--cutoff','300','--lp-order','4',
                     '--window-seconds',str(lf_window_seconds)]
                if lf_max_speakers:
                    cmd.extend(['--max-speakers',str(lf_max_speakers)])
            else:
                cmd=[sys.executable,str(ROOT/'LF_experiment'/'lf_generalized.py'),
                     '--data-root',str(audio),'--events-csv',str(lf_eventfile),
                     '--filter-exe',str(binary),'--out',str(lf_out),
                     '--window-seconds',str(lf_window_seconds),
                     '--trees',str(lf_trees),'--preset','focused']
                if lf_max_speakers:
                    cmd.extend(['--max-speakers',str(lf_max_speakers)])
            _run(cmd)
            if not (lf_out/'model.joblib').is_file() or not (lf_out/'evaluation.json').is_file():
                raise TrainingError('LF trainer exited without model.joblib and evaluation.json')
            manifest['lf']={'status':'complete','mode':lf_mode,
                            'model':'lf/model.joblib','metrics':'lf/evaluation.json',
                            'events_csv':str(lf_eventfile),
                            'window_seconds':lf_window_seconds}
            save()
        if save_full_dataset:
            path=generate_full_dataset(audio,dest/'full_length_dataset',base['filters'],int(base['seed']),binary,timeout=timeout)
            manifest['full_dataset']={'status':'complete','manifest':str(path.relative_to(dest))};save()
        manifest['status']='complete';save()
    except Exception as exc:
        manifest['status']='incomplete';manifest['error']=str(exc);save()
        raise
    return dest/'training_manifest.json'
