#!/usr/bin/env python3
"""Standalone, supervised LF classifier for physical-noise Ardour dataset.
Does not import or modify Layer_1, Layer_2 or Offline_app.
"""
import argparse
import csv
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

LABELS = ('low_frequency', '60hz_hum', 'broadband')
FEATURE_VERSION = 'lf_fft_v1'


def feature_vector(audio, sr):
    """Fixed-size FFT and temporal features, calculated on LF-isolated audio."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        raise ValueError('Empty audio window')
    x = x - x.mean()
    # Welch is more robust than a single FFT to speaker and noise phase.
    freq, psd = welch(x, fs=sr, nperseg=min(len(x), 8192))
    valid = (freq >= 10) & (freq <= min(sr / 2, 1000))
    f = freq[valid]
    p = psd[valid]
    eps = 1e-20
    total = float(p.sum()) + eps
    bands = ((10, 35), (35, 55), (55, 65), (65, 90),
             (90, 115), (115, 160), (160, 220), (220, 300),
             (300, 400), (400, 600), (600, 1000))
    feat = [float(np.log10(np.mean(x*x) + eps)),
            float(np.log10(total)),
            float((f*p).sum()/total),
            float(np.sqrt((((f - (f*p).sum()/total)**2)*p).sum()/total)),
            float(np.exp(np.log(p+eps).mean())/(p.mean()+eps))]
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        feat.append(float(p[mask].sum()/total))
    # Harmonic/neighbor peak contrast around potential electrical hum.
    for center in (60, 120, 180, 240):
        tone = p[np.abs(f-center) <= 3].sum()
        neighbor = p[(np.abs(f-center) > 6) & (np.abs(f-center) <= 18)].sum()
        feat.extend((float(tone/total), float(np.log10((tone+eps)/(neighbor+eps)))))
    # Modulation signature: RMS per 100 ms, and changes in its envelope.
    frame_n = max(1, int(sr*0.1))
    chunks = len(x)//frame_n
    if chunks:
        envelope = np.sqrt(np.mean(x[:chunks*frame_n].reshape(chunks, frame_n)**2, axis=1))
        mu = envelope.mean()+1e-12
        feat += [float(envelope.std()/mu),
                 float(np.mean(np.abs(np.diff(envelope)))/mu) if chunks > 1 else 0.,
                 float(np.max(envelope)/mu)]
    else:
        feat += [0., 0., 0.]
    return np.nan_to_num(np.asarray(feat, dtype=float), nan=0., posinf=0., neginf=0.)


def run_c_filter(executable, source, destination, order, hp, lp, operation):
    # filter.c: argv[4] is HP; argv[5] is LP; 1=LP, 2=HP, 5=inv HP.
    cmd = [str(executable), 'HP' if operation in (2, 5) else 'LP',
           'butter', str(order), str(hp), str(lp),
           str(source), str(destination), str(operation)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode or not destination.is_file():
        raise RuntimeError(f'C filter failed: {" ".join(cmd)}\n{result.stdout}\n{result.stderr}')


def read_windows(path, length_s=5.):
    with sf.SoundFile(path) as f:
        if f.channels != 1:
            raise ValueError(f'Expected mono audio: {path}')
        frames = round(length_s*f.samplerate)
        index = 0
        while True:
            audio = f.read(frames, dtype='float32', always_2d=False)
            if len(audio) == 0:
                break
            # Exclude partial trailing windows: consistent positive/negative duration.
            if len(audio) != frames:
                break
            yield index*length_s, audio, f.samplerate
            index += 1


def load_events(csv_path):
    out = defaultdict(list)
    with open(csv_path, newline='') as stream:
        for row in csv.DictReader(stream):
            if str(row.get('region_muted','0')).strip() not in ('0','false','False',''):
                continue
            labels = set(row['noise_labels'].split(';')) & set(LABELS)
            if not labels:
                continue
            start = float(row['start_seconds']) - float(row.get('session_range_start_seconds') or 0)
            end = float(row['end_seconds']) - float(row.get('session_range_start_seconds') or 0)
            if end > start:
                out[row['session']].append((start, end, labels))
    return out


def label_window(events, start, window_s, positive_occupancy=.8, negative_occupancy=.2):
    """Return multi-label target or None for ambiguous boundary windows."""
    end = start + window_s
    scores = []
    for label in LABELS:
        intervals = sorted((max(start,s), min(end,e)) for s,e,names in events if label in names and min(end,e)>max(start,s))
        # Union overlapped label intervals, avoiding double-counting.
        covered = 0.
        if intervals:
            lo,hi = intervals[0]
            for s,e in intervals[1:]:
                if s <= hi: hi=max(hi,e)
                else: covered += hi-lo; lo,hi=s,e
            covered += hi-lo
        fraction = covered/window_s
        if negative_occupancy < fraction < positive_occupancy:
            return None
        scores.append(int(fraction >= positive_occupancy))
    return scores


def discover_speakers(data_root, events):
    # Only match exact <speaker>.wav; avoid reading unrelated exports.
    for speaker in sorted(events, key=lambda x: (int(x[7:]) if x.startswith('Speaker') and x[7:].isdigit() else x)):
        path = Path(data_root)/f'{speaker}.wav'
        if path.is_file():
            yield speaker, path
        else:
            print(f'[skip] Missing {path}')


def make_preprocessed(exe, input_path, base, order, cutoff, lp_order):
    base.mkdir(parents=True, exist_ok=True)
    forward = base/'forward.wav'
    inverse = base/'known_inverse.wav'
    isolated = base/'lf_isolated.wav'
    if not forward.exists():
        run_c_filter(exe, input_path, forward, order, cutoff, 1, 2)
    if not inverse.exists():
        run_c_filter(exe, forward, inverse, order, cutoff, 1, 5)
    if not isolated.exists():
        run_c_filter(exe, inverse, isolated, lp_order, 1, cutoff, 1)
    return isolated


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def train(args):
    events = load_events(args.events_csv)
    available = list(discover_speakers(args.data_root, events))
    if args.max_speakers:
        available = available[:args.max_speakers]
    if len(available) < 3:
        raise SystemExit('Need at least three labeled, available speaker recordings.')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = Path(args.filter_exe).resolve()
    if not exe.exists():
        raise SystemExit(f'Missing C executable: {exe}. Compile Filters/filter first.')
    X, Y, groups, ids = [], [], [], []
    for number,(speaker, wav) in enumerate(available,1):
        print(f'[{number}/{len(available)}] {speaker}', flush=True)
        # Temporary WAV processing avoids >20 GB disk usage by default.
        with tempfile.TemporaryDirectory(prefix='lf_preprocess_') as tmp:
            folder = out/'processed_audio'/speaker if args.keep_audio else Path(tmp)
            isolated = make_preprocessed(exe, wav.resolve(), folder,
                                        args.hp_order, args.cutoff, args.lp_order)
            for start, audio, sr in read_windows(isolated, args.window_seconds):
                label = label_window(events[speaker], start, args.window_seconds)
                if label is None:
                    continue
                X.append(feature_vector(audio, sr))
                Y.append(label)
                groups.append(speaker)
                ids.append({'speaker':speaker,'start_seconds':start, 'labels':label})
    if not X:
        raise SystemExit('No valid labeled windows; check timestamp offsets and dataset.')
    X, Y, groups = np.asarray(X), np.asarray(Y), np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_fraction, random_state=args.seed)
    tr,te = next(splitter.split(X,Y,groups))
    if not all(Y[tr,k].sum() and Y[te,k].sum() for k in range(len(LABELS))):
        print('[warning] At least one class lacks positive examples in train or test split')
    models=[]
    results={}
    for col,label in enumerate(LABELS):
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight='balanced_subsample',
                                       n_jobs=-1, random_state=args.seed)
        model.fit(X[tr],Y[tr,col])
        pred=model.predict(X[te])
        precision,recall,f1,_=precision_recall_fscore_support(Y[te,col],pred,
                    average='binary',zero_division=0)
        tn,fp,fn,tp = confusion_matrix(Y[te,col],pred,labels=[0,1]).ravel()
        results[label]={'precision':float(precision),'recall':float(recall),'f1':float(f1),
                        'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
        models.append(model)
    package={'models':models,'labels':LABELS,'feature_version':FEATURE_VERSION,
             'window_seconds':args.window_seconds,'hp_order':args.hp_order,
             'cutoff':args.cutoff,'lp_order':args.lp_order,
             'positive_occupancy':.8, 'negative_occupancy':.2}
    joblib.dump(package,out/'model.joblib')
    summary={'n_train_windows':len(tr),'n_test_windows':len(te),
             'n_train_speakers':len(set(groups[tr])),
             'n_test_speakers':len(set(groups[te])),
             'test_speakers':sorted(set(groups[te])),
             'metrics':results,
             'warning':'Ardour region timing must match WAV export offset; physically repeated noise sources are not independent realizations.'}
    write_json(out/'evaluation.json',summary)
    write_json(out/'heldout_windows.json',[dict(ids[i], predicted=[int(m.predict(X[i:i+1])[0]) for m in models]) for i in te])
    print(json.dumps(summary,indent=2))
    print(f'Written: {out / "model.joblib"}, evaluation.json, heldout_windows.json')


def predict(args):
    package = joblib.load(args.model)
    if package['feature_version'] != FEATURE_VERSION:
        raise SystemExit('Incompatible feature version')
    cutoff = args.lp_cutoff if args.lp_cutoff else package['cutoff']
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = Path(args.filter_exe).resolve()
    with tempfile.TemporaryDirectory(prefix='lf_predict_') as tmp:
        path = out/'lf_isolated.wav' if args.keep_audio else Path(tmp)/'lf_isolated.wav'
        run_c_filter(exe, Path(args.input).resolve(), path, package['lp_order'], 1, cutoff, 1)
        windows=[]
        for start,audio,sr in read_windows(path, package['window_seconds']):
            fv=feature_vector(audio,sr).reshape(1,-1)
            scores={}
            for label,model in zip(package['labels'],package['models']):
                cls=list(model.classes_)
                score=float(model.predict_proba(fv)[0,cls.index(1)]) if 1 in cls else 0.
                scores[label]=round(score,5)
            windows.append({'start_seconds':start,'end_seconds':start+package['window_seconds'],
                            'scores':scores,'detected':{k:v >= args.threshold for k,v in scores.items()}})
    write_json(out/'lf_prediction.json',{'input':str(Path(args.input).resolve()),
                   'cutoff_hz':cutoff, 'threshold':args.threshold,
                   'feature_version':FEATURE_VERSION,'windows':windows})
    print(f'Wrote {len(windows)} windows: {out / "lf_prediction.json"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    t=subs.add_parser('train',help='Prepare C-filtered windows, fit and evaluate RF')
    t.add_argument('--data-root',required=True,help='Directory with Speaker1.wav, etc')
    t.add_argument('--events-csv',required=True)
    t.add_argument('--filter-exe',default='Filters/filter')
    t.add_argument('--out',required=True)
    t.add_argument('--hp-order',type=int,choices=(1,2,4),default=2)
    t.add_argument('--cutoff',type=float,default=300.)
    t.add_argument('--lp-order',type=int,choices=(1,2,4),default=4)
    t.add_argument('--window-seconds',type=float,default=5.)
    t.add_argument('--test-fraction',type=float,default=.2)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--max-speakers',type=int,default=None)
    t.add_argument('--keep-audio',action='store_true')
    p=subs.add_parser('predict',help='Predict from an ALREADY reconstructed mono WAV')
    p.add_argument('--input',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--filter-exe',default='Filters/filter')
    p.add_argument('--lp-cutoff',type=float,help='Estimated HP cutoff in Hz')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--out',required=True)
    p.add_argument('--keep-audio',action='store_true')
    args=parser.parse_args()
    if args.command=='train':train(args)
    else:predict(args)


if __name__=='__main__':main()
