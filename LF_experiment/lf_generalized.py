#!/usr/bin/env python3
"""Standalone generalized LF Random Forest training with persistent feature cache.

Works alongside LF_experiment/lf_experiment.py; does not replace it.
Training and evaluation speakers are split BEFORE augmenting filter conditions.
"""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

import lf_experiment as base

LABELS = ('low_frequency', '60hz_hum')
CACHE_VERSION = 'generalized_lf_cache_v1'
# Name, forward order, forward cutoff, inverse cutoff, isolation LP cutoff.
DEFAULT_VARIANTS = [
    ('o1_fc200_exact', 1, 200, 200, 200),
    ('o1_fc300_exact', 1, 300, 300, 300),
    ('o1_fc400_exact', 1, 400, 400, 400),
    ('o2_fc200_exact', 2, 200, 200, 200),
    ('o2_fc300_exact', 2, 300, 300, 300),
    ('o2_fc400_exact', 2, 400, 400, 400),
    ('o1_fc300_inv335', 1, 300, 335, 335),
    ('o2_fc300_inv335', 2, 300, 335, 335),
]

FULL_VARIANTS = DEFAULT_VARIANTS[:6] + [
    ('o1_fc150_exact', 1, 150, 150, 150),
    ('o2_fc150_exact', 2, 150, 150, 150),
    ('o1_fc600_exact', 1, 600, 600, 600),
    ('o2_fc600_exact', 2, 600, 600, 600),
] + DEFAULT_VARIANTS[6:]


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def file_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def config_signature(wav, exe, events_csv, variant, window_seconds, lp_order):
    payload = {
        'schema': CACHE_VERSION,
        'feature_version': base.FEATURE_VERSION,
        'wav': file_fingerprint(wav),
        'executable': file_fingerprint(exe),
        'events_csv_sha256': hashlib.sha256(Path(events_csv).read_bytes()).hexdigest(),
        'variant': variant, 'window_seconds': window_seconds, 'lp_order': lp_order,
        'positive_occupancy': .8, 'negative_occupancy': .2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def custom_variants(args):
    if not args.variant:
        return FULL_VARIANTS if args.preset == 'full' else DEFAULT_VARIANTS
    variants = []
    for raw in args.variant:
        fields = raw.split(':')
        if len(fields) != 5:
            raise SystemExit('Variant must be NAME:ORDER:FORWARD_HP:INVERSE_HP:LP')
        name = fields[0]
        if not name.replace('_', '').isalnum():
            raise SystemExit('Variant NAME must be alphanumeric or underscores')
        order = int(fields[1]); fwd, inv, lp = map(float, fields[2:])
        if order not in (1, 2) or min(fwd, inv, lp) <= 0:
            raise SystemExit('Only order 1 or 2 and positive cutoffs are allowed')
        variants.append((name, order, fwd, inv, lp))
    if len(set(v[0] for v in variants)) != len(variants):
        raise SystemExit('Variant names must be unique')
    return variants


def extract_variant(base_dir, wav, speaker, events, exe, events_csv, variant,
                    window_seconds, lp_order, cache_root, keep_audio=False, recompute=False):
    cache_dir = cache_root/speaker
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir/(variant[0]+'.npz')
    sig = config_signature(wav, exe, events_csv, variant, window_seconds, lp_order)
    if filename.exists() and not recompute:
        try:
            with np.load(filename, allow_pickle=False) as stored:
                if str(stored['signature'].item()) == sig:
                    return stored['X'], stored['Y'], stored['starts'], True
        except (ValueError, KeyError, OSError):
            pass
    name, order, forward_fc, inverse_fc, lp_fc = variant
    with tempfile.TemporaryDirectory(prefix='lf_generalized_') as tmp:
        work = base_dir/'processed_audio'/speaker/name if keep_audio else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        forward = work/'forward.wav'
        inverse = work/'inverse.wav'
        isolated = work/'isolated.wav'
        base.run_c_filter(exe, wav, forward, order, forward_fc, 1, 2)
        base.run_c_filter(exe, forward, inverse, order, inverse_fc, 1, 5)
        base.run_c_filter(exe, inverse, isolated, lp_order, 1, lp_fc, 1)
        X, Y, starts = [], [], []
        for start, audio, sr in base.read_windows(isolated, window_seconds):
            full_y = base.label_window(events[speaker], start, window_seconds)
            if full_y is None:
                continue
            X.append(base.feature_vector(audio, sr).astype(np.float32))
            Y.append(full_y[:2])  # base.LABELS order: low_frequency, 60hz_hum, broadband
            starts.append(start)
    if not X:
        raise RuntimeError(f'No evaluable windows for {speaker} {name}')
    arrays = {'X':np.asarray(X, np.float32), 'Y':np.asarray(Y, np.uint8),
              'starts':np.asarray(starts, np.float64), 'signature':np.asarray(sig)}
    tmp_path = filename.with_suffix('.tmp.npz')
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, filename)
    return arrays['X'], arrays['Y'], arrays['starts'], False


def metrics(y, pred):
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':round(precision,6) if precision is not None else None,
            'recall':round(recall,6) if recall is not None else None,
            'f1':round(f1,6) if f1 is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='Filters/dataset/noisy')
    parser.add_argument('--events-csv', default='layer2_noise_events.csv')
    parser.add_argument('--filter-exe', default='Filters/filter')
    parser.add_argument('--out', default='results/lf-rf-generalized-v1')
    parser.add_argument('--cache-dir', default=None,
                        help='Optional shared feature cache (default: OUT/feature_cache)')
    parser.add_argument('--window-seconds', type=float, default=5.)
    parser.add_argument('--lp-order', type=int, choices=(4,), default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-fraction', type=float, default=.2)
    parser.add_argument('--max-speakers', type=int, default=0,
                        help='Use small number for smoke tests; 0 uses all speakers')
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--preset', choices=('focused', 'full'), default='focused',
                        help='focused: 8 configs at 200/300/400; full: adds 150/600 (12 configs)')
    parser.add_argument('--variant', action='append',
                        help='Custom NAME:ORDER:FORWARD_HP:INVERSE_HP:LP; repeatable. Replaces defaults.')
    parser.add_argument('--recompute', action='store_true', help='Rebuild existing feature caches')
    parser.add_argument('--keep-audio', action='store_true', help='Preserve all intermediate WAVs (large!)')
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.trees < 1 or not 0 < args.test_fraction < 1:
        parser.error('Invalid numeric training parameter')
    events_csv = Path(args.events_csv).resolve()
    data_root = Path(args.data_root).resolve()
    exe = Path(args.filter_exe).resolve()
    for path in (events_csv, exe):
        if not path.is_file():
            parser.error(f'Missing {path}')
    events = base.load_events(events_csv)
    speakers = list(base.discover_speakers(data_root, events))
    if args.max_speakers:
        speakers = speakers[:args.max_speakers]
    if len(speakers) < 4:
        parser.error('Need >= 4 speakers')
    variants = custom_variants(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else out/'feature_cache'
    # The same deterministic speaker split as the original baseline for all 109 speakers.
    speaker_names = np.asarray([s for s, _ in speakers])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_fraction,
                                 random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(speaker_names, groups=speaker_names))
    train_speakers = set(speaker_names[tr_idx]); test_speakers = set(speaker_names[te_idx])
    assert not (train_speakers & test_speakers)
    print(f'[split] training speakers={len(train_speakers)}, '
          f'testing speakers={len(test_speakers)}; '
          f'{len(variants)} variants; {args.trees} trees/model', flush=True)
    manifest = {'train_speakers':sorted(train_speakers),'test_speakers':sorted(test_speakers),
                'variants':[{'name':v[0],'order':v[1], 'forward_hp_hz':v[2],
                             'inverse_hp_hz':v[3], 'lp_hz':v[4]} for v in variants],
                'feature_version':base.FEATURE_VERSION,
                'labels':list(LABELS),'window_seconds':args.window_seconds,
                'seed':args.seed}
    atomic_json(out/'experiment.json', manifest)
    data = {v[0]:{'train_X':[],'train_Y':[],'test_X':[],'test_Y':[],
                  'test_starts':[],'test_speakers':[]} for v in variants}
    misses = hits = 0
    for j, (speaker, wav) in enumerate(speakers, 1):
        print(f'[{j}/{len(speakers)}] {speaker}', flush=True)
        split = 'train' if speaker in train_speakers else 'test'
        for variant in variants:
            X, Y, starts, cached = extract_variant(out, wav.resolve(), speaker, events,
                    exe, events_csv, variant, args.window_seconds, args.lp_order, cache_root,
                    args.keep_audio, args.recompute)
            hits += int(cached); misses += int(not cached)
            node = data[variant[0]]
            node[f'{split}_X'].append(X)
            node[f'{split}_Y'].append(Y)
            if split == 'test':
                node['test_starts'].extend(starts.tolist())
                node['test_speakers'].extend([speaker]*len(starts))
        print(f'  cache: {hits} reused, {misses} processed', flush=True)
    # Fit models using a single set of train speakers, all their configuration variants.
    train_X = np.vstack([x for v in variants for x in data[v[0]]['train_X']])
    train_Y = np.vstack([y for v in variants for y in data[v[0]]['train_Y']])
    print(f'[train] {len(train_X)} windows, {train_X.shape[1]} features; '
          f'{len(test_speakers)} held-out speakers', flush=True)
    models = []
    for k, name in enumerate(LABELS):
        print(f'[train] fitting {name} ({args.trees} trees)', flush=True)
        model = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=3,
                class_weight='balanced_subsample', n_jobs=-1, random_state=args.seed)
        model.fit(train_X,train_Y[:,k])
        models.append(model)
    # Keep original package keys for compatibility with original `predict` command.
    package = {'models':models,'labels':LABELS,'feature_version':base.FEATURE_VERSION,
               'window_seconds':args.window_seconds,'hp_order':None,'cutoff':300.,
               'lp_order':args.lp_order,'positive_occupancy':.8,'negative_occupancy':.2,
               'training_variants':manifest['variants'],
               'model_kind':'lf_generalized_v1'}
    joblib.dump(package, out/'model.joblib')
    print('[evaluate] models saved; batched held-out predictions', flush=True)
    summary = {'n_train_windows':int(len(train_X)),'n_train_speakers':len(train_speakers),
               'n_test_speakers':len(test_speakers), 'test_speakers':sorted(test_speakers),
               'by_variant':{},'warning':'Speaker split is independent, but reused physical noise recordings across speakers are not independent noise realizations.'}
    pooled_truth,pooled_prediction = [], []
    rows = []
    for variant in variants:
        node = data[variant[0]]
        X = np.vstack(node['test_X']); Y = np.vstack(node['test_Y'])
        probabilities = np.column_stack([
            m.predict_proba(X)[:,list(m.classes_).index(1)] if 1 in m.classes_
            else np.zeros(len(X)) for m in models])
        predicted = (probabilities >= .5).astype(np.uint8)
        summary['by_variant'][variant[0]] = {'n_test_windows':int(len(X)),
            'metrics':{name:metrics(Y[:,k],predicted[:,k]) for k,name in enumerate(LABELS)}}
        pooled_truth.append(Y);pooled_prediction.append(predicted)
        for j, start in enumerate(node['test_starts']):
            rows.append([variant[0],node['test_speakers'][j],start,
                int(Y[j,0]),int(Y[j,1]),round(float(probabilities[j,0]),6),
                round(float(probabilities[j,1]),6),int(predicted[j,0]),int(predicted[j,1])])
        atomic_json(out/'evaluation.json',summary)  # checkpoint after each variant
        print(f'[evaluate] {variant[0]} LF={summary["by_variant"][variant[0]]["metrics"]["low_frequency"]["f1"]}, '
              f'hum={summary["by_variant"][variant[0]]["metrics"]["60hz_hum"]["f1"]}',flush=True)
    Y = np.vstack(pooled_truth); P = np.vstack(pooled_prediction)
    summary['pooled_metrics'] = {name:metrics(Y[:,k],P[:,k]) for k,name in enumerate(LABELS)}
    summary['n_test_windows_total_over_variants'] = len(Y)
    atomic_json(out/'evaluation.json',summary)
    with open(out/'heldout_windows.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant','speaker','start_seconds','true_low_frequency','true_60hz_hum',
                         'score_low_frequency','score_60hz_hum','pred_low_frequency','pred_60hz_hum'])
        writer.writerows(rows)
    print('[done] '+str(out/'evaluation.json'),flush=True)
    print(json.dumps(summary['pooled_metrics'], indent=2),flush=True)


if __name__ == '__main__':
    main()
