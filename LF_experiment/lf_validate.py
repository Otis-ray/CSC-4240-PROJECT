#!/usr/bin/env python3
"""Independent validation suite for LF_experiment/lf_experiment.py.
Uses the SAVED RF model; does not retrain or touch Layer1/Layer2.
"""
import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np


def load_module(script):
    spec = importlib.util.spec_from_file_location('lf_original_experiment', script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import LF feature extraction script: {script}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def batch_scores(m, pkg, wav, threshold):
    starts, features = [], []
    for start, audio, sr in m.read_windows(wav, pkg['window_seconds']):
        starts.append(float(start))
        features.append(m.feature_vector(audio, sr))
    if not features:
        raise RuntimeError(f'No complete windows: {wav}')
    matrix = np.stack(features)
    scores = {}
    for name, clf in zip(pkg['labels'], pkg['models']):
        if 1 not in clf.classes_:
            scores[name] = np.zeros(len(matrix))
        else:
            scores[name] = clf.predict_proba(matrix)[:, list(clf.classes_).index(1)]
    rows = []
    for i, t in enumerate(starts):
        row = {'start_seconds': t, 'end_seconds': t + pkg['window_seconds'],
               'scores': {k: round(float(v[i]), 6) for k,v in scores.items()},
               'detected': {k: bool(v[i] >= threshold) for k,v in scores.items()}}
        rows.append(row)
    return rows


def evaluate_rows(rows, events, m, pkg):
    labels = list(pkg['labels'])
    count = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for k in labels}
    evaluable = 0
    unknown = 0
    for row in rows:
        y = m.label_window(events, row['start_seconds'], pkg['window_seconds'],
                           pkg.get('positive_occupancy', .8),
                           pkg.get('negative_occupancy', .2))
        if y is None:
            unknown += 1
            continue
        evaluable += 1
        for k, truth in zip(labels, y):
            pred = row['detected'][k]
            count[k][('tp' if truth else 'fp') if pred else ('fn' if truth else 'tn')] += 1
    for k, c in count.items():
        tp,fp,fn = c['tp'], c['fp'], c['fn']
        c['precision'] = round(tp/(tp+fp), 5) if tp+fp else None
        c['recall'] = round(tp/(tp+fn), 5) if tp+fn else None
        c['f1'] = round(2*tp/(2*tp+fp+fn), 5) if 2*tp+fp+fn else None
    return {'evaluable_windows': evaluable, 'ambiguous_windows': unknown, 'metrics': count}


def negative_metrics(rows):
    return {'windows': len(rows), 'positive_windows':
            {k: sum(int(r['detected'][k]) for r in rows) for k in rows[0]['detected']},
            'max_score': {k: round(max(r['scores'][k] for r in rows), 6) for k in rows[0]['scores']}}


def create_audio(m, exe, src, folder, order, forward_fc, inverse_fc, lp_fc):
    folder.mkdir(parents=True, exist_ok=True)
    fwd, inv, isolated = folder/'forward.wav', folder/'inverse.wav', folder/'isolated.wav'
    m.run_c_filter(exe, src, fwd, order, forward_fc, 1, 2)
    m.run_c_filter(exe, fwd, inv, order, inverse_fc, 1, 5)
    m.run_c_filter(exe, inv, isolated, 4, 1, lp_fc, 1)
    return isolated


def isolate_reconstruction(m, exe, src, folder, lp_fc):
    folder.mkdir(parents=True, exist_ok=True)
    isolated = folder/'isolated.wav'
    m.run_c_filter(exe, src, isolated, 4, 1, lp_fc, 1)
    return isolated


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-script', default='LF_experiment/lf_experiment.py')
    p.add_argument('--model', default='results/lf-rf-hp300-order2/model.joblib')
    p.add_argument('--evaluation', default='results/lf-rf-hp300-order2/evaluation.json')
    p.add_argument('--events-csv', default='layer2_noise_events.csv')
    p.add_argument('--filter-exe', default='Filters/filter')
    p.add_argument('--noisy-root', default='Filters/dataset/noisy')
    p.add_argument('--clean-root', default='Filters/dataset/clean')
    p.add_argument('--out', default='results/lf-validation')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--max-speakers', type=int, default=0,
                   help='0 means all held-out speakers; 3 for a quick smoke test')
    p.add_argument('--keep-audio', action='store_true', help='Keep intermediate WAVs (large disk usage)')
    subs = p.add_subparsers(dest='test', required=True)
    s = subs.add_parser('heldout', help='Test the saved model on held-out speaker WAVs')
    s.add_argument('--recon-root', default=None,
                   help='Optional directory containing ALREADY automatically reconstructed SpeakerN.wav files')
    s.add_argument('--lp-cutoff', type=float, default=335.,
                   help='LP cutoff for recon-root recordings (default 335 Hz; use each file actual Layer1 estimate if available)')
    r = subs.add_parser('robustness', help='Change forward HP order and cutoff while holding model fixed')
    r.add_argument('--speaker', default='Speaker1')
    r.add_argument('--variants', choices=['quick','full'], default='quick')
    n = subs.add_parser('negative', help='Evaluate clean speech plus optional independent negative recordings')
    n.add_argument('--extra-wav', action='append', default=[],
                   help='Optional WAV known to contain NO three target noise types; repeatable')
    n.add_argument('--extra-already-reconstructed', action='store_true',
                   help='Treat --extra-wav as already reconstructed instead of forward/inverse processing')
    return p


def main():
    args = make_parser().parse_args()
    mod = load_module(Path(args.experiment_script).resolve())
    pkg = joblib.load(args.model)
    if pkg['feature_version'] != mod.FEATURE_VERSION:
        raise SystemExit('Feature version mismatch between model and LF_experiment script')
   # Allow models trained on a subset of the original labels.
    if any(label not in mod.LABELS for label in pkg['labels']):
        raise SystemExit('Model contains unrecognized labels')

    # Preserve the expected label order.
    expected = [label for label in mod.LABELS if label in pkg['labels']]

    if list(pkg['labels']) != expected:
        raise SystemExit('Label order mismatch')
    eval_info = json.loads(Path(args.evaluation).read_text())
    heldout = eval_info['test_speakers']
    if args.max_speakers:
        heldout = heldout[:args.max_speakers]
    event_lookup = mod.load_events(args.events_csv)
    exe = Path(args.filter_exe).resolve()
    if not exe.is_file():
        raise SystemExit(f'Cannot find C filter executable: {exe}')
    outroot = Path(args.out).resolve()/args.test
    outroot.mkdir(parents=True, exist_ok=True)
    reports = []

    def do_one(name, src, order=2, forward_fc=300, inverse_fc=300, lp_fc=300,
               events=None, reconstructed=False, negative=False):
        src = Path(src).resolve()
        if not src.is_file():
            print(f'[skip] Missing {src}', flush=True)
            return
        print(f'[run] {name}: {src.name}; HP(order={order}, fc={forward_fc}); '
              f'inverse={inverse_fc}, LP={lp_fc}', flush=True)
        with tempfile.TemporaryDirectory(prefix='lf_validation_') as temp:
            folder = outroot/'wav'/name if args.keep_audio else Path(temp)
            if reconstructed:
                isolated = isolate_reconstruction(mod, exe, src, folder, lp_fc)
            else:
                isolated = create_audio(mod, exe, src, folder, order, forward_fc, inverse_fc, lp_fc)
            rows = batch_scores(mod, pkg, isolated, args.threshold)
            report = {'name': name, 'source': str(src), 'hp_order':order,
                      'forward_hp_hz':forward_fc if not reconstructed else None,
                      'inverse_hp_hz':inverse_fc if not reconstructed else None,
                      'lp_hz':lp_fc, 'n_windows':len(rows)}
            if events is not None:
                report.update(evaluate_rows(rows,events,mod,pkg))
            if negative:
                report['negative_control'] = negative_metrics(rows)
            # Detailed per-window predictions for checking true/false detections.
            mod.write_json(outroot/'predictions'/f'{name}.json', {
                'source':str(src), 'test':args.test,'windows':rows})
            reports.append(report)
            mod.write_json(outroot/'summary.json', {
                'test':args.test,'training_model':str(Path(args.model).resolve()),
                'threshold':args.threshold,'results':reports})
    if args.test=='heldout':
        for speaker in heldout:
            if args.recon_root:
                do_one(speaker, Path(args.recon_root)/f'{speaker}.wav', events=event_lookup[speaker],
                       reconstructed=True,lp_fc=args.lp_cutoff)
            else:
                do_one(speaker,Path(args.noisy_root)/f'{speaker}.wav',
                       events=event_lookup[speaker], lp_fc=300)
    elif args.test=='robustness':
        speaker=args.speaker
        source=Path(args.noisy_root)/f'{speaker}.wav'
        variants = [(2,300,300,300,'baseline_2_300'),
                    (2,300,335,335,'cutoff_error_2_300_to_335'),
                    (1,300,300,300,'order1_300'),
                    (4,300,300,300,'order4_300')]
        if args.variants=='full':
            variants += [(2,150,150,150,'order2_150'),
                         (2,600,600,600,'order2_600'),
                         (2,150,180,180,'order2_150_to_180'),
                         (2,600,670,670,'order2_600_to_670')]
        for order,forward_fc,inverse_fc,lp_fc,name in variants:
            do_one(f'{speaker}_{name}', source, order,forward_fc,inverse_fc,lp_fc,
                   events=event_lookup[speaker])
    elif args.test=='negative':
        found=0
        for speaker in heldout:
            src = Path(args.clean_root)/f'{speaker}.wav'
            if src.is_file():
                found+=1
                do_one(f'clean_{speaker}',src,negative=True)
        if not found:
            print(f'[note] No held-out clean WAVs found under {args.clean_root}. '
                  'Set --clean-root to the correct directory.',flush=True)
        for i,src in enumerate(args.extra_wav,1):
            do_one(f'extra_negative_{i}',src,reconstructed=args.extra_already_reconstructed,
                   negative=True)
    if not reports:
        raise SystemExit('No recordings processed; check paths and speaker filenames')
    print(f'Completed {len(reports)} recordings. Summary: {outroot/"summary.json"}')
    for r in reports:
        if 'metrics' in r:
            print(r['name'], ' '.join(f'{k}:F1={v["f1"]} TP={v["tp"]} FP={v["fp"]} FN={v["fn"]}'
                                       for k,v in r['metrics'].items()))
        else:
            print(r['name'],r['negative_control']['positive_windows'])

if __name__=='__main__':
    main()
