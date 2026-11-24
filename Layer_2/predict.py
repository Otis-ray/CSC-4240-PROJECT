"""Multi-label, window-level Layer 2 noise detection with event intervals."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from Layer_2.audio_features import FEATURE_NAMES, SAMPLE_RATE, iter_audio_windows
from Layer_2.prepare_dataset import CLASSES


def class_probabilities(model, X):
    """Return Nx4 probabilities of positive class, including constant models."""
    raw = model.predict_proba(X)
    if len(raw) != len(CLASSES):
        raise ValueError(f'Expected {len(CLASSES)} label classifiers, got {len(raw)}')
    columns = []
    for estimator, probs in zip(model.estimators_, raw):
        known = list(estimator.classes_)
        columns.append(probs[:, known.index(1)] if 1 in known else np.zeros(len(X)))
    return np.column_stack(columns)


def consolidate_events(windows, labels=CLASSES):
    """Merge consecutive positive windows into one event per noise type."""
    events = []
    for label in labels:
        active = None
        for row in windows:
            score = row['scores'][label]
            if row['detected'][label]:
                if active is None:
                    active = {'type': label, 'start_seconds': row['start_seconds'],
                              'end_seconds': row['end_seconds'], 'scores': [score]}
                else:
                    active['end_seconds'] = row['end_seconds']
                    active['scores'].append(score)
            elif active is not None:
                events.append({'type': active['type'], 'start_seconds': active['start_seconds'],
                               'end_seconds': active['end_seconds'],
                               'mean_confidence': round(float(np.mean(active['scores'])), 4)})
                active = None
        if active is not None:
            events.append({'type': active['type'], 'start_seconds': active['start_seconds'],
                           'end_seconds': active['end_seconds'],
                           'mean_confidence': round(float(np.mean(active['scores'])), 4)})
    return sorted(events, key=lambda event: (event['start_seconds'], event['type']))


def predict_noise(model_path, input_wav, output_json, *, threshold=None):
    payload = joblib.load(model_path)
    if payload.get('schema_version') != 2:
        raise ValueError('This model is not the refactored multi-label Layer 2 model. Retrain with training_model.py.')
    if payload.get('feature_names') != FEATURE_NAMES or payload.get('sample_rate') != SAMPLE_RATE or payload.get('classes') != list(CLASSES):
        raise ValueError('Model feature/sample-rate/label schema does not match prediction code.')
    window_seconds = float(payload['window_seconds'])
    threshold = float(payload.get('threshold', 0.5) if threshold is None else threshold)
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    rows = list(iter_audio_windows(input_wav, window_seconds=window_seconds, include_partial=True))
    if not rows:
        raise ValueError('Input WAV contains no usable audio windows.')
    X = np.vstack([x[2] for x in rows])
    scores = class_probabilities(payload['model'], X)
    windows = []
    for (start, end, _), score_vec in zip(rows, scores):
        windows.append({'start_seconds': round(start, 6), 'end_seconds': round(end, 6),
                        'scores': {label: round(float(score_vec[i]), 4) for i, label in enumerate(CLASSES)},
                        'detected': {label: bool(score_vec[i] >= threshold) for i, label in enumerate(CLASSES)}})
    events = consolidate_events(windows)
    summary = {label: {'detected': any(w['detected'][label] for w in windows),
                       'max_score': round(float(np.max(scores[:, i])), 4),
                       'positive_windows': sum(w['detected'][label] for w in windows)}
               for i, label in enumerate(CLASSES)}
    result = {'schema_version': 2, 'file': str(Path(input_wav).resolve()),
              'analysis_scope': 'window_level_multilabel', 'window_seconds': window_seconds,
              'threshold': threshold, 'noise_types': summary, 'events': events, 'windows': windows,
              'note': 'Event boundaries are quantized to window edges; model scores are not calibrated probabilities.'}
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Classified {len(windows)} windows, detected {len(events)} events. Saved: {output}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify noise events in one WAV using a trained multi-label model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    predict_noise(args.model, args.input, args.output, threshold=args.threshold)
