"""Grouped, window-level, multi-label noise detection using Random Forest.

Outputs four independent presence decisions, NOT one exclusive class.
Trains only on user-supplied labeled WAVs; generates no synthetic audio.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GroupShuffleSplit

from Layer_2.audio_features import FEATURE_NAMES, SAMPLE_RATE, WINDOW_SECONDS, iter_audio_windows
from Layer_2.prepare_dataset import CLASSES


def label_coverage(events, start, end):
    """Fraction of window overlapped by each type, merging intervals per type."""
    overlaps = {label: [] for label in CLASSES}
    for event in events:
        left = max(float(event['start_seconds']), start)
        right = min(float(event['end_seconds']), end)
        if right <= left:
            continue
        for label in event['labels']:
            overlaps[label].append((left, right))
    fractions = []
    for label in CLASSES:
        length = 0.0
        current = None
        for left, right in sorted(overlaps[label]):
            if current is None:
                current = [left, right]
            elif left <= current[1]:
                current[1] = max(right, current[1])
            else:
                length += current[1] - current[0]
                current = [left, right]
        if current:
            length += current[1] - current[0]
        fractions.append(min(1.0, length / (end - start)))
    return np.array(fractions, dtype=np.float64)


def window_label(fractions, *, positive=0.80, negative=0.10):
    """Return labels or None for ambiguous boundary windows.

    If a class covers >=80%, present. If <=10%, absent. Ambiguous intermediate
    coverages cause us to omit the window entirely, avoiding false labels.
    """
    if np.any((fractions > negative) & (fractions < positive)):
        return None
    return (fractions >= positive).astype(np.int8)


def load_training_windows(index_path, window_seconds=WINDOW_SECONDS, max_windows_per_speaker=None):
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    if index.get('label_names') != list(CLASSES):
        raise ValueError('Index label names do not match model label schema.')
    features, labels, speakers = [], [], []
    skipped = 0
    for i, row in enumerate(index['sessions'], start=1):
        n = 0
        for start, end, feature in iter_audio_windows(row['wav_path'], window_seconds=window_seconds):
            if max_windows_per_speaker is not None and n >= max_windows_per_speaker:
                break
            n += 1
            y = window_label(label_coverage(row['events'], start, end))
            if y is None:
                skipped += 1
                continue
            features.append(feature)
            labels.append(y)
            speakers.append(row['speaker'])
        if i % 10 == 0 or i == len(index['sessions']):
            print(f'Processed {i}/{len(index["sessions"])} speakers; usable windows: {len(labels)}', flush=True)
    if not features:
        raise ValueError('No usable windows. Check exported audio length and event alignment.')
    return np.array(features), np.array(labels), np.array(speakers), skipped


def train_model(index_path, output, *, window_seconds=WINDOW_SECONDS, test_fraction=0.25,
                seed=42, max_windows_per_speaker=None):
    X, y, groups, skipped = load_training_windows(index_path, window_seconds, max_windows_per_speaker)
    if len(np.unique(groups)) < 3:
        raise ValueError('At least three distinct speakers required for train/test split.')
    # All classes must vary between 0 and 1 in both train and held-out groups.
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_fraction, random_state=seed)
    for train_idx, test_idx in splitter.split(X, groups=groups):
        if all(len(np.unique(y[part, k])) == 2 for part in (train_idx, test_idx) for k in range(len(CLASSES))):
            break
    else:
        raise ValueError('Cannot split speakers with positive and negative windows for every class.')

    # MultiOutputClassifier gives each noise type an independent binary model.
    # Use independent trees per label to allow any combination of classes.
    from sklearn.multioutput import MultiOutputClassifier
    model = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1), n_jobs=1)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx]).astype(np.int8)
    pr, re, f1, _ = precision_recall_fscore_support(y[test_idx], pred, average=None, zero_division=0)
    metrics = {
        'samples': int(len(X)), 'skipped_ambiguous_windows': int(skipped),
        'train_windows': int(len(train_idx)), 'test_windows': int(len(test_idx)),
        'train_speakers': sorted(set(groups[train_idx].tolist())),
        'test_speakers': sorted(set(groups[test_idx].tolist())),
        'subset_accuracy': float(np.mean(np.all(y[test_idx] == pred, axis=1))),
        'micro_f1': float(f1_score(y[test_idx], pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y[test_idx], pred, average='macro', zero_division=0)),
        'per_class': {
            label: {'precision': float(pr[k]), 'recall': float(re[k]), 'f1': float(f1[k]),
                    'test_positive_windows': int(y[test_idx, k].sum()),
                    'test_negative_windows': int(len(test_idx) - y[test_idx, k].sum())}
            for k, label in enumerate(CLASSES)
        },
        'window_seconds': window_seconds,
        'note': 'Held-out speakers; test data comes from noisy mixture recordings, not reconstructed audio.'
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'schema_version': 2, 'model': model, 'feature_names': FEATURE_NAMES,
                 'sample_rate': SAMPLE_RATE, 'window_seconds': window_seconds,
                 'classes': list(CLASSES), 'threshold': 0.5}, output)
    metrics_path = output.with_name(output.stem + '_metrics.json')
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(f'Saved model: {output}\nMetrics: {metrics_path}')
    print(f'Window exact-match: {metrics["subset_accuracy"]:.3f}; micro F1: {metrics["micro_f1"]:.3f}; macro F1: {metrics["macro_f1"]:.3f}')
    for label, m in metrics['per_class'].items():
        print(f'  {label}: precision={m["precision"]:.3f} recall={m["recall"]:.3f} F1={m["f1"]:.3f}')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train Layer 2 multi-label window classifier.')
    p.add_argument('--manifest', required=True, help='audio_index.json from prepare_dataset')
    p.add_argument('--output', default='Layer_2/models/noise_multilabel.joblib')
    p.add_argument('--window-seconds', type=float, default=WINDOW_SECONDS)
    p.add_argument('--test-fraction', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-windows-per-speaker', type=int, default=None, help='Quick test only; limits each speaker to first N windows')
    a = p.parse_args()
    train_model(a.manifest, a.output, window_seconds=a.window_seconds,
                test_fraction=a.test_fraction, seed=a.seed,
                max_windows_per_speaker=a.max_windows_per_speaker)
