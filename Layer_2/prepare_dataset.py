"""Validate Ardour event labels against exported speech+noise WAVs.

Does not duplicate, transform, or generate audio. Keeps the ground-truth event
JSON intact. Output is a *resolved* per-session index used by training.
"""
import argparse
import json
import re
import wave
from pathlib import Path

from Layer_2.audio_features import SAMPLE_RATE, validate_wav

CLASSES = ('low_frequency', 'high_frequency', 'broadband', '60hz_hum')


def load_event_manifest(path):
    with open(path, encoding='utf-8') as fp:
        raw = json.load(fp)
    sessions = raw.get('sessions')
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError('Expected the Ardour event manifest JSON with nonempty sessions.')
    for speaker, entry in sessions.items():
        for event in entry.get('events', []):
            labels = event.get('labels', [])
            if not isinstance(labels, list) or any(x not in CLASSES for x in labels):
                raise ValueError(f'Unrecognized labels in {speaker}: {labels}')
            begin = float(event['start_seconds'])
            finish = float(event['end_seconds'])
            if not 0 <= begin < finish:
                raise ValueError(f'Invalid event interval in {speaker}: {begin}..{finish}')
    return raw


def prepare_dataset(events, audio_dir, output_file, *, strict=False):
    """Create file index from <audio_dir>/SpeakerN.wav and Ardour manifest.

    In strict mode all sessions require a corresponding WAV. Export offsets
    must be validated before training; they are not inferred from filenames.
    """
    source = load_event_manifest(events)
    audio_dir = Path(audio_dir).resolve()
    index = []
    missing = []
    mismatches = []
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)
    for speaker, entry in sorted(source['sessions'].items(), key=lambda pair: int(re.search(r'\d+', pair[0]).group()) if re.search(r'\d+', pair[0]) else pair[0]):
        wav = audio_dir / f'{speaker}.wav'
        if not wav.is_file():
            missing.append(speaker)
            continue
        with wave.open(str(wav), 'rb') as w:
            validate_wav(w)
            duration = w.getnframes() / SAMPLE_RATE
        last_event = max((float(e['end_seconds']) for e in entry['events']), default=0.0)
        if last_event > duration + 5.0:
            mismatches.append({'speaker': speaker, 'last_event_seconds': round(last_event, 3), 'wav_seconds': round(duration, 3)})
        index.append({'speaker': speaker, 'wav_path': str(wav), 'duration_seconds': duration, 'events': entry['events']})
    if strict and (missing or mismatches):
        raise ValueError(f'Missing audio: {missing}; event/WAV duration mismatches: {mismatches}')
    if len(index) < 2:
        raise ValueError('Need at least two source recordings for grouped training/evaluation.')
    result = {'schema_version': 1, 'label_names': list(CLASSES), 'source_events': str(Path(events).resolve()),
              'audio_dir': str(audio_dir), 'sessions': index, 'missing_speakers': missing,
              'duration_mismatches': mismatches,
              'warning': 'Assumes WAV sample 0 equals Ardour timeline 0; verify export alignment.'}
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Indexed {len(index)} WAV recordings; missing: {len(missing)}; duration warnings: {len(mismatches)}')
    print(f'Saved: {output}')
    if mismatches:
        print('WARNING: Some event times extend beyond WAV duration. Check export offsets/alignment.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Layer 2 event/WAV index (no audio copying).')
    parser.add_argument('--events', required=True, help='layer2_noise_events.json')
    parser.add_argument('--audio-dir', required=True, help='Folder of exported Speaker1.wav..Speaker109.wav')
    parser.add_argument('--output', default='Layer_2/data/audio_index.json')
    parser.add_argument('--strict', action='store_true', help='Fail if any WAV is missing or events extend past its duration')
    args = parser.parse_args()
    prepare_dataset(args.events, args.audio_dir, args.output, strict=args.strict)
