import csv
import wave
import argparse
import os
from pathlib import Path


def validate_wav(file_path):
    with wave.open(str(file_path), "rb") as wav:
        valid = (wav.getframerate() == 48000 and
                 wav.getnchannels() == 1 and
                 wav.getsampwidth() == 2 and
                 wav.getcomptype() == "NONE")
        if not valid:
            raise ValueError(f"Invalid WAV format: {file_path}") # Requires 48 kHz mono 16-bit PCM

def prepare_dataset(data_dir, output_file):
    data_dir = Path(data_dir)
    output = Path(output_file)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    rows = []
    classes = set()
    sources = {}
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            for wav_file in sorted(source_dir.rglob("*")):
                if not wav_file.is_file() or wav_file.suffix.lower() != ".wav":
                    continue
                validate_wav(wav_file)
                classes.add(label_dir.name)
                sources.setdefault(label_dir.name, set()).add(source_dir.name)
                rows.append({
                    "wav_path": os.path.relpath(wav_file, output.parent), # Stores path relative to the CSV
                    "label": label_dir.name, # Noise classification from folder name
                    "source_id": source_dir.name # Original recording or speaker identifier
                })
    if len(classes) != 4:
        raise ValueError(f"Expected 4 noise classes, found {len(classes)}.")
    for label, source_ids in sources.items():
        if len(source_ids) < 2:
            raise ValueError(f"Not enough independent sources for {label}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["wav_path", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows) # Generates the CSV used by train_model.py
    print(f"Dataset created: {output}")
    print(f"Total WAV files: {len(rows)}")
    print(f"Noise classes: {len(classes)}")
    for label in sorted(classes):
        count = sum(row["label"] == label for row in rows)
        print(f"{label}: {count} recordings, {len(sources[label])} sources")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="Layer_2/data/manifest.csv")
    args = parser.parse_args()
    prepare_dataset(args.data, args.output)
