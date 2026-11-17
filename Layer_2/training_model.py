import csv
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from audio_features import extract_features, FEATURE_NAMES, SAMPLE_RATE, WINDOW_SECONDS

def load_dataset(manifest):
    manifest = Path(manifest)
    features, labels, groups = [], [], []
    
    with open(manifest, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        
        if not {"wav_path", "label", "source_id"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain wav_path, label, and source_id.")
            
        for row in reader:
            path = Path(row["wav_path"])
            
            if not path.is_absolute():
                path = manifest.parent / path
            if not row["label"].strip() or not row["source_id"].strip():
                raise ValueError("Every recording needs a label and source_id.")
                
            features.append(extract_features(path)) # Extract audio features from the WAV
            labels.append(row["label"].strip())
            groups.append(row["source_id"].strip()) # Keep recordings from the same source together
            
    if not features:
        raise ValueError("No training recordings found.")
        
    return np.array(features), np.array(labels), np.array(groups)

def train_model(manifest, output):
    X, y, groups = load_dataset(manifest)
    classes = np.unique(y)
    
    if len(classes) != 4:
        raise ValueError("Expected four noise classes.")
        
    for label in classes:
        if len(np.unique(groups[y == label])) < 2:
            raise ValueError(f"Need at least two independent sources for {label}.")
            
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=42) # 75/25 training/testing split
    
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(y[train_idx]) == set(classes) and set(y[test_idx]) == set(classes):
            break
    else:
        raise ValueError("Could not create a split containing all four classes.")
        
    model = RandomForestClassifier(
        n_estimators=300, # Number of decision trees
        class_weight="balanced", # Rebalances for unequal class sizes
        random_state=42,
        n_jobs=-1 # Using available CPU cores
    )
    model.fit(X[train_idx], y[train_idx]) # Train using labeled audio features
    predictions = model.predict(X[test_idx])
    accuracy = accuracy_score(y[test_idx], predictions)
    report = classification_report(y[test_idx], predictions, zero_division=0)

    print(f"\nTraining recordings: {len(train_idx)}")
    print(f"Testing recordings: {len(test_idx)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(report)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "classes": classes.tolist()
    }, output) # Save the model/model configuration
    print(f"Model saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="models/noise_rf.joblib")
    args = parser.parse_args()
    train_model(args.manifest, args.output)
