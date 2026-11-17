
import json
import argparse
import joblib
from pathlib import Path
from audio_features import extract_features, FEATURE_NAMES, SAMPLE_RATE, WINDOW_SECONDS


def predict_noise(model_path, input_wav, output_json):
    data = joblib.load(model_path) # Loads the trained model and its configuration

    if (data.get("feature_names") != FEATURE_NAMES or
        data.get("sample_rate") != SAMPLE_RATE or
        data.get("window_seconds") != WINDOW_SECONDS):
        raise ValueError("Model configuration does not match the audio feature extractor.")

    model = data["model"]
    features = extract_features(input_wav).reshape(1, -1) # Extracts features from the reconstructed WAV

    probabilities = model.predict_proba(features)[0] # Estimates probability for each noise class
    index = probabilities.argmax() # Finds the class with the highest probability

    results = {
        "file": str(input_wav),
        "noise_type": str(model.classes_[index]),
        "confidence": round(float(probabilities[index]), 4),
        "class_scores": {
            str(label): round(float(score), 4)
            for label, score in zip(model.classes_, probabilities)
        },
        "analysis_scope": "whole_recording"
    }

    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4) # Saves structured results for the offline application

    print(json.dumps(results, indent=4))
    print(f"Results saved to: {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predict_noise(args.model, args.input, args.output)
