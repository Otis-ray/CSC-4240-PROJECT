"""Canonical inference model locations (one active trained model per subsystem)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
LAYER1_MODEL = MODEL_ROOT / "layer1" / "model.joblib"
LAYER2_MODEL = MODEL_ROOT / "layer2" / "model.joblib"
LF_MODEL = MODEL_ROOT / "lf" / "model.joblib"
