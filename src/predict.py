"""Inference module for predicting truthfulness of news claims."""
import json
from pathlib import Path
import sys
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import clean_text, CLASSES
from src.explain import explain_prediction

MODEL_DIR = ROOT_DIR / "models"


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    """Load trained model, vectorizer, and metadata from disk."""
    model_path = model_dir / "model.joblib"
    vectorizer_path = model_dir / "vectorizer.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts not found. Please run 'python src/train.py' first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metadata": metadata,
    }


def predict_claim(statement: str, artifacts: dict) -> dict:
    """Classify a statement as FAKE or REAL and compute word contributions."""
    cleaned = clean_text(statement)
    if len(cleaned.split()) < 2:
        raise ValueError("Please enter a longer claim (at least two words).")

    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]

    x_vec = vectorizer.transform([cleaned])
    predicted_label = str(model.predict(x_vec)[0])

    # Class probabilities
    probabilities = model.predict_proba(x_vec)[0]
    class_idx = list(model.classes_).index(predicted_label)
    confidence = float(probabilities[class_idx])

    fake_idx = list(model.classes_).index("FAKE")
    real_idx = list(model.classes_).index("REAL")

    fake_prob = float(probabilities[fake_idx])
    real_prob = float(probabilities[real_idx])

    # Local feature explanations
    explanation = explain_prediction(cleaned, vectorizer, model)

    return {
        "statement": statement,
        "cleaned_text": cleaned,
        "prediction": predicted_label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "word_count": len(cleaned.split()),
        "explanation": explanation,
    }
