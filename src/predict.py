"""Load saved LIAR artifacts and predict a single political statement."""
from __future__ import annotations

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class ModelNotTrainedError(FileNotFoundError):
    pass


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    paths = {"model": model_dir / "final_model.joblib",
             "vectorizer": model_dir / "tfidf_vectorizer.joblib",
             "explainer": model_dir / "explanation_model.joblib",
             "metadata": model_dir / "metadata.joblib"}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ModelNotTrainedError("TruthLens model isn't trained yet. Run: python src/train.py")
    return {name: joblib.load(path) for name, path in paths.items()}


def predict_statement(statement: str, artifacts: dict) -> dict:
    if not statement or not statement.strip():
        raise ValueError("Please enter a meaningful claim.")
    from src.preprocessing import clean_text, binary_label, LIAR_LABELS
    cleaned = clean_text(statement)
    if len(cleaned.split()) < 3:
        raise ValueError("Please enter a little more text (at least three meaningful words).")
    model = artifacts["model"]
    predicted = str(model.predict([cleaned])[0])
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved model does not provide calibrated class probabilities.")
    raw_probabilities = model.predict_proba([cleaned])[0]
    probabilities = {label: float(raw_probabilities[list(model.classes_).index(label)])
                     for label in LIAR_LABELS}
    explanation = _explanation(cleaned, predicted, artifacts)
    return {"label": predicted, "binary_interpretation": binary_label(predicted),
            "confidence": probabilities[predicted], "probabilities": probabilities,
            "model_name": artifacts["metadata"]["model_name"], "word_count": len(cleaned.split()),
            "explanation": explanation}


def _explanation(cleaned: str, predicted: str, artifacts: dict) -> dict:
    from src.explain import explain_prediction
    return explain_prediction(cleaned, artifacts["vectorizer"], artifacts["explainer"], predicted)
