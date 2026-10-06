"""Prediction helpers for saved TruthLens artifacts."""
from __future__ import annotations

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class ModelNotTrainedError(FileNotFoundError):
    pass


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    required = [model_dir / "final_model.joblib", model_dir / "tfidf_vectorizer.joblib", model_dir / "metadata.joblib"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ModelNotTrainedError("Model not trained yet. Run: python src/train.py")
    return {"model": joblib.load(required[0]), "vectorizer": joblib.load(required[1]),
            "metadata": joblib.load(required[2])}


def predict_article(article: str, artifacts: dict) -> dict:
    if not article or not article.strip():
        raise ValueError("Paste an article with at least a few words to analyze.")
    from src.preprocessing import clean_text
    cleaned = clean_text(article)
    if len(cleaned.split()) < 5:
        raise ValueError("Please enter a little more text (at least five meaningful words).")
    vector = artifacts["vectorizer"].transform([cleaned])
    model = artifacts["model"]
    label = str(model.predict(vector)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(probabilities[list(model.classes_).index(label)])
    elif hasattr(model, "decision_function"):
        # Model is calibrated during training; this branch is only a defensive fallback.
        raise RuntimeError("Saved classifier does not provide calibrated probabilities.")
    else:
        raise RuntimeError("Saved classifier does not provide confidence estimates.")
    from src.explain import explain_prediction
    explainer = artifacts["metadata"].get("explainer_model", model)
    explanation = explain_prediction(cleaned, artifacts["vectorizer"], explainer)
    return {"label": label, "confidence": confidence, "model_name": artifacts["metadata"]["model_name"],
            "word_count": len(article.split()), "explanation": explanation}
