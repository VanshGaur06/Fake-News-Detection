"""Load the WELFake production artifacts and classify full news articles."""
from __future__ import annotations

import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CLASS_LABELS = ("FAKE", "REAL")


class ModelNotTrainedError(FileNotFoundError):
    pass


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    paths = {"model": model_dir / "fake_news_model.joblib",
             "vectorizer": model_dir / "fake_news_vectorizer.joblib",
             "metadata": model_dir / "fake_news_metadata.json"}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ModelNotTrainedError("WELFake model artifacts are missing. Run: python src/train.py")
    artifacts = {"model": joblib.load(paths["model"]), "vectorizer": joblib.load(paths["vectorizer"])}
    artifacts["metadata"] = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    classes = [str(label) for label in artifacts["model"].classes_]
    if classes != list(CLASS_LABELS) or artifacts["metadata"].get("dataset") != "WELFake":
        raise ModelNotTrainedError("Production artifacts are not the trained WELFake binary model; retrain with: python src/train.py")
    artifacts["explainer"] = artifacts["model"]
    return artifacts


def predict_article(title: str, text: str, artifacts: dict) -> dict:
    from src.preprocessing import clean_text
    combined = " ".join(part.strip() for part in (title or "", text or "") if part and part.strip())
    cleaned = clean_text(combined)
    if len(cleaned.split()) < 3:
        raise ValueError("Please enter a longer article (at least three words).")
    vector = artifacts["vectorizer"].transform([cleaned])
    model = artifacts["model"]
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved Logistic Regression model does not provide predict_proba().")
    predicted = str(model.predict(vector)[0])
    raw_probabilities = model.predict_proba(vector)[0]
    probabilities = {label: float(raw_probabilities[list(model.classes_).index(label)])
                     for label in CLASS_LABELS}
    from src.explain import explain_prediction
    explanation = explain_prediction(cleaned, artifacts["vectorizer"], model, predicted)
    return {
        "label": predicted, "probabilities": probabilities,
        "fake_probability": probabilities["FAKE"], "real_probability": probabilities["REAL"],
        "confidence": probabilities[predicted], "model_name": "Logistic Regression",
        "word_count": len(cleaned.split()), "explanation": explanation,
    }


def predict_statement(statement: str, artifacts: dict) -> dict:
    """Compatibility entry point for the UI's single free-text field."""
    return predict_article("", statement, artifacts)
