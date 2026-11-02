"""Unit and integration tests for the TruthLens fake news detector."""
import pytest
import numpy as np

from src.dataset import load_split, clean_text, CLASSES
from src.features import build_vectorizer
from src.predict import load_artifacts, predict_claim
from src.explain import explain_prediction


def test_clean_text():
    raw = "Breaking: Check https://example.com for <b>urgent</b> updates!!!"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "<b>" not in cleaned
    assert "urgent" in cleaned
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_load_dataset_splits():
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    assert len(train_df) > 9000
    assert len(val_df) > 1000
    assert len(test_df) > 1000

    for df in [train_df, val_df, test_df]:
        assert set(df["label"].unique()) == set(CLASSES)
        assert "cleaned_text" in df.columns
        assert df["cleaned_text"].str.len().gt(0).all()


def test_feature_extraction():
    vectorizer = build_vectorizer(max_features=500)
    samples = ["The economy showed signs of growth.", "Unemployment dropped this quarter."]
    X = vectorizer.fit_transform(samples)
    assert X.shape[0] == 2
    assert X.shape[1] > 0


def test_model_artifacts():
    artifacts = load_artifacts()
    assert "model" in artifacts
    assert "vectorizer" in artifacts
    assert "metadata" in artifacts

    model = artifacts["model"]
    assert list(model.classes_) == CLASSES
    assert hasattr(model, "predict_proba")


def test_prediction_output():
    artifacts = load_artifacts()
    claim = "The government plans to increase federal spending on education."
    result = predict_claim(claim, artifacts)

    assert result["prediction"] in CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert 0.0 <= result["real_probability"] <= 1.0
    assert np.isclose(result["fake_probability"] + result["real_probability"], 1.0)
    assert "explanation" in result


def test_explain_prediction():
    artifacts = load_artifacts()
    claim = "Wisconsin layoffs doubled this year according to state reports."
    explanation = explain_prediction(claim, artifacts["vectorizer"], artifacts["model"])

    assert "words" in explanation
    assert isinstance(explanation["words"], list)


def test_short_claim_error():
    artifacts = load_artifacts()
    with pytest.raises(ValueError):
        predict_claim("Hello", artifacts)
