from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text, normalize_label, combine_article
from src.predict import predict_article, load_artifacts, ModelNotTrainedError
from src.explain import explain_prediction


def test_text_preprocessing_removes_markup_urls_and_keeps_negation():
    assert clean_text("<b>NOT true!</b> Visit https://example.com") == "not true visit"


@pytest.mark.parametrize("raw,expected", [("Fake", "FAKE"), ("0", "FAKE"), ("Real", "REAL"), (1, "REAL"), ("unknown", None)])
def test_label_normalization(raw, expected):
    assert normalize_label(raw) == expected


def test_combine_title_and_body():
    assert combine_article("Headline", "Body") == "Headline Body"


def test_prediction_empty_and_short_inputs():
    with pytest.raises(ValueError):
        predict_article("  ", {})
    with pytest.raises(ValueError):
        predict_article("Only four useful words", {})


def test_model_loading_reports_missing_artifacts():
    with pytest.raises(ModelNotTrainedError):
        load_artifacts(Path.cwd() / ".missing-test-artifacts")


def test_explanation_contributions_and_direction():
    texts = ["fake scandal secret", "real official report", "fake scandal", "real official"]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(matrix, ["FAKE", "REAL", "FAKE", "REAL"])
    result = explain_prediction("fake scandal", vectorizer, model)
    assert result["available"]
    assert result["features"]
    assert all(feature["direction"] in {"FAKE", "REAL"} for feature in result["features"])


def test_confidence_uses_actual_probability():
    from sklearn.naive_bayes import MultinomialNB
    texts = ["fake scandal secret report", "real official report statement", "fake rumor scandal", "real public official news"]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = MultinomialNB().fit(matrix, ["FAKE", "REAL", "FAKE", "REAL"])
    artifacts = {"vectorizer": vectorizer, "model": model, "metadata": {"model_name": "Naive Bayes", "explainer_model": model}}
    result = predict_article("fake scandal secret report official", artifacts)
    assert 0 <= result["confidence"] <= 1
    cleaned = clean_text("fake scandal secret report official")
    predicted = model.predict(vectorizer.transform([cleaned]))[0]
    expected = model.predict_proba(vectorizer.transform([cleaned]))[0][list(model.classes_).index(predicted)]
    assert np.isclose(result["confidence"], expected)
