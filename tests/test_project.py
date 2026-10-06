from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.dataset import LIAR_COLUMNS, load_splits
from src.evaluate import classification_metrics
from src.explain import explain_prediction
from src.features import make_vectorizer
from src.predict import ModelNotTrainedError, load_artifacts, predict_statement
from src.preprocessing import BINARY_INTERPRETATION, LIAR_LABELS, binary_label, clean_text, normalize_liar_label
from src.train import make_pipeline, tune_on_validation
from sklearn.naive_bayes import MultinomialNB


@pytest.mark.parametrize("raw", LIAR_LABELS)
def test_liar_labels_normalize_and_binary_map(raw):
    assert normalize_liar_label(raw.upper()) == raw
    assert binary_label(raw) == BINARY_INTERPRETATION[raw]


def test_invalid_label_is_rejected():
    assert normalize_liar_label("unverified") is None


def test_cleaning_preserves_negation_and_numbers():
    assert clean_text("<b>NOT true!</b> 25% https://example.com") == "not true 25%"


class MemoryPath:
    """Tiny fake path for exercising split schemas without making training data."""
    def __init__(self, name="root", exists=True):
        self.name, self._exists = name, exists

    def __truediv__(self, child):
        return MemoryPath(child, self._exists)

    def exists(self):
        return self._exists

    def __str__(self):
        return self.name


def test_official_tsv_loading_and_duplicate_removal(monkeypatch):
    rows = []
    for index, label in enumerate(LIAR_LABELS):
        row = [f"id-{index}", label, f"statement {label} sample {index}", "economy, taxes",
               "speaker", "job", "state", "party", "0", "1", "2", "3", "4", "context"]
        rows.append(row)
    files = {"train.tsv": rows + [rows[0]], "valid.tsv": [[*row[:2], row[2] + " val", *row[3:]] for row in rows],
             "test.tsv": [[*row[:2], row[2] + " test", *row[3:]] for row in rows]}
    frames = {name: pd.DataFrame(split_rows, columns=LIAR_COLUMNS) for name, split_rows in files.items()}
    monkeypatch.setattr("src.dataset.pd.read_csv", lambda path, **kwargs: frames[path.name].copy())
    splits, duplicates = load_splits(MemoryPath())
    assert {name: len(frame) for name, frame in splits.items()} == {"train": 6, "validation": 6, "test": 6}
    assert duplicates["training_duplicates_removed"] == 1
    assert set(splits["train"].label) == set(LIAR_LABELS)


def test_missing_dataset_has_readable_error(monkeypatch):
    monkeypatch.setattr(MemoryPath, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError, match="LIAR"):
        load_splits(MemoryPath())


def make_training_examples():
    texts, labels = [], []
    terms = ["pants fire burning", "false claim denied", "barely true uncertain", "half true partial", "mostly true accurate", "true verified"]
    for label, phrase in zip(LIAR_LABELS, terms):
        for number in range(8):
            texts.append(f"{phrase} policy statement report evidence example {number}")
            labels.append(label)
    return np.asarray(texts), np.asarray(labels)


def test_tfidf_and_multiclass_training_and_probabilities():
    texts, labels = make_training_examples()
    model = make_pipeline(MultinomialNB()).fit(texts, labels)
    probabilities = model.predict_proba(["verified true statement report evidence"])[0]
    assert list(model.classes_) == sorted(LIAR_LABELS)
    assert np.isclose(probabilities.sum(), 1.0)
    assert make_vectorizer().ngram_range == (1, 2)


def test_validation_grid_search_fits_and_selects_parameters():
    texts, labels = make_training_examples()
    model, params = tune_on_validation("test", make_pipeline(MultinomialNB()),
        {"classifier__alpha": [0.5, 1.0]}, texts[:36], labels[:36], texts[36:], labels[36:])
    assert model.predict(texts[36:]).shape[0] == len(texts[36:])
    assert params["classifier__alpha"] in {0.5, 1.0}


def test_six_class_metrics_and_binary_metrics_are_separate():
    labels = list(LIAR_LABELS) * 2
    metrics = classification_metrics(labels, labels)
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 6
    assert metrics["binary_interpretation"]["macro_f1"] == 1.0
    assert len(metrics["binary_interpretation"]["confusion_matrix"]) == 2


def test_multiclass_linear_explanation_has_support_and_opposition():
    texts, labels = make_training_examples()
    vectorizer = make_vectorizer().fit(texts)
    matrix = vectorizer.transform(texts)
    model = LogisticRegression(max_iter=1000).fit(matrix, labels)
    result = explain_prediction(texts[0], vectorizer, model, "pants-fire")
    assert result["available"]
    assert result["competitor"] in LIAR_LABELS
    assert all(item["direction"] in {"supports", "pushes away"} for item in result["features"])


def test_empty_and_short_claim_inputs():
    with pytest.raises(ValueError):
        predict_statement("", {})
    with pytest.raises(ValueError):
        predict_statement("two words", {})


def test_six_class_prediction_probability_and_binary_result():
    class FixedProbabilityModel:
        classes_ = np.asarray(LIAR_LABELS)
        def predict(self, texts):
            return np.asarray(["half-true"])
        def predict_proba(self, texts):
            return np.asarray([[.03, .04, .08, .62, .15, .08]])

    vectorizer = make_vectorizer().set_params(max_df=1.0).fit(["the senator supported a new public policy"] * 4)
    result = predict_statement("The senator supported the new public policy", {
        "model": FixedProbabilityModel(), "vectorizer": vectorizer,
        "explainer": object(), "metadata": {"model_name": "Unit test"}})
    assert result["label"] == "half-true"
    assert result["binary_interpretation"] == "REAL / HIGHER TRUTHFULNESS"
    assert result["confidence"] == pytest.approx(.62)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_missing_model_artifacts():
    with pytest.raises(ModelNotTrainedError):
        load_artifacts(Path(".absent-model-artifacts"))
