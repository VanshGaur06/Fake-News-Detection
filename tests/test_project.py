from pathlib import Path
import numpy as np
import pytest

from src.dataset import DATA_PATH, LABELS, load_dataset, stratified_splits
from src.predict import load_artifacts, predict_article


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="session")
def heldout(dataset):
    frame, _ = dataset
    return stratified_splits(frame)


def test_welfake_csv_columns_rows_and_binary_mapping(dataset):
    frame, summary = dataset
    assert summary["source_rows"] == 72134
    assert set(("title", "text", "article", "label")).issubset(frame.columns)
    assert set(frame["label"].unique()) == set(LABELS)
    assert summary["class_distribution"]["FAKE"] > 0
    assert summary["class_distribution"]["REAL"] > 0
    assert frame["article"].str.strip().ne("").all()


def test_stratified_splits_are_disjoint_and_approximately_80_10_10(heldout):
    train, validation, test = (heldout[key] for key in ("train", "validation", "test"))
    total = len(train) + len(validation) + len(test)
    assert abs(len(train) / total - 0.8) < 0.002
    assert abs(len(validation) / total - 0.1) < 0.002
    assert abs(len(test) / total - 0.1) < 0.002
    train_texts = set(train.article.str.casefold())
    validation_texts = set(validation.article.str.casefold())
    test_texts = set(test.article.str.casefold())
    assert train_texts.isdisjoint(validation_texts)
    assert train_texts.isdisjoint(test_texts)
    assert validation_texts.isdisjoint(test_texts)
    for split in (train, validation, test):
        assert set(split.label.unique()) == set(LABELS)


def test_saved_model_is_welfake_logistic_regression():
    artifacts = load_artifacts()
    assert list(artifacts["model"].classes_) == ["FAKE", "REAL"]
    assert artifacts["metadata"]["dataset"] == "WELFake"
    assert artifacts["metadata"]["model_name"] == "Logistic Regression"
    assert artifacts["metadata"]["leakage_controls"]["vectorizer_fit_rows"] == artifacts["metadata"]["split_sizes"]["train"]
    assert artifacts["metadata"]["leakage_controls"]["validation_and_test_used_for_fit"] is False


@pytest.mark.parametrize("label", LABELS)
def test_inference_on_real_heldout_article_per_class(label, heldout):
    artifacts = load_artifacts()
    sample = heldout["test"].loc[heldout["test"].label.eq(label)].iloc[0]
    result = predict_article(sample.title, sample.text, artifacts)
    assert result["label"] in LABELS
    assert set(result["probabilities"]) == set(LABELS)
    assert np.isclose(sum(result["probabilities"].values()), 1.0)
    assert 0 <= result["fake_probability"] <= 1
    assert 0 <= result["real_probability"] <= 1
    assert result["model_name"] == "Logistic Regression"
    assert result["explanation"]["available"]


def test_missing_artifacts_are_not_replaced_by_old_liar_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="WELFake"):
        load_artifacts(tmp_path)
