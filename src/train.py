"""Train and evaluate the production WELFake TF-IDF + Logistic Regression model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
RANDOM_STATE = 42
LABELS = ["FAKE", "REAL"]

from src.dataset import DATA_PATH, LABELS as DATA_LABELS, load_dataset, stratified_splits
from src.features import make_vectorizer
from src.preprocessing import clean_text


def evaluate(model, matrix, labels) -> dict:
    predictions = model.predict(matrix)
    report = classification_report(labels, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        "classification_report": report,
    }


def train(data_path: Path = DATA_PATH, output_dir: Path = MODEL_DIR) -> dict:
    frame, data_summary = load_dataset(data_path)
    if tuple(DATA_LABELS) != tuple(LABELS):
        raise RuntimeError("Dataset class order must be FAKE, REAL.")
    splits = stratified_splits(frame, RANDOM_STATE)
    clean_splits = {}
    for name, part in splits.items():
        clean_splits[name] = {
            "text": part["article"].map(clean_text).to_numpy(),
            "label": part["label"].to_numpy(),
        }
        if len(clean_splits[name]["text"]) != len(part):
            raise RuntimeError(f"Unexpected row change while preparing {name} split.")

    # Fit TF-IDF and Logistic Regression on training rows only.
    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(clean_splits["train"]["text"])
    y_train = clean_splits["train"]["label"]
    model = LogisticRegression(max_iter=1500, solver="liblinear", random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    if list(model.classes_) != LABELS:
        raise RuntimeError(f"Expected model classes {LABELS}, got {model.classes_.tolist()}.")

    validation_x = vectorizer.transform(clean_splits["validation"]["text"])
    test_x = vectorizer.transform(clean_splits["test"]["text"])
    validation_metrics = evaluate(model, validation_x, clean_splits["validation"]["label"])
    test_metrics = evaluate(model, test_x, clean_splits["test"]["label"])
    split_sizes = {name: int(len(part)) for name, part in splits.items()}
    train_distribution = {label: int(count) for label, count in splits["train"]["label"].value_counts().reindex(LABELS, fill_value=0).items()}
    metadata = {
        "task": "binary_fake_news", "dataset": "WELFake", "dataset_path": str(data_path),
        "model_name": "Logistic Regression", "class_labels": LABELS,
        "source_label_mapping": {"0": "FAKE", "1": "REAL"},
        "random_state": RANDOM_STATE, "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        **data_summary, "split_sizes": split_sizes, "train_class_distribution": train_distribution,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "vectorizer_configuration": {"max_features": vectorizer.max_features,
            "ngram_range": list(vectorizer.ngram_range), "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df},
        "leakage_controls": {
            "exact_article_duplicates_removed_before_split": True,
            "conflicting_duplicate_article_groups_removed": True,
            "stratified_splits": True,
            "vectorizer_fit_rows": split_sizes["train"],
            "classifier_fit_rows": split_sizes["train"],
            "validation_and_test_used_for_fit": False,
            "subject_or_date_features_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fake_news_model.joblib")
    joblib.dump(vectorizer, output_dir / "fake_news_vectorizer.joblib")
    (output_dir / "fake_news_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total = data_summary["total_samples"]
    distribution = data_summary["class_distribution"]
    print(f"Dataset: WELFake ({data_path})")
    print(f"Total samples: {total:,}")
    print(f"FAKE samples: {distribution['FAKE']:,}")
    print(f"REAL samples: {distribution['REAL']:,}")
    print(f"Train/validation/test sizes: {split_sizes['train']:,} / {split_sizes['validation']:,} / {split_sizes['test']:,}")
    print(f"model.classes_: {model.classes_.tolist()}")
    for split_name, scores in (("Validation", validation_metrics), ("Test", test_metrics)):
        print(f"\n{split_name} metrics")
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Precision (macro): {scores['precision_macro']:.4f}")
        print(f"Recall (macro): {scores['recall_macro']:.4f}")
        print(f"F1 (macro): {scores['f1_macro']:.4f}")
        print("Confusion matrix [FAKE, REAL]:")
        print(scores["confusion_matrix"])
        print("Classification report:")
        split_key = split_name.lower()
        print(classification_report(clean_splits[split_key]["label"],
            model.predict(vectorizer.transform(clean_splits[split_key]["text"])),
            labels=LABELS, target_names=LABELS, zero_division=0))
    print("\nLeakage checks: exact duplicates removed before split; no article overlaps across splits; "
          "TF-IDF and Logistic Regression fit on training rows only; subject/date excluded.")
    return metadata


if __name__ == "__main__":
    train()
