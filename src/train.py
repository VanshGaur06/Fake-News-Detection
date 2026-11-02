"""Train and evaluate fake news detection models on the LIAR dataset."""
import json
from pathlib import Path
import sys
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.dataset import DATA_DIR, CLASSES, load_all_splits
from src.features import build_vectorizer
from src.evaluate import compute_metrics

MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"


def get_candidate_models():
    """Return candidate classification models for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000),
    }


def train_and_evaluate(data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR):
    """Run full training pipeline: feature extraction, model comparison, and serialization."""
    print("Loading LIAR dataset splits...")
    splits, summary = load_all_splits(data_dir)

    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    print(f"Loaded: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    # Step 1: Fit TF-IDF on training text only (prevent data leakage)
    print("Vectorizing text with TF-IDF...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_val = vectorizer.transform(val_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Step 2: Compare candidate models on validation split
    print("\nComparing candidate models on validation set:")
    candidate_models = get_candidate_models()
    comparison_results = {}

    for name, clf in candidate_models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        comparison_results[name] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        }
        print(f" - {name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_macro']:.4f}")

    # Step 3: Train primary model (Logistic Regression for calibrated probabilities and explainability)
    primary_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    primary_model.fit(X_train, y_train)

    val_preds = primary_model.predict(X_val)
    val_metrics = compute_metrics(y_val, val_preds)

    # Step 4: Final evaluation on the held-out test set
    test_preds = primary_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)

    print("\nFinal Test Metrics (Logistic Regression):")
    print(f" - Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f" - Macro F1:      {test_metrics['f1_macro']:.4f}")
    print(f" - Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Step 5: Save artifacts and metadata
    model_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(primary_model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")

    metadata = {
        "dataset": "LIAR Benchmark",
        "task": "Binary Fake News Detection",
        "classes": CLASSES,
        "primary_model": "Logistic Regression",
        "dataset_summary": summary,
        "model_comparison": comparison_results,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(METRICS_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts successfully saved to '{model_dir}' and '{METRICS_DIR}'.")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
