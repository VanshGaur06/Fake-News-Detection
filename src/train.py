"""Train, compare, and persist leakage-safe TruthLens models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "dataset.csv"
LABEL_COLUMN = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20

sys.path.insert(0, str(ROOT))
from src.evaluate import classification_metrics, cross_validation_summary
from src.features import make_vectorizer
from src.preprocessing import clean_text, combine_article, normalize_label


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place a labeled CSV at data/dataset.csv.")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV: {exc}") from exc
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if LABEL_COLUMN not in frame.columns:
        raise ValueError("Dataset needs a 'label' column (accepted labels: Fake/Real or 0/1).")
    if "text" not in frame.columns and "title" not in frame.columns:
        raise ValueError("Dataset needs a 'text' and/or 'title' column containing article content.")
    title = frame["title"] if "title" in frame else pd.Series([""] * len(frame), index=frame.index)
    body = frame["text"] if "text" in frame else pd.Series([""] * len(frame), index=frame.index)
    frame["article"] = [clean_text(combine_article(t, b)) for t, b in zip(title, body)]
    frame["target"] = frame[LABEL_COLUMN].map(normalize_label)
    invalid = int(frame["target"].isna().sum())
    frame = frame[frame["target"].notna() & frame["article"].str.strip().ne("")].copy()
    if len(frame) < 14:
        raise ValueError("At least 14 usable labeled articles are required for a stratified 80/20 split and 5-fold validation.")
    counts = frame["target"].value_counts()
    if set(counts.index) != {"FAKE", "REAL"} or counts.min() < 7:
        raise ValueError("Dataset needs at least seven examples of each class (FAKE and REAL) for stratified training and 5-fold validation.")
    frame = frame.drop_duplicates(subset="article", keep="first")
    counts = frame["target"].value_counts()
    if len(frame) < 14 or set(counts.index) != {"FAKE", "REAL"} or counts.min() < 7:
        raise ValueError("After removing exact duplicate articles, at least 14 records and seven per class are required.")
    return frame.reset_index(drop=True), str(invalid)


def pipeline(classifier) -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer()), ("classifier", classifier)])


def train(path: Path = DATASET_PATH, output_dir: Path = ROOT / "models") -> dict:
    frame, invalid_count = load_dataset(path)
    X, y = frame.article, frame.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    configs = {
        "Naive Bayes": (pipeline(MultinomialNB()), {}),
        "Logistic Regression": (pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                                {"classifier__C": [0.1, 1.0, 5.0]}),
        "Linear SVM": (pipeline(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
                       {"classifier__C": [0.1, 1.0, 5.0]}),
        "Random Forest": (pipeline(RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                           n_jobs=1, random_state=RANDOM_STATE)), {}),
    }
    results, fitted, cv_stats = {}, {}, {}
    for name, (estimator, grid) in configs.items():
        if grid:
            search = GridSearchCV(estimator, grid, scoring=make_scorer(f1_score, pos_label="FAKE"),
                                  cv=folds, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
        else:
            best = clone(estimator).fit(X_train, y_train)
        positive_scorer = make_scorer(f1_score, pos_label="FAKE")
        from sklearn.metrics import precision_score, recall_score
        cv = cross_validate(best, X_train, y_train, cv=folds, scoring={
            "accuracy": "accuracy", "f1": positive_scorer,
            "precision": make_scorer(precision_score, pos_label="FAKE"),
            "recall": make_scorer(recall_score, pos_label="FAKE")}, n_jobs=-1)
        cv_stats[name] = cross_validation_summary({"test_accuracy": cv["test_accuracy"],
            "test_f1": cv["test_f1"], "test_precision": cv["test_precision"], "test_recall": cv["test_recall"]})
        metrics = classification_metrics(y_test, best.predict(X_test))
        results[name] = {**metrics, "best_params": (search.best_params_ if grid else {})}
        fitted[name] = best
        print(f"{name}: F1={metrics['f1']:.3f}, accuracy={metrics['accuracy']:.3f}")
    baseline = y_train.value_counts(normalize=True).max()
    # Model selection uses training-only CV; the held-out test set is only for final reporting.
    winner = max(results, key=lambda name: (cv_stats[name]["mean_f1"],
        cv_stats[name]["mean_precision"], cv_stats[name]["mean_recall"], cv_stats[name]["mean_accuracy"]))
    # Refit only on training data; test data remains held out for the reported metrics.
    best_pipeline = fitted[winner]
    vectorizer = best_pipeline.named_steps["tfidf"]
    classifier = best_pipeline.named_steps["classifier"]
    X_train_vec = vectorizer.transform(X_train)
    if winner == "Linear SVM":
        explainer_model = classifier
        final_model = CalibratedClassifierCV(estimator=clone(classifier), method="sigmoid", cv=folds, ensemble=False)
        final_model.fit(X_train_vec, y_train)
    elif hasattr(classifier, "predict_proba"):
        final_model, explainer_model = classifier, classifier
    else:
        explainer_model = classifier
        final_model = CalibratedClassifierCV(estimator=clone(classifier), method="sigmoid", cv=folds,
                                             ensemble=False).fit(X_train_vec, y_train)
    # Report the deployed estimator's test metrics, including calibrated decisions where applicable.
    results[winner] = {**classification_metrics(y_test, final_model.predict(vectorizer.transform(X_test))),
                       "best_params": results[winner]["best_params"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_dir / "final_model.joblib")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    metadata = {"model_name": winner, "metrics": results[winner], "all_models": results,
                "cross_validation": cv_stats, "baseline_accuracy": float(baseline),
                "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                "dataset_rows_after_cleaning": len(frame), "invalid_labels_removed": int(invalid_count),
                "positive_class": "FAKE", "confidence_method": "predict_proba; Linear SVM uses sigmoid calibration fitted on training data",
                "explainer_model": explainer_model}
    joblib.dump(metadata, output_dir / "metadata.joblib")
    # Separate JSON keeps the UI and human-readable output independent of the model binary.
    public_metadata = {k: v for k, v in metadata.items() if k != "explainer_model"}
    metrics_dir = ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation.json").write_text(json.dumps(public_metadata, indent=2), encoding="utf-8")
    print(f"Selected model: {winner}; held-out test metrics: {results[winner]}")
    return metadata


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}")
