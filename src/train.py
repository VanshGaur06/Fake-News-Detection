"""Train and evaluate models on the official LIAR train/validation/test splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
RANDOM_STATE = 42
N_SPLITS = 5
sys.path.insert(0, str(ROOT))
from src.dataset import DATA_DIR, load_splits
from src.evaluate import classification_metrics, cross_validation_summary
from src.features import make_vectorizer
from src.preprocessing import LIAR_LABELS, BINARY_INTERPRETATION


def make_pipeline(classifier) -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer()), ("classifier", classifier)])


def model_candidates() -> dict:
    return {
        "Naive Bayes": (make_pipeline(MultinomialNB()), {}),
        "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=2500, random_state=RANDOM_STATE)), {
            "classifier__C": [0.25, 1.0, 4.0], "classifier__class_weight": [None, "balanced"]}),
        "Linear SVM": (make_pipeline(LinearSVC(random_state=RANDOM_STATE)), {
            "classifier__C": [0.25, 1.0, 4.0], "classifier__class_weight": [None, "balanced"]}),
        "Random Forest": (make_pipeline(RandomForestClassifier(n_estimators=120, max_depth=35,
            min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=1, random_state=RANDOM_STATE)), {}),
    }


def tune_on_validation(name: str, pipeline: Pipeline, grid: dict,
                       train_text: np.ndarray, train_labels: np.ndarray,
                       validation_text: np.ndarray, validation_labels: np.ndarray) -> tuple[Pipeline, dict]:
    """Small validation search; each TF-IDF fit sees training statements only."""
    if not grid:
        fitted = clone(pipeline).fit(train_text, train_labels)
        return fitted, {}
    text = np.concatenate([train_text, validation_text])
    labels = np.concatenate([train_labels, validation_labels])
    fold_ids = np.concatenate([np.full(len(train_text), -1), np.zeros(len(validation_text))]).astype(int)
    search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=PredefinedSplit(fold_ids),
                          n_jobs=1, refit=False, error_score="raise")
    search.fit(text, labels)
    chosen = clone(pipeline).set_params(**search.best_params_).fit(train_text, train_labels)
    return chosen, search.best_params_


def cross_validate_candidates(candidates: dict, train_text: np.ndarray, train_labels: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}
    results = {}
    for name, estimator in candidates.items():
        fold_scores = cross_validate(estimator, train_text, train_labels, cv=cv, scoring=scoring,
                                     n_jobs=1, return_train_score=False, error_score="raise")
        results[name] = cross_validation_summary(fold_scores)
        print(f"{name} CV macro F1: {results[name]['mean_macro_f1']:.4f} "
              f"(±{results[name]['std_macro_f1']:.4f})", flush=True)
    return results


def evaluate_validation(splits: dict) -> tuple[dict, dict, dict]:
    train_text = splits["train"]["cleaned_statement"].to_numpy()
    train_labels = splits["train"]["label"].to_numpy()
    validation_text = splits["validation"]["cleaned_statement"].to_numpy()
    validation_labels = splits["validation"]["label"].to_numpy()
    configs = model_candidates()
    fitted_models, validation_metrics, chosen_params = {}, {}, {}
    for name, (estimator, grid) in configs.items():
        fitted, params = tune_on_validation(name, estimator, grid, train_text, train_labels,
                                            validation_text, validation_labels)
        fitted_models[name] = fitted
        chosen_params[name] = params
        validation_metrics[name] = classification_metrics(validation_labels, fitted.predict(validation_text))
        print(f"{name} validation macro F1: {validation_metrics[name]['macro_f1']:.4f}", flush=True)
    # Re-evaluate the selected hyperparameter configuration with five folds on training only.
    selected_candidates = {name: fitted_models[name] for name in configs}
    cv_metrics = cross_validate_candidates(selected_candidates, train_text, train_labels)
    return fitted_models, validation_metrics, {"best_params": chosen_params, "cross_validation": cv_metrics}


def train(data_dir: Path = DATA_DIR, output_dir: Path = MODEL_DIR) -> dict:
    splits, duplicate_summary = load_splits(data_dir)
    for name, frame in splits.items():
        if len(frame) < N_SPLITS or set(frame.label) != set(LIAR_LABELS):
            raise ValueError(f"The LIAR {name} split must contain all six labels and enough rows for validation.")
    fitted, validation_results, selection_meta = evaluate_validation(splits)
    winner = max(fitted, key=lambda name: (validation_results[name]["macro_f1"],
        validation_results[name]["accuracy"], validation_results[name]["weighted_f1"]))
    combined_text = np.concatenate([splits["train"]["cleaned_statement"].to_numpy(),
                                    splits["validation"]["cleaned_statement"].to_numpy()])
    combined_labels = np.concatenate([splits["train"]["label"].to_numpy(),
                                      splits["validation"]["label"].to_numpy()])
    selected_config = model_candidates()[winner][0]
    selected_config.set_params(**selection_meta["best_params"][winner])
    if winner == "Linear SVM":
        final_model = CalibratedClassifierCV(estimator=selected_config, method="sigmoid",
                                             cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                                             ensemble=False)
        final_model.fit(combined_text, combined_labels)
        explanation_pipeline = final_model.estimator_
    else:
        final_model = clone(selected_config).fit(combined_text, combined_labels)
        explanation_pipeline = final_model

    # The official test split is touched only after model and hyperparameter selection.
    test_text = splits["test"]["cleaned_statement"].to_numpy()
    test_labels = splits["test"]["label"].to_numpy()
    test_predictions = final_model.predict(test_text)
    metrics = classification_metrics(test_labels, test_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = explanation_pipeline.named_steps["tfidf"]
    explainer = explanation_pipeline.named_steps["classifier"]
    joblib.dump(final_model, output_dir / "final_model.joblib")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(explainer, output_dir / "explanation_model.joblib")
    metadata = {
        "model_name": winner, "class_labels": list(LIAR_LABELS),
        "binary_mapping": BINARY_INTERPRETATION,
        "test_metrics": metrics, "validation_metrics": validation_results,
        "cross_validation": selection_meta["cross_validation"],
        "best_params": selection_meta["best_params"],
        "dataset_dimensions": {name: int(len(frame)) for name, frame in splits.items()},
        "class_distribution": {name: {label: int(count) for label, count in frame.label.value_counts().reindex(LIAR_LABELS, fill_value=0).items()}
                                for name, frame in splits.items()},
        "duplicate_handling": duplicate_summary, "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS, "positive_binary_class": "FAKE / LOW TRUTHFULNESS",
        "confidence_method": "predict_proba; Linear SVM uses sigmoid calibration with training+validation only",
        "training_configuration": {"vectorizer": make_vectorizer().get_params(),
                                   "candidate_models": list(model_candidates())},
    }
    joblib.dump(metadata, output_dir / "metadata.joblib")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_ready = json.dumps(metadata, indent=2, default=lambda value: value.__name__ if isinstance(value, type) else str(value))
    (METRICS_DIR / "evaluation.json").write_text(json_ready, encoding="utf-8")
    print(f"Selected on validation: {winner} (macro F1 {validation_results[winner]['macro_f1']:.4f})")
    print(f"Untouched LIAR test accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    return metadata


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}")
