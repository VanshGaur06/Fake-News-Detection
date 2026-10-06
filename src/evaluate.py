"""Six-class LIAR and project-level binary evaluation helpers."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score)

from src.preprocessing import LIAR_LABELS, binary_label


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=list(LIAR_LABELS),
        output_dict=True, zero_division=0)
    binary_true = [binary_label(label) for label in y_true]
    binary_pred = [binary_label(label) for label in y_pred]
    binary_labels = ["FAKE / LOW TRUTHFULNESS", "REAL / HIGHER TRUTHFULNESS"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(LIAR_LABELS), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(LIAR_LABELS), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LIAR_LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LIAR_LABELS), average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LIAR_LABELS)).tolist(),
        "classification_report": {label: {key: float(value) for key, value in values.items()}
                                  for label, values in report.items() if isinstance(values, dict)},
        "binary_interpretation": {
            "accuracy": float(accuracy_score(binary_true, binary_pred)),
            "macro_precision": float(precision_score(binary_true, binary_pred, labels=binary_labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(binary_true, binary_pred, labels=binary_labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(binary_true, binary_pred, labels=binary_labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(binary_true, binary_pred, labels=binary_labels, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(binary_true, binary_pred, labels=binary_labels).tolist(),
            "labels": binary_labels,
        },
    }


def cross_validation_summary(scores: dict[str, Any]) -> dict[str, float]:
    return {key: float(np.mean(scores[score_key])) for key, score_key in {
        "mean_accuracy": "test_accuracy", "mean_macro_f1": "test_macro_f1",
        "mean_weighted_f1": "test_weighted_f1"}.items()} | {
        "std_accuracy": float(np.std(scores["test_accuracy"])),
        "std_macro_f1": float(np.std(scores["test_macro_f1"]))}
