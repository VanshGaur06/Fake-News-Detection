"""Shared evaluation helpers."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="FAKE", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="FAKE", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="FAKE", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"]).tolist(),
    }


def cross_validation_summary(scores: dict[str, Any]) -> dict[str, float]:
    return {"mean_accuracy": float(np.mean(scores["test_accuracy"])),
            "mean_precision": float(np.mean(scores["test_precision"])),
            "mean_recall": float(np.mean(scores["test_recall"])),
            "mean_f1": float(np.mean(scores["test_f1"])),
            "std_accuracy": float(np.std(scores["test_accuracy"])),
            "std_f1": float(np.std(scores["test_f1"]))}
