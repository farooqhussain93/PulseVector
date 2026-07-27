"""Classification evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_positive_class_probability(model: Any, features: Any) -> np.ndarray | None:
    """Return positive-class probabilities when the estimator exposes valid values."""

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    positive = probabilities[:, 1]
    if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
        return None
    return positive


def calculate_specificity(y_true: Any, y_pred: Any) -> float:
    """Calculate true-negative rate for binary predictions."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_probability: Any | None = None,
) -> dict[str, Any]:
    """Calculate the project evaluation contract for one model."""

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": calculate_specificity(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Not detected", "Present"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_probability is not None:
        probability = np.asarray(y_probability, dtype=float)
        if probability.shape[0] == len(y_true) and np.isfinite(probability).all():
            metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        else:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics
