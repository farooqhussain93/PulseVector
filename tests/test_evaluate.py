from __future__ import annotations

import numpy as np

from ml.evaluate import calculate_specificity, evaluate_predictions, get_positive_class_probability


class ProbabilityModel:
    def predict_proba(self, features):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


class NoProbabilityModel:
    pass


class InvalidProbabilityModel:
    def predict_proba(self, features):
        return np.array([[1.2, -0.2], [0.1, 0.9]])


def test_specificity_calculation() -> None:
    assert calculate_specificity([0, 0, 1, 1], [0, 1, 1, 1]) == 0.5


def test_specificity_handles_zero_negative_denominator() -> None:
    assert calculate_specificity([1, 1], [1, 0]) == 0.0


def test_evaluate_predictions_returns_required_metrics() -> None:
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert set(["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]).issubset(result)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["classification_report"]["Present"]["support"] == 2.0


def test_evaluate_predictions_without_probability_sets_none_auc() -> None:
    assert evaluate_predictions([0, 1], [0, 1])["roc_auc"] is None


def test_evaluate_predictions_invalid_probability_shape_sets_none_auc() -> None:
    result = evaluate_predictions([0, 1], [0, 1], [0.5])
    assert result["roc_auc"] is None


def test_probability_extraction_returns_positive_column() -> None:
    values = get_positive_class_probability(ProbabilityModel(), [[1], [2]])
    assert np.allclose(values, [0.2, 0.9])


def test_probability_extraction_handles_unsupported_or_invalid_models() -> None:
    assert get_positive_class_probability(NoProbabilityModel(), [[1]]) is None
    assert get_positive_class_probability(InvalidProbabilityModel(), [[1], [2]]) is None
