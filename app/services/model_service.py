"""Model loading, caching, and prediction service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.schemas.prediction import PredictionRequest, PredictionResponse
from ml.preprocessing import CLASS_MAPPING, FEATURE_COLUMNS


class ModelUnavailableError(RuntimeError):
    """Raised when a valid prediction model cannot be served."""


@lru_cache(maxsize=4)
def load_model(model_path: str) -> Any:
    """Load and cache a serialized model by resolved path."""

    path = Path(model_path)
    if not path.exists():
        raise ModelUnavailableError(f"Prediction model is unavailable: {path.name}")
    try:
        model = joblib.load(path)
    except Exception as exc:  # joblib may surface several corruption exceptions
        raise ModelUnavailableError("Prediction model could not be loaded.") from exc
    if not hasattr(model, "predict"):
        raise ModelUnavailableError("Prediction artifact is not a valid estimator.")
    return model


def clear_model_cache() -> None:
    load_model.cache_clear()


def _probability(model: Any, frame: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        values = np.asarray(model.predict_proba(frame), dtype=float)
    except Exception:
        return None
    if values.shape != (1, 2):
        return None
    positive = float(values[0, 1])
    if not np.isfinite(positive) or not 0 <= positive <= 1:
        return None
    return positive


def predict_patient(
    request: PredictionRequest,
    metrics: dict[str, Any],
    *,
    model_path: Path | None = None,
) -> PredictionResponse:
    """Run one validated prediction and attach verified model metadata."""

    selected_path = (model_path or settings.model_path).resolve()
    model = load_model(str(selected_path))
    record = request.to_model_record()
    frame = pd.DataFrame([[record[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    try:
        predicted_class = int(model.predict(frame)[0])
    except Exception as exc:
        raise ModelUnavailableError("Prediction could not be completed with the loaded model.") from exc
    if predicted_class not in {0, 1}:
        raise ModelUnavailableError("Model returned an unsupported class label.")

    probability = _probability(model, frame)
    selected = metrics.get("selection", {})
    held_out = metrics.get("held_out_test_metrics", {})
    winner_name = selected.get("winner_name")
    if not winner_name:
        raise ModelUnavailableError("Winning model metadata is unavailable.")

    label = (
        CLASS_MAPPING["positive_label"]
        if predicted_class == CLASS_MAPPING["positive_class"]
        else CLASS_MAPPING["negative_label"]
    )
    verified = {
        key: (float(held_out[key]) if held_out.get(key) is not None else None)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity")
    }
    return PredictionResponse(
        predicted_class=predicted_class,
        label=label,
        positive_class_probability=probability,
        probability_label=(
            "Model-estimated probability based on the supplied features."
            if probability is not None
            else None
        ),
        winning_model=str(winner_name),
        evaluation_metrics=verified,
        disclaimer=settings.educational_disclaimer,
    )
