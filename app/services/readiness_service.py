"""Honest application readiness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.schemas.prediction import PredictionRequest
from app.services.model_service import ModelUnavailableError, load_model, predict_patient
from app.services.report_service import ReportUnavailableError, read_metrics
from ml.preprocessing import FEATURE_COLUMNS

_READINESS_PROBE = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def check_readiness(
    *,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify artifacts, metadata, and one real end-to-end prediction."""

    selected_model = (model_path or settings.model_path).resolve()
    selected_metrics = (metrics_path or settings.metrics_path).resolve()
    checks = {
        "model": False,
        "metrics": False,
        "class_mapping": False,
        "feature_schema": False,
        "winner_metadata": False,
        "prediction_probe": False,
    }
    errors: list[str] = []
    metrics: dict[str, Any] | None = None

    try:
        load_model(str(selected_model))
        checks["model"] = True
    except ModelUnavailableError as exc:
        errors.append(str(exc))

    try:
        metrics = read_metrics(selected_metrics)
        checks["metrics"] = True

        class_mapping = metrics.get("class_mapping", {})
        checks["class_mapping"] = {
            class_mapping.get("negative_class"), class_mapping.get("positive_class")
        } == {0, 1}

        checks["feature_schema"] = list(metrics.get("feature_list", [])) == FEATURE_COLUMNS

        winner_name = metrics.get("selection", {}).get("winner_name")
        checks["winner_metadata"] = isinstance(winner_name, str) and bool(winner_name.strip())

        if not checks["class_mapping"]:
            errors.append("Class mapping is incomplete or invalid.")
        if not checks["feature_schema"]:
            errors.append("Feature schema metadata is missing or inconsistent.")
        if not checks["winner_metadata"]:
            errors.append("Winning model metadata is unavailable.")
    except ReportUnavailableError as exc:
        errors.append(str(exc))

    prerequisites = all(
        checks[key]
        for key in ("model", "metrics", "class_mapping", "feature_schema", "winner_metadata")
    )
    if prerequisites and metrics is not None:
        try:
            probe = PredictionRequest.model_validate(_READINESS_PROBE)
            result = predict_patient(probe, metrics, model_path=selected_model)
            checks["prediction_probe"] = result.predicted_class in {0, 1}
            if not checks["prediction_probe"]:
                errors.append("Prediction readiness probe returned an invalid class.")
        except (ValidationError, ModelUnavailableError) as exc:
            errors.append(f"Prediction readiness probe failed: {exc}")
        except Exception:
            errors.append("Prediction readiness probe failed unexpectedly.")

    ready = all(checks.values())
    return ready, {"ready": ready, "checks": checks, "errors": errors}
