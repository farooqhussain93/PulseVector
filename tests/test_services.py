from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from app.config import Settings
from app.main import _json_safe
from app.schemas.prediction import PredictionRequest
from app.services.model_service import (
    ModelUnavailableError,
    clear_model_cache,
    load_model,
    predict_patient,
)
from app.services.readiness_service import check_readiness
from app.services.report_service import (
    ReportUnavailableError,
    read_chart,
    read_dataset_summary,
    read_json,
    read_metrics,
)





def test_json_safe_sanitizes_validation_payloads() -> None:
    payload = {
        "finite": 1.5,
        "non_finite": [math.inf, -math.inf, math.nan],
        "error": ValueError("invalid"),
        "other": Path("sample"),
    }
    safe = _json_safe(payload)
    assert safe["finite"] == 1.5
    assert safe["non_finite"] == ["inf", "-inf", "nan"]
    assert safe["error"] == "invalid"
    assert safe["other"] == "sample"


def test_settings_paths_are_project_relative(tmp_path: Path) -> None:
    configured = Settings(project_root=tmp_path)
    assert configured.templates_dir == tmp_path / "templates"
    assert configured.static_dir == tmp_path / "static"
    assert configured.model_path == tmp_path / "models" / "best_model.joblib"
    assert configured.metrics_path == tmp_path / "reports" / "metrics.json"
    assert configured.dataset_summary_path == tmp_path / "reports" / "dataset_summary.json"
    assert configured.reports_dir == tmp_path / "reports"


class InvalidEstimator:
    pass


class BrokenPredictionModel:
    def predict(self, frame):
        raise RuntimeError("boom")


class InvalidClassModel:
    def predict(self, frame):
        return np.array([4])


class NoProbabilityModel:
    def predict(self, frame):
        return np.array([0])


def test_schema_rejects_extra_fields(valid_payload: dict[str, float | int]) -> None:
    with pytest.raises(Exception):
        PredictionRequest.model_validate({**valid_payload, "unexpected": 1})


def test_schema_rejects_invalid_ranges(valid_payload: dict[str, float | int]) -> None:
    with pytest.raises(Exception):
        PredictionRequest.model_validate({**valid_payload, "age": 10})
    with pytest.raises(Exception):
        PredictionRequest.model_validate({**valid_payload, "thal": 5})
    for non_finite in (math.inf, -math.inf, math.nan):
        with pytest.raises(Exception):
            PredictionRequest.model_validate({**valid_payload, "age": non_finite})


def test_schema_preserves_feature_order(valid_payload: dict[str, float | int]) -> None:
    request = PredictionRequest.model_validate(valid_payload)
    assert list(request.to_model_record()) == [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]


def test_read_json_and_specialized_reports(isolated_pipeline: dict[str, object]) -> None:
    reports = Path(isolated_pipeline["reports"])
    metrics = read_metrics(reports / "metrics.json")
    dataset = read_dataset_summary(reports / "dataset_summary.json")
    assert metrics["selection"]["winner_name"]
    assert dataset["rows"] == 303


def test_report_service_rejects_missing_malformed_and_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(ReportUnavailableError, match="unavailable"):
        read_json(tmp_path / "missing.json")
    malformed = tmp_path / "bad.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(ReportUnavailableError, match="malformed"):
        read_json(malformed)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ReportUnavailableError, match="missing fields"):
        read_json(incomplete, required_keys=("b",))


def test_read_chart_returns_only_valid_plotly_html(tmp_path: Path) -> None:
    assert read_chart(tmp_path, "missing.html") is None
    (tmp_path / "plain.html").write_text("<div>plain</div>", encoding="utf-8")
    assert read_chart(tmp_path, "plain.html") is None
    (tmp_path / "chart.html").write_text("<div class='plotly-graph-div'></div>", encoding="utf-8")
    assert "plotly" in read_chart(tmp_path, "chart.html")
    assert read_chart(tmp_path, "chart.txt") is None


def test_model_load_cache_and_clear(isolated_pipeline: dict[str, object]) -> None:
    clear_model_cache()
    path = str((Path(isolated_pipeline["models"]) / "best_model.joblib").resolve())
    first = load_model(path)
    second = load_model(path)
    assert first is second
    clear_model_cache()
    third = load_model(path)
    assert third is not first


def test_model_load_rejects_missing_corrupt_and_invalid(tmp_path: Path) -> None:
    clear_model_cache()
    with pytest.raises(ModelUnavailableError, match="unavailable"):
        load_model(str(tmp_path / "missing.joblib"))
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"bad")
    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        load_model(str(corrupt))
    invalid = tmp_path / "invalid.joblib"
    joblib.dump(InvalidEstimator(), invalid)
    with pytest.raises(ModelUnavailableError, match="valid estimator"):
        load_model(str(invalid))


def test_predict_patient_returns_verified_metadata(
    isolated_pipeline: dict[str, object], valid_payload: dict[str, float | int]
) -> None:
    reports = Path(isolated_pipeline["reports"])
    model_path = Path(isolated_pipeline["models"]) / "best_model.joblib"
    metrics = read_metrics(reports / "metrics.json")
    response = predict_patient(PredictionRequest.model_validate(valid_payload), metrics, model_path=model_path)
    assert response.predicted_class in {0, 1}
    assert response.winning_model == metrics["selection"]["winner_name"]
    assert response.positive_class_probability is not None
    assert 0 <= response.positive_class_probability <= 1
    assert response.evaluation_metrics["f1"] == metrics["held_out_test_metrics"]["f1"]


def test_predict_patient_handles_no_probability(tmp_path: Path, valid_payload: dict[str, float | int]) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(NoProbabilityModel(), path)
    metrics = {"selection": {"winner_name": "No Probability"}, "held_out_test_metrics": {}}
    response = predict_patient(PredictionRequest.model_validate(valid_payload), metrics, model_path=path)
    assert response.positive_class_probability is None
    assert response.probability_label is None


def test_predict_patient_rejects_bad_predictions_and_missing_metadata(tmp_path: Path, valid_payload: dict[str, float | int]) -> None:
    request = PredictionRequest.model_validate(valid_payload)
    for model, message in [(BrokenPredictionModel(), "could not be completed"), (InvalidClassModel(), "unsupported class")]:
        path = tmp_path / f"{model.__class__.__name__}.joblib"
        joblib.dump(model, path)
        with pytest.raises(ModelUnavailableError, match=message):
            predict_patient(request, {"selection": {"winner_name": "X"}, "held_out_test_metrics": {}}, model_path=path)
    path = tmp_path / "no_probability.joblib"
    joblib.dump(NoProbabilityModel(), path)
    with pytest.raises(ModelUnavailableError, match="metadata"):
        predict_patient(request, {"selection": {}, "held_out_test_metrics": {}}, model_path=path)


def test_readiness_true_for_isolated_artifacts(isolated_pipeline: dict[str, object]) -> None:
    ready, payload = check_readiness(
        model_path=Path(isolated_pipeline["models"]) / "best_model.joblib",
        metrics_path=Path(isolated_pipeline["reports"]) / "metrics.json",
    )
    assert ready is True
    assert all(payload["checks"].values())
    assert payload["errors"] == []


def test_readiness_false_for_missing_model_and_metrics(tmp_path: Path) -> None:
    clear_model_cache()
    ready, payload = check_readiness(model_path=tmp_path / "missing.joblib", metrics_path=tmp_path / "missing.json")
    assert ready is False
    assert payload["checks"]["model"] is False
    assert payload["checks"]["metrics"] is False
    assert len(payload["errors"]) == 2

def test_readiness_rejects_missing_winner_metadata(
    tmp_path: Path, isolated_pipeline: dict[str, object]
) -> None:
    metrics = json.loads(
        (Path(isolated_pipeline["reports"]) / "metrics.json").read_text(encoding="utf-8")
    )
    metrics["selection"]["winner_name"] = ""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    ready, payload = check_readiness(
        model_path=Path(isolated_pipeline["models"]) / "best_model.joblib",
        metrics_path=metrics_path,
    )

    assert ready is False
    assert payload["checks"]["winner_metadata"] is False
    assert payload["checks"]["prediction_probe"] is False
    assert "Winning model metadata is unavailable." in payload["errors"]


def test_readiness_rejects_inconsistent_feature_schema(
    tmp_path: Path, isolated_pipeline: dict[str, object]
) -> None:
    metrics = json.loads(
        (Path(isolated_pipeline["reports"]) / "metrics.json").read_text(encoding="utf-8")
    )
    metrics["feature_list"] = list(reversed(metrics["feature_list"]))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    ready, payload = check_readiness(
        model_path=Path(isolated_pipeline["models"]) / "best_model.joblib",
        metrics_path=metrics_path,
    )

    assert ready is False
    assert payload["checks"]["feature_schema"] is False
    assert payload["checks"]["prediction_probe"] is False

def test_readiness_rejects_failed_prediction_probe(
    isolated_pipeline: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_probe(*args, **kwargs):
        raise ModelUnavailableError("probe unavailable")

    monkeypatch.setattr("app.services.readiness_service.predict_patient", fail_probe)
    ready, payload = check_readiness(
        model_path=Path(isolated_pipeline["models"]) / "best_model.joblib",
        metrics_path=Path(isolated_pipeline["reports"]) / "metrics.json",
    )
    assert ready is False
    assert payload["checks"]["prediction_probe"] is False
    assert any("probe unavailable" in error for error in payload["errors"])


def test_readiness_hides_unexpected_probe_failure(
    isolated_pipeline: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_probe(*args, **kwargs):
        raise RuntimeError("sensitive probe detail")

    monkeypatch.setattr("app.services.readiness_service.predict_patient", fail_probe)
    ready, payload = check_readiness(
        model_path=Path(isolated_pipeline["models"]) / "best_model.joblib",
        metrics_path=Path(isolated_pipeline["reports"]) / "metrics.json",
    )
    assert ready is False
    assert payload["checks"]["prediction_probe"] is False
    assert payload["errors"] == ["Prediction readiness probe failed unexpectedly."]
    assert "sensitive" not in str(payload)

