from __future__ import annotations

import json
import math

from app.services.model_service import ModelUnavailableError


def test_health_is_liveness_only(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "PulseVector"}


def test_ready_reports_real_artifact_checks(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["checks"]["model"] is True
    assert response.json()["checks"]["winner_metadata"] is True
    assert response.json()["checks"]["prediction_probe"] is True


def test_all_page_routes_render(client) -> None:
    expected = {
        "/": "Classification engineering",
        "/dataset": "One approved source",
        "/eda": "Patterns in the dataset",
        "/models": "Five candidates",
        "/predict": "Explore the model",
        "/about": "complete classification system",
    }
    for route, text in expected.items():
        response = client.get(route)
        assert response.status_code == 200
        assert text in response.text
        assert "not a medical diagnostic tool" in response.text


def test_plotly_library_loads_before_inline_chart_scripts(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<script src="http://testserver/static/js/plotly.min.js"></script>' in response.text
    assert '<script defer src="http://testserver/static/js/plotly.min.js"></script>' not in response.text


def test_openapi_docs_available(client) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/predict" in schema["paths"]
    assert schema["info"]["title"] == "PulseVector API"


def test_valid_json_prediction(client, valid_payload) -> None:
    response = client.post("/api/predict", json=valid_payload)
    body = response.json()
    assert response.status_code == 200
    assert body["predicted_class"] in {0, 1}
    assert body["winning_model"]
    assert body["evaluation_metrics"]["f1"] > 0
    assert 0 <= body["positive_class_probability"] <= 1


def test_invalid_json_prediction_returns_422(client, valid_payload) -> None:
    response = client.post("/api/predict", json={**valid_payload, "age": 5})
    assert response.status_code == 422


def test_non_finite_json_values_return_controlled_422(client, valid_payload) -> None:
    for non_finite in (math.inf, math.nan):
        payload = {**valid_payload, "age": non_finite}
        response = client.post(
            "/api/predict",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        assert isinstance(response.json()["detail"], list)
        assert "Internal Server Error" not in response.text


def test_missing_json_field_returns_422(client, valid_payload) -> None:
    valid_payload.pop("thal")
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 422


def test_extra_json_field_returns_422(client, valid_payload) -> None:
    response = client.post("/api/predict", json={**valid_payload, "extra": 1})
    assert response.status_code == 422


def test_valid_html_form_prediction(client, valid_payload, isolated_pipeline) -> None:
    response = client.post("/prediction/", data=valid_payload)
    assert response.status_code == 200
    assert "Prediction result" in response.text
    assert "Positive-class probability" in response.text
    assert isolated_pipeline["metrics"]["selection"]["winner_name"] in response.text


def test_invalid_html_form_prediction_returns_422(client, valid_payload) -> None:
    response = client.post("/prediction/", data={**valid_payload, "age": "invalid"})
    assert response.status_code == 422
    assert "Check the submitted values" in response.text


def test_api_service_unavailable_is_controlled(client, valid_payload, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise ModelUnavailableError("Prediction model is unavailable")
    monkeypatch.setattr("app.routes.prediction.predict_patient", fail)
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction model is unavailable"


def test_html_service_unavailable_is_controlled(client, valid_payload, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise ModelUnavailableError("Prediction model is unavailable")
    monkeypatch.setattr("app.routes.prediction.predict_patient", fail)
    response = client.post("/prediction/", data=valid_payload)
    assert response.status_code == 503
    assert "Prediction model is unavailable" in response.text


def test_api_unexpected_failure_hides_stack_trace(client, valid_payload, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive stack detail")
    monkeypatch.setattr("app.routes.prediction.predict_patient", fail)
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed unexpectedly."
    assert "sensitive" not in response.text


def test_dashboard_status_uses_readiness_result(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.routes.pages.check_readiness",
        lambda: (False, {"ready": False, "checks": {"model": False, "metrics": True}, "errors": ["model missing"]}),
    )
    response = client.get("/")
    assert response.status_code == 200
    assert "Artifacts unavailable" in response.text
    assert "Prediction ready" not in response.text


def test_ready_returns_503_when_not_ready(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.routes.pages.check_readiness",
        lambda: (False, {"ready": False, "checks": {"model": False}, "errors": ["missing"]}),
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
