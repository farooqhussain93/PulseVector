"""Jinja2 page routes and operational endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings
from app.services.data_service import FEATURE_DESCRIPTIONS, FORM_OPTIONS
from app.services.readiness_service import check_readiness
from app.services.report_service import (
    ReportUnavailableError,
    read_chart,
    read_dataset_summary,
    read_metrics,
)

router = APIRouter()


def _safe_reports() -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    metrics: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    try:
        metrics = read_metrics(settings.metrics_path)
    except ReportUnavailableError as exc:
        errors.append(str(exc))
    try:
        dataset = read_dataset_summary(settings.dataset_summary_path)
    except ReportUnavailableError as exc:
        errors.append(str(exc))
    return metrics, dataset, errors


def _render(request: Request, template: str, context: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    base = {
        "request": request,
        "project_name": settings.project_name,
        "disclaimer": settings.educational_disclaimer,
    }
    base.update(context)
    return request.app.state.templates.TemplateResponse(request, template, base, status_code)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.project_name}


@router.get("/ready")
def ready() -> JSONResponse:
    is_ready, payload = check_readiness()
    return JSONResponse(payload, status_code=200 if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE)


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    metrics, dataset, errors = _safe_reports()
    is_ready, readiness = check_readiness()
    return _render(
        request,
        "index.html",
        {
            "page": "dashboard",
            "metrics": metrics,
            "dataset": dataset,
            "report_errors": errors,
            "readiness": readiness,
            "is_ready": is_ready,
            "class_chart": read_chart(settings.reports_dir, "class_distribution.html"),
        },
    )


@router.get("/dataset", response_class=HTMLResponse)
def dataset_page(request: Request) -> HTMLResponse:
    _, dataset, errors = _safe_reports()
    return _render(
        request,
        "dataset.html",
        {
            "page": "dataset",
            "dataset": dataset,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "report_errors": errors,
            "class_chart": read_chart(settings.reports_dir, "class_distribution.html"),
        },
    )


@router.get("/eda", response_class=HTMLResponse)
def eda_page(request: Request) -> HTMLResponse:
    return _render(
        request,
        "eda.html",
        {
            "page": "eda",
            "feature_chart": read_chart(settings.reports_dir, "feature_distributions.html"),
            "correlation_chart": read_chart(settings.reports_dir, "correlation_heatmap.html"),
            "probability_chart": read_chart(settings.reports_dir, "probability_distribution.html"),
        },
    )


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request) -> HTMLResponse:
    metrics, _, errors = _safe_reports()
    return _render(
        request,
        "models.html",
        {
            "page": "models",
            "metrics": metrics,
            "report_errors": errors,
            "comparison_chart": read_chart(settings.reports_dir, "model_comparison.html"),
            "confusion_chart": read_chart(settings.reports_dir, "confusion_matrices.html"),
            "roc_chart": read_chart(settings.reports_dir, "roc_curves.html"),
            "importance_chart": read_chart(settings.reports_dir, "feature_importance.html"),
        },
    )


@router.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request) -> HTMLResponse:
    return _render(
        request,
        "predict.html",
        {
            "page": "predict",
            "form_options": FORM_OPTIONS,
            "form_values": {},
            "errors": [],
            "result": None,
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about_page(request: Request) -> HTMLResponse:
    return _render(request, "about.html", {"page": "about"})
