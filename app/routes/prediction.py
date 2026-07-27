"""Prediction endpoints for HTML forms and JSON clients."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.config import settings
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.data_service import FORM_OPTIONS
from app.services.model_service import ModelUnavailableError, predict_patient
from app.services.report_service import ReportUnavailableError, read_metrics

router = APIRouter()


def _prediction_context(**extra: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "page": "predict",
        "form_options": FORM_OPTIONS,
        "disclaimer": settings.educational_disclaimer,
        "form_values": {},
        "errors": [],
        "result": None,
    }
    context.update(extra)
    return context


@router.post("/api/predict", response_model=PredictionResponse)
def api_predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        metrics = read_metrics(settings.metrics_path)
        return predict_patient(payload, metrics)
    except (ModelUnavailableError, ReportUnavailableError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed unexpectedly.",
        ) from exc


@router.post("/prediction/", response_class=HTMLResponse)
async def form_predict(request: Request) -> HTMLResponse:
    form = await request.form()
    values = {key: value for key, value in form.items()}
    try:
        payload = PredictionRequest.model_validate(values)
        metrics = read_metrics(settings.metrics_path)
        result = predict_patient(payload, metrics)
        context = _prediction_context(request=request, form_values=values, result=result)
        return request.app.state.templates.TemplateResponse(request, "predict.html", context)
    except ValidationError as exc:
        errors = [error["msg"] for error in exc.errors()]
        context = _prediction_context(request=request, form_values=values, errors=errors)
        return request.app.state.templates.TemplateResponse(request, "predict.html", context, 422)
    except (ModelUnavailableError, ReportUnavailableError) as exc:
        context = _prediction_context(request=request, form_values=values, errors=[str(exc)])
        return request.app.state.templates.TemplateResponse(request, "predict.html", context, 503)
    except Exception:
        context = _prediction_context(
            request=request,
            form_values=values,
            errors=["Prediction failed unexpectedly. Please try again after checking service readiness."],
        )
        return request.app.state.templates.TemplateResponse(request, "predict.html", context, 500)
