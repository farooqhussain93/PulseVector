"""PulseVector FastAPI application factory."""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routes import pages, prediction


def _json_safe(value: Any) -> Any:
    """Convert validation-error payloads into strict JSON-compatible values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return str(value)
    return str(value)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PulseVector API",
        description=(
            "Educational FastAPI service comparing five machine-learning classifiers on the UCI Cleveland heart dataset."
        ),
        version="1.0.1",
        contact={"name": "PulseVector Portfolio Project"},
        license_info={"name": "MIT"},
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})

    app.state.templates = Jinja2Templates(directory=str(settings.templates_dir))
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.include_router(pages.router)
    app.include_router(prediction.router)
    return app


app = create_app()
