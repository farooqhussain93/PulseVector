"""Safe access to generated JSON and Plotly report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportUnavailableError(RuntimeError):
    """Raised when a required report cannot be read or validated."""


def read_json(path: Path, *, required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    if not path.exists():
        raise ReportUnavailableError(f"Required report is unavailable: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportUnavailableError(f"Required report is malformed: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ReportUnavailableError(f"Required report has an invalid structure: {path.name}")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ReportUnavailableError(f"Required report is missing fields: {missing}")
    return payload


def read_metrics(path: Path) -> dict[str, Any]:
    return read_json(
        path,
        required_keys=("candidate_models", "selection", "held_out_test_metrics", "class_mapping", "feature_list"),
    )


def read_dataset_summary(path: Path) -> dict[str, Any]:
    return read_json(path, required_keys=("rows", "feature_count", "binary_target_distribution"))


def read_chart(reports_dir: Path, filename: str) -> str | None:
    path = reports_dir / filename
    if not path.exists() or path.suffix != ".html":
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return content if "plotly" in content.lower() else None
