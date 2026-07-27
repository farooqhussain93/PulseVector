from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app  # noqa: E402
from app.services.model_service import clear_model_cache  # noqa: E402
from ml.train import TrainingPaths, train_all  # noqa: E402


@pytest.fixture
def valid_payload() -> dict[str, float | int]:
    return {
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


@pytest.fixture(scope="session")
def isolated_pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("pipeline")
    models = root / "models"
    reports = root / "reports"
    dataset = PROJECT_ROOT / "data" / "raw" / "heart_disease_cleveland.csv"
    metrics = train_all(TrainingPaths(dataset=dataset, models_dir=models, reports_dir=reports))
    return {"root": root, "models": models, "reports": reports, "metrics": metrics}


@pytest.fixture
def isolated_settings(isolated_pipeline: dict[str, object], monkeypatch: pytest.MonkeyPatch):
    models = Path(isolated_pipeline["models"])
    reports = Path(isolated_pipeline["reports"])
    test_settings = SimpleNamespace(
        project_name="PulseVector",
        educational_disclaimer=(
            "This project is an educational machine learning demonstration and is not a medical diagnostic tool."
        ),
        templates_dir=PROJECT_ROOT / "templates",
        static_dir=PROJECT_ROOT / "static",
        model_path=models / "best_model.joblib",
        metrics_path=reports / "metrics.json",
        dataset_summary_path=reports / "dataset_summary.json",
        reports_dir=reports,
    )

    for module_name in (
        "app.main",
        "app.routes.pages",
        "app.routes.prediction",
        "app.services.model_service",
        "app.services.readiness_service",
    ):
        monkeypatch.setattr(f"{module_name}.settings", test_settings)

    clear_model_cache()
    yield test_settings
    clear_model_cache()


@pytest.fixture
def client(isolated_settings) -> TestClient:
    del isolated_settings
    return TestClient(create_app())
