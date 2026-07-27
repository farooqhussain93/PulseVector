"""Central application configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_name: str = "PulseVector"
    project_root: Path = Path(__file__).resolve().parents[1]
    educational_disclaimer: str = (
        "This project is an educational machine learning demonstration and is not a medical diagnostic tool."
    )

    @property
    def templates_dir(self) -> Path:
        return self.project_root / "templates"

    @property
    def static_dir(self) -> Path:
        return self.project_root / "static"

    @property
    def model_path(self) -> Path:
        return self.project_root / "models" / "best_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.project_root / "reports" / "metrics.json"

    @property
    def dataset_summary_path(self) -> Path:
        return self.project_root / "reports" / "dataset_summary.json"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"


settings = Settings()
