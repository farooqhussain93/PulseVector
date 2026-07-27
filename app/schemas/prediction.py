"""Validated patient feature contract for HTML and JSON predictions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.preprocessing import FEATURE_COLUMNS


class PredictionRequest(BaseModel):
    """Cleveland dataset feature schema with explicit validation boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type")
    trestbps: float = Field(..., ge=70, le=250, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG category")
    thalach: float = Field(..., ge=60, le=230, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., description="Thalassemia category: 3, 6, or 7")

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value: int) -> int:
        if value not in {3, 6, 7}:
            raise ValueError("thal must be one of 3, 6, or 7")
        return value

    def to_model_record(self) -> dict[str, Any]:
        """Return model feature names in the exact training order."""

        payload = self.model_dump()
        return {column: payload[column] for column in FEATURE_COLUMNS}


class PredictionResponse(BaseModel):
    predicted_class: int
    label: str
    positive_class_probability: float | None
    probability_label: str | None
    winning_model: str
    evaluation_metrics: dict[str, float | None]
    disclaimer: str
