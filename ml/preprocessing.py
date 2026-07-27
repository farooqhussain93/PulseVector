"""Preprocessing utilities for the PulseVector classification pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "num"

CLASS_MAPPING = {
    "negative_class": 0,
    "negative_label": "Heart disease not detected",
    "positive_class": 1,
    "positive_label": "Heart disease present",
    "source_mapping": "0 = no disease; values 1-4 = disease present",
}


@dataclass(frozen=True)
class PipelineSpec:
    """Describes the preprocessing requirements for a candidate estimator."""

    scale_numeric: bool


def validate_feature_columns(columns: list[str]) -> None:
    """Raise when the model feature set or ordering differs from the contract."""

    if columns != FEATURE_COLUMNS:
        raise ValueError(
            "Feature columns must match the documented feature contract and order. "
            f"Expected {FEATURE_COLUMNS}, received {columns}."
        )


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Build a leakage-safe preprocessing graph for numeric and categorical inputs."""

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: Any, *, scale_numeric: bool) -> Pipeline:
    """Combine model-appropriate preprocessing and an estimator in one pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", estimator),
        ]
    )
