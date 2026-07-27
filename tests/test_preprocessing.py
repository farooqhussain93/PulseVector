from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    build_model_pipeline,
    build_preprocessor,
    validate_feature_columns,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [50, 1, 4, 130, 240, 0, 0, 150, 0, 1.0, 2, 0, 3],
            [60, 0, 3, 140, 260, 1, 1, 130, 1, 2.0, 3, np.nan, 7],
        ],
        columns=FEATURE_COLUMNS,
    )


def test_feature_groups_cover_contract_without_overlap() -> None:
    assert set(NUMERICAL_COLUMNS).isdisjoint(CATEGORICAL_COLUMNS)
    assert set(NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS) == set(FEATURE_COLUMNS)


def test_validate_feature_columns_accepts_exact_order() -> None:
    validate_feature_columns(FEATURE_COLUMNS.copy())


def test_validate_feature_columns_rejects_wrong_order() -> None:
    with pytest.raises(ValueError, match="feature contract"):
        validate_feature_columns(list(reversed(FEATURE_COLUMNS)))


def test_scaled_preprocessor_contains_standard_scaler() -> None:
    preprocessor = build_preprocessor(scale_numeric=True)
    numeric = preprocessor.transformers[0][1]
    assert isinstance(numeric.named_steps["scaler"], StandardScaler)


def test_tree_preprocessor_omits_standard_scaler() -> None:
    preprocessor = build_preprocessor(scale_numeric=False)
    numeric = preprocessor.transformers[0][1]
    assert "scaler" not in numeric.named_steps


def test_preprocessor_imputes_and_encodes_finite_values() -> None:
    transformed = build_preprocessor(scale_numeric=True).fit_transform(sample_frame())
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


def test_model_pipeline_fits_and_predicts() -> None:
    frame = sample_frame()
    target = pd.Series([0, 1])
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000), scale_numeric=True)
    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)
    assert predictions.tolist() == [0, 1]
