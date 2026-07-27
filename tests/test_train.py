from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from ml.preprocessing import FEATURE_COLUMNS
from ml.train import (
    N_SPLITS,
    TrainingPaths,
    build_candidates,
    build_dataset_summary,
    extract_feature_importance,
    load_dataset,
    make_cv_splits,
    make_split,
    prepare_features_target,
    select_winner,
    summarize_cv,
    train_all,
)
from scripts.verify_artifacts import verify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "raw" / "heart_disease_cleveland.csv"


def test_load_dataset_uses_approved_columns() -> None:
    data = load_dataset(DATASET)
    assert data.columns.tolist() == FEATURE_COLUMNS + ["num"]
    assert data.shape == (303, 14)


def test_load_dataset_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("missing.csv"))


def test_load_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [50], "num": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_load_dataset_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=FEATURE_COLUMNS + ["num"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


def test_load_dataset_rejects_invalid_target(tmp_path: Path) -> None:
    data = load_dataset(DATASET).head(2)
    data.loc[data.index[0], "num"] = 9
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected target"):
        load_dataset(path)


def test_target_mapping_is_explicit_binary() -> None:
    features, target = prepare_features_target(load_dataset(DATASET))
    assert features.columns.tolist() == FEATURE_COLUMNS
    assert set(target.unique()) == {0, 1}
    assert int(target.sum()) == 139


def test_prepare_features_rejects_single_class() -> None:
    data = load_dataset(DATASET).copy()
    data["num"] = 0
    with pytest.raises(ValueError, match="both classes"):
        prepare_features_target(data)


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    features, target = prepare_features_target(load_dataset(DATASET))
    first = make_split(features, target)
    second = make_split(features, target)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert set(first[0].index).isdisjoint(first[1].index)
    assert len(first[1]) == 61
    assert abs(first[3].mean() - target.mean()) < 0.02


def test_shared_cv_splits_are_stratified() -> None:
    features, target = prepare_features_target(load_dataset(DATASET))
    x_train, _, y_train, _ = make_split(features, target)
    splits = make_cv_splits(x_train, y_train)
    assert len(splits) == N_SPLITS
    for train_idx, valid_idx in splits:
        assert set(train_idx).isdisjoint(valid_idx)
        assert abs(y_train.iloc[valid_idx].mean() - y_train.mean()) < 0.08


def test_all_five_candidates_share_feature_contract() -> None:
    candidates = build_candidates()
    assert set(candidates) == {"logistic_regression", "knn", "decision_tree", "random_forest", "svm"}
    assert all(candidate.pipeline.steps[0][0] == "preprocessor" for candidate in candidates.values())


def test_cv_summary_means_and_standard_deviations() -> None:
    raw = {f"test_{metric}": np.array([0.6, 0.8]) for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    summary = summarize_cv(raw)
    assert summary["f1"]["mean"] == pytest.approx(0.7)
    assert summary["f1"]["std"] == pytest.approx(0.1)


def test_winner_selection_uses_f1_then_recall_then_accuracy() -> None:
    results = {
        "a": {"cross_validation": {"f1": {"mean": 0.8}, "recall": {"mean": 0.7}, "accuracy": {"mean": 0.9}}},
        "b": {"cross_validation": {"f1": {"mean": 0.8}, "recall": {"mean": 0.8}, "accuracy": {"mean": 0.7}}},
        "c": {"cross_validation": {"f1": {"mean": 0.79}, "recall": {"mean": 0.99}, "accuracy": {"mean": 0.99}}},
    }
    assert select_winner(results) == "b"


def test_winner_selection_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="No candidate"):
        select_winner({})


def test_dataset_summary_records_data_quality() -> None:
    data = load_dataset(DATASET)
    _, target = prepare_features_target(data)
    summary = build_dataset_summary(data, target)
    assert summary["rows"] == 303
    assert summary["total_missing_values"] == 6
    assert summary["duplicate_rows"] == 0
    assert summary["binary_target_distribution"] == {"0": 164, "1": 139}


def test_extract_feature_importance_unsupported_model_returns_none() -> None:
    features, target = prepare_features_target(load_dataset(DATASET))
    from ml.preprocessing import build_model_pipeline
    model = build_model_pipeline(DummyClassifier(strategy="most_frequent"), scale_numeric=False)
    model.fit(features, target)
    assert extract_feature_importance(model) is None


@pytest.mark.integration
def test_isolated_pipeline_generates_complete_artifacts(isolated_pipeline: dict[str, object]) -> None:
    models = isolated_pipeline["models"]
    reports = isolated_pipeline["reports"]
    assert (models / "best_model.joblib").exists()
    assert len(list(models.glob("*_pipeline.joblib"))) == 5
    assert (reports / "metrics.json").exists()
    assert (reports / "model_comparison.html").exists()
    assert (reports / "test_predictions.csv").exists()


@pytest.mark.integration
def test_pipeline_winner_is_cv_based_and_metadata_complete(isolated_pipeline: dict[str, object]) -> None:
    metrics = isolated_pipeline["metrics"]
    candidate_f1 = {
        key: value["cross_validation"]["f1"]["mean"]
        for key, value in metrics["candidate_models"].items()
    }
    assert metrics["selection"]["winner_key"] == max(candidate_f1, key=candidate_f1.get)
    assert "Held-out test metrics were not used" in metrics["selection"]["policy"]
    assert len(metrics["split_metadata"]["test_indices"]) == 61


@pytest.mark.integration
def test_saved_winner_reloads_and_predicts_finite_probability(isolated_pipeline: dict[str, object]) -> None:
    model = joblib.load(isolated_pipeline["models"] / "best_model.joblib")
    features, _ = prepare_features_target(load_dataset(DATASET))
    probability = model.predict_proba(features.head(1))[0, 1]
    assert np.isfinite(probability)
    assert 0 <= probability <= 1


@pytest.mark.integration
def test_artifact_verifier_recomputes_metrics(isolated_pipeline: dict[str, object]) -> None:
    verified = verify(isolated_pipeline["models"], isolated_pipeline["reports"])
    assert verified["f1"] == pytest.approx(isolated_pipeline["metrics"]["held_out_test_metrics"]["f1"])


def test_artifact_verifier_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify(tmp_path / "models", tmp_path / "reports")


def test_train_all_rejects_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_all(TrainingPaths(tmp_path / "none.csv", tmp_path / "models", tmp_path / "reports"))
