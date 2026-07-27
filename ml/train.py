"""Train, compare, select, and persist PulseVector classification models."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.evaluate import evaluate_predictions, get_positive_class_probability
from ml.plots import (
    create_confusion_matrices,
    create_dataset_charts,
    create_feature_importance_chart,
    create_model_comparison_chart,
    create_probability_distribution,
    create_roc_curves,
)
from ml.preprocessing import (
    CLASS_MAPPING,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model_pipeline,
    validate_feature_columns,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SELECTION_METRIC = "f1"
TIE_BREAKERS = ["recall", "accuracy"]


@dataclass(frozen=True)
class TrainingPaths:
    """Filesystem contract for reproducible training outputs."""

    dataset: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class Candidate:
    """Candidate model configuration."""

    key: str
    display_name: str
    pipeline: Any


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the approved Cleveland dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, na_values=["?"])
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].copy()
    if data.empty:
        raise ValueError("Dataset is empty.")

    invalid_targets = sorted(set(data[TARGET_COLUMN].dropna().astype(int)) - {0, 1, 2, 3, 4})
    if invalid_targets:
        raise ValueError(f"Unexpected target values: {invalid_targets}")

    return data


def prepare_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the documented feature matrix and binary target."""

    features = data[FEATURE_COLUMNS].copy()
    validate_feature_columns(features.columns.tolist())
    target = (data[TARGET_COLUMN].astype(int) > 0).astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("Binary target must contain both classes 0 and 1.")
    return features, target


def build_candidates() -> dict[str, Candidate]:
    """Create all five project classifiers with model-appropriate preprocessing."""

    return {
        "logistic_regression": Candidate(
            key="logistic_regression",
            display_name="Logistic Regression",
            pipeline=build_model_pipeline(
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
        ),
        "knn": Candidate(
            key="knn",
            display_name="K-Nearest Neighbors",
            pipeline=build_model_pipeline(KNeighborsClassifier(n_neighbors=7), scale_numeric=True),
        ),
        "decision_tree": Candidate(
            key="decision_tree",
            display_name="Decision Tree",
            pipeline=build_model_pipeline(
                DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE),
                scale_numeric=False,
            ),
        ),
        "random_forest": Candidate(
            key="random_forest",
            display_name="Random Forest",
            pipeline=build_model_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale_numeric=False,
            ),
        ),
        "svm": Candidate(
            key="svm",
            display_name="Support Vector Machine",
            pipeline=build_model_pipeline(
                SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
        ),
    }


def make_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the one shared stratified train/test split."""

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def make_cv_splits(features: pd.DataFrame, target: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize shared stratified folds so every model uses identical row indices."""

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(features, target))


def summarize_cv(raw: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Convert cross_validate arrays into serializable mean/std summaries."""

    summary: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = np.asarray(raw[f"test_{metric}"], dtype=float)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def select_winner(candidate_results: dict[str, dict[str, Any]]) -> str:
    """Select by CV F1, then recall, then accuracy; never inspect test metrics."""

    if not candidate_results:
        raise ValueError("No candidate results were provided.")

    def ranking(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, str]:
        key, payload = item
        cv = payload["cross_validation"]
        return (
            float(cv[SELECTION_METRIC]["mean"]),
            float(cv[TIE_BREAKERS[0]]["mean"]),
            float(cv[TIE_BREAKERS[1]]["mean"]),
            key,
        )

    return max(candidate_results.items(), key=ranking)[0]


def build_dataset_summary(data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Create a concise, UI-safe dataset audit report."""

    missing = data.isna().sum()
    binary_counts = target.value_counts().sort_index()
    return {
        "dataset_name": "UCI Heart Disease - Processed Cleveland",
        "source": "UCI Machine Learning Repository",
        "source_file": "processed.cleveland.data",
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_mapping": CLASS_MAPPING,
        "original_target_distribution": {
            str(key): int(value) for key, value in data[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "binary_target_distribution": {
            "0": int(binary_counts.get(0, 0)),
            "1": int(binary_counts.get(1, 0)),
        },
        "missing_values": {key: int(value) for key, value in missing.items() if int(value) > 0},
        "total_missing_values": int(missing.sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_feature_columns": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_feature_columns": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "educational_disclaimer": (
            "This project is an educational machine learning demonstration and is not a medical diagnostic tool."
        ),
    }


def extract_feature_importance(model: Any) -> pd.DataFrame | None:
    """Return transformed feature contributions for supported winner estimators."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = np.asarray(preprocessor.get_feature_names_out())

    values: np.ndarray | None = None
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_, dtype=float)).ravel()

    if values is None or values.shape[0] != feature_names.shape[0]:
        return None
    return pd.DataFrame({"feature": feature_names, "importance": values})


def train_all(paths: TrainingPaths) -> dict[str, Any]:
    """Run the complete leakage-safe training, comparison, and artifact pipeline."""

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(paths.dataset)
    features, target = prepare_features_target(data)
    x_train, x_test, y_train, y_test = make_split(features, target)
    cv_splits = make_cv_splits(x_train, y_train)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    candidate_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    candidates = build_candidates()

    for key, candidate in candidates.items():
        cv_raw = cross_validate(
            clone(candidate.pipeline),
            x_train,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        model = clone(candidate.pipeline)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probability = get_positive_class_probability(model, x_test)
        test_metrics = evaluate_predictions(y_test, predictions, probability)

        artifact_name = f"{key}_pipeline.joblib"
        joblib.dump(model, paths.models_dir / artifact_name)
        fitted_models[key] = model
        if probability is not None:
            probabilities[key] = probability

        candidate_results[key] = {
            "display_name": candidate.display_name,
            "artifact": artifact_name,
            "cross_validation": summarize_cv(cv_raw),
            "test_metrics": test_metrics,
        }

    winner_key = select_winner(candidate_results)
    winner_model = fitted_models[winner_key]
    joblib.dump(winner_model, paths.models_dir / "best_model.joblib")

    winner_predictions = winner_model.predict(x_test)
    winner_probability = get_positive_class_probability(winner_model, x_test)
    winner_test_metrics = evaluate_predictions(y_test, winner_predictions, winner_probability)

    test_predictions = pd.DataFrame(
        {
            "row_index": x_test.index,
            "actual_class": y_test.to_numpy(),
            "predicted_class": winner_predictions,
            "positive_class_probability": (
                winner_probability if winner_probability is not None else np.full(len(y_test), np.nan)
            ),
        }
    ).sort_values("row_index")
    test_predictions.to_csv(paths.reports_dir / "test_predictions.csv", index=False)

    dataset_summary = build_dataset_summary(data, target)
    (paths.reports_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )

    confusion_payload = {
        key: payload["test_metrics"]["confusion_matrix"] for key, payload in candidate_results.items()
    }
    (paths.reports_dir / "confusion_matrices.json").write_text(
        json.dumps(confusion_payload, indent=2), encoding="utf-8"
    )

    reports_payload = {
        key: payload["test_metrics"]["classification_report"]
        for key, payload in candidate_results.items()
    }
    (paths.reports_dir / "classification_reports.json").write_text(
        json.dumps(reports_payload, indent=2), encoding="utf-8"
    )

    feature_importance = extract_feature_importance(winner_model)
    feature_importance_file: str | None = None
    if feature_importance is not None:
        feature_importance_file = "feature_importance.csv"
        feature_importance.to_csv(paths.reports_dir / feature_importance_file, index=False)

    metrics: dict[str, Any] = {
        "project_name": "PulseVector",
        "task": "Binary heart disease classification",
        "candidate_models": candidate_results,
        "selection": {
            "metric": SELECTION_METRIC,
            "tie_breakers": TIE_BREAKERS,
            "winner_key": winner_key,
            "winner_name": candidates[winner_key].display_name,
            "policy": (
                "Winner selected from training-set five-fold stratified cross-validation only. "
                "Held-out test metrics were not used for selection."
            ),
        },
        "held_out_test_metrics": winner_test_metrics,
        "class_mapping": CLASS_MAPPING,
        "feature_list": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": N_SPLITS,
            "shuffle": True,
            "same_materialized_folds_for_all_models": True,
        },
        "split_metadata": {
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "training_indices": [int(value) for value in sorted(x_train.index.tolist())],
            "test_indices": [int(value) for value in sorted(x_test.index.tolist())],
        },
        "artifact_metadata": {
            "best_model": "best_model.joblib",
            "feature_importance": feature_importance_file,
            "test_predictions": "test_predictions.csv",
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "educational_disclaimer": (
            "This project is an educational machine learning demonstration and is not a medical diagnostic tool."
        ),
    }

    create_dataset_charts(data, paths.reports_dir)
    create_model_comparison_chart(metrics, paths.reports_dir)
    create_confusion_matrices(metrics, paths.reports_dir)
    create_roc_curves(
        y_test,
        probabilities,
        {key: candidate.display_name for key, candidate in candidates.items()},
        paths.reports_dir,
    )
    if winner_probability is not None:
        create_probability_distribution(y_test, winner_probability, paths.reports_dir)
    if feature_importance is not None:
        create_feature_importance_chart(feature_importance, paths.reports_dir)

    (paths.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
