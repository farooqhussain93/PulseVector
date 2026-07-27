"""Validate required generated artifacts and independently recompute winner metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluate import evaluate_predictions  # noqa: E402

REQUIRED_MODELS = [
    "logistic_regression_pipeline.joblib",
    "knn_pipeline.joblib",
    "decision_tree_pipeline.joblib",
    "random_forest_pipeline.joblib",
    "svm_pipeline.joblib",
    "best_model.joblib",
]
REQUIRED_REPORTS = [
    "metrics.json",
    "dataset_summary.json",
    "test_predictions.csv",
    "classification_reports.json",
    "confusion_matrices.json",
]


def verify(models_dir: Path, reports_dir: Path) -> dict[str, float]:
    missing = [str(models_dir / name) for name in REQUIRED_MODELS if not (models_dir / name).exists()]
    missing += [str(reports_dir / name) for name in REQUIRED_REPORTS if not (reports_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(reports_dir / "test_predictions.csv")
    if predictions.empty:
        raise ValueError("test_predictions.csv is empty.")

    probability = predictions["positive_class_probability"].to_numpy(dtype=float)
    probability_arg = probability if np.isfinite(probability).all() else None
    independent = evaluate_predictions(
        predictions["actual_class"], predictions["predicted_class"], probability_arg
    )

    recorded = metrics["held_out_test_metrics"]
    for metric in ("accuracy", "precision", "recall", "f1", "specificity"):
        if not np.isclose(independent[metric], recorded[metric], atol=1e-12):
            raise ValueError(f"Metric mismatch for {metric}: {independent[metric]} != {recorded[metric]}")
    if recorded.get("roc_auc") is not None and not np.isclose(
        independent["roc_auc"], recorded["roc_auc"], atol=1e-12
    ):
        raise ValueError("Metric mismatch for roc_auc.")

    model = joblib.load(models_dir / "best_model.joblib")
    if not hasattr(model, "predict"):
        raise ValueError("best_model.joblib does not expose predict().")

    return {metric: float(independent[metric]) for metric in ("accuracy", "precision", "recall", "f1")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PulseVector artifacts.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify(args.models_dir, args.reports_dir)
    print("Artifacts verified:", ", ".join(f"{key}={value:.3f}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
