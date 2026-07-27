"""Command-line entry point for the complete PulseVector ML pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import TrainingPaths, train_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all PulseVector classifiers and generate reports.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "heart_disease_cleveland.csv",
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_all(
        TrainingPaths(dataset=args.dataset, models_dir=args.models_dir, reports_dir=args.reports_dir)
    )
    winner = metrics["selection"]["winner_name"]
    f1 = metrics["held_out_test_metrics"]["f1"]
    print(f"Pipeline complete. Winner: {winner}; held-out F1: {f1:.3f}")


if __name__ == "__main__":
    main()
