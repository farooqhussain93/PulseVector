"""Plotly chart generation for PulseVector reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

BRAND = {
    "navy": "#0B2230",
    "teal": "#0F9D8A",
    "coral": "#F26B4A",
    "cream": "#F5FAF8",
    "slate": "#46616E",
}


def _write_fragment(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Inter, Arial, sans-serif", "color": BRAND["navy"]},
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    path.write_text(
        fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False}),
        encoding="utf-8",
    )


def create_dataset_charts(data: pd.DataFrame, output_dir: Path) -> list[str]:
    """Create class, feature, and correlation charts."""

    generated: list[str] = []
    binary = (data["num"] > 0).astype(int).map({0: "Not detected", 1: "Present"})

    class_counts = binary.value_counts().rename_axis("Class").reset_index(name="Records")
    fig = px.bar(
        class_counts,
        x="Class",
        y="Records",
        title="Target Class Distribution",
        text="Records",
        color="Class",
        color_discrete_map={"Not detected": BRAND["teal"], "Present": BRAND["coral"]},
    )
    fig.update_layout(showlegend=False)
    _write_fragment(fig, output_dir / "class_distribution.html")
    generated.append("class_distribution.html")

    melted = data[["age", "trestbps", "chol", "thalach", "oldpeak"]].melt(
        var_name="Feature", value_name="Value"
    )
    fig = px.histogram(
        melted,
        x="Value",
        facet_col="Feature",
        facet_col_wrap=3,
        nbins=24,
        title="Selected Numerical Feature Distributions",
    )
    fig.update_layout(height=650, showlegend=False)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    _write_fragment(fig, output_dir / "feature_distributions.html")
    generated.append("feature_distributions.html")

    correlation_data = data.copy()
    correlation_data["target"] = (correlation_data["num"] > 0).astype(int)
    correlation = correlation_data.drop(columns=["num"]).corr(numeric_only=True)
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    fig.update_layout(height=760)
    _write_fragment(fig, output_dir / "correlation_heatmap.html")
    generated.append("correlation_heatmap.html")
    return generated


def create_model_comparison_chart(metrics: dict[str, Any], output_dir: Path) -> str:
    """Create a grouped comparison chart from cross-validation means."""

    rows: list[dict[str, Any]] = []
    for model_name, payload in metrics["candidate_models"].items():
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            rows.append(
                {
                    "Model": payload["display_name"],
                    "Metric": metric.replace("_", " ").upper(),
                    "Score": payload["cross_validation"][metric]["mean"],
                }
            )
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Cross-Validation Model Comparison",
        range_y=[0, 1],
    )
    fig.update_layout(xaxis_tickangle=-18, legend_orientation="h", legend_y=-0.25)
    filename = "model_comparison.html"
    _write_fragment(fig, output_dir / filename)
    return filename


def create_confusion_matrices(metrics: dict[str, Any], output_dir: Path) -> str:
    """Create a five-model confusion-matrix comparison view."""

    candidates = list(metrics["candidate_models"].items())
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[payload["display_name"] for _, payload in candidates],
        horizontal_spacing=0.12,
        vertical_spacing=0.22,
    )
    for index, (_, payload) in enumerate(candidates):
        row = index // 3 + 1
        col = index % 3 + 1
        matrix = np.asarray(payload["test_metrics"]["confusion_matrix"])
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, BRAND["cream"]], [1, BRAND["teal"]]],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    fig.update_layout(title="Held-Out Test Confusion Matrices", height=720)
    filename = "confusion_matrices.html"
    _write_fragment(fig, output_dir / filename)
    return filename


def create_roc_curves(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    display_names: dict[str, str],
    output_dir: Path,
) -> str:
    """Create held-out ROC curves for candidates supporting probability estimates."""

    fig = go.Figure()
    for name, probability in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_test, probability)
        fig.add_trace(
            go.Scatter(
                x=false_positive,
                y=true_positive,
                mode="lines",
                name=display_names[name],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash"},
            name="Random baseline",
        )
    )
    fig.update_layout(
        title="Held-Out Test ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend_orientation="h",
        legend_y=-0.25,
    )
    filename = "roc_curves.html"
    _write_fragment(fig, output_dir / filename)
    return filename


def create_probability_distribution(
    y_true: pd.Series,
    probability: np.ndarray,
    output_dir: Path,
) -> str:
    """Create a winner probability-distribution chart."""

    frame = pd.DataFrame(
        {
            "Model-estimated probability": probability,
            "Actual class": y_true.map({0: "Not detected", 1: "Present"}).to_numpy(),
        }
    )
    fig = px.histogram(
        frame,
        x="Model-estimated probability",
        color="Actual class",
        nbins=18,
        barmode="overlay",
        opacity=0.72,
        title="Winning Model Probability Distribution",
        color_discrete_map={"Not detected": BRAND["teal"], "Present": BRAND["coral"]},
    )
    filename = "probability_distribution.html"
    _write_fragment(fig, output_dir / filename)
    return filename


def create_feature_importance_chart(frame: pd.DataFrame, output_dir: Path) -> str:
    """Create a top-feature chart from an exported importance table."""

    top = frame.sort_values("importance", ascending=False).head(15).sort_values("importance")
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title="Winning Model - Top Feature Contributions",
    )
    filename = "feature_importance.html"
    _write_fragment(fig, output_dir / filename)
    return filename
