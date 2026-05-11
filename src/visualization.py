"""Visualization utilities for complaint analytics and model performance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px



def _write_plot(fig, save_path: Optional[str | Path]) -> None:
    """Persist a Plotly figure as interactive HTML when requested."""
    if not save_path:
        return
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    fig.write_html(output_path, include_plotlyjs="cdn")



def plot_complaint_length_distribution(
    df: pd.DataFrame,
    text_col: str = "complaint_text",
    save_path: Optional[str | Path] = None,
):
    """Plot complaint length distribution."""
    plot_df = pd.DataFrame(
        {
            "word_count": df[text_col].astype(str).apply(lambda text: len(text.split())),
        }
    )
    fig = px.histogram(
        plot_df,
        x="word_count",
        nbins=15,
        title="Complaint Length Distribution",
        color_discrete_sequence=["#16c6a5"],
        labels={"word_count": "Number of Words"},
    )
    fig.update_layout(yaxis_title="Frequency")
    _write_plot(fig, save_path)
    return fig



def plot_frequent_issues(
    df: pd.DataFrame,
    issue_col: str = "true_core_issue",
    top_n: int = 10,
    save_path: Optional[str | Path] = None,
):
    """Plot most frequent complaint issues."""
    issue_counts = (
        df[issue_col]
        .fillna("Unknown")
        .value_counts()
        .head(top_n)
        .rename_axis("issue")
        .reset_index(name="count")
    )
    fig = px.bar(
        issue_counts,
        x="count",
        y="issue",
        orientation="h",
        title="Most Frequent Complaint Issues",
        color="count",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(yaxis_title="Issue", xaxis_title="Count", coloraxis_showscale=False)
    _write_plot(fig, save_path)
    return fig



def plot_model_comparison(metrics_df: pd.DataFrame, save_path: Optional[str | Path] = None):
    """Plot model comparison graph."""
    melted = metrics_df.melt(
        id_vars=["model"],
        value_vars=["department_accuracy", "priority_accuracy", "precision", "avg_confidence"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        melted,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        title="Model Comparison by Quality Metrics",
        color_discrete_sequence=["#5b8def", "#16c6a5", "#ff9f43"],
    )
    fig.update_layout(yaxis_title="Score", xaxis_title="Metric")
    _write_plot(fig, save_path)
    return fig



def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
):
    """Plot confusion matrix heatmap."""
    fig = px.imshow(
        confusion_df,
        text_auto=True,
        title=title,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    _write_plot(fig, save_path)
    return fig



def generate_visual_reports(
    df: pd.DataFrame,
    evaluation_tables: Dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Generate interactive HTML visual reports to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_complaint_length_distribution(df, save_path=output_dir / "complaint_length_distribution.html")
    plot_frequent_issues(df, save_path=output_dir / "frequent_issues.html")

    if "metrics" in evaluation_tables:
        plot_model_comparison(evaluation_tables["metrics"], save_path=output_dir / "model_comparison.html")
    if "rule_based_confusion" in evaluation_tables:
        plot_confusion_matrix(
            evaluation_tables["rule_based_confusion"],
            title="Rule-Based Confusion Matrix",
            save_path=output_dir / "rule_based_confusion_matrix.html",
        )
