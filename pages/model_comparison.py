"""Model comparison page."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_utils import get_comparison_artifacts, inject_css, render_header, render_lazy_action, render_metric_card, render_section_intro
from src.model_comparison import ModelComparisonRunner

st.set_page_config(page_title="Model Comparison", page_icon=":material/balance:", layout="wide")
inject_css()
render_header(
    "Model Comparison",
    "Benchmark rule-based, classical ML, and retrieval-augmented pipelines with grouped evaluation that keeps near-duplicate templates out of both train and test.",
    chips=["Grouped holdout protocol", "Accuracy, precision, recall, F1", "Confusion matrices"],
)

render_section_intro("Evaluation note", "This page is lazy-loaded to keep the dashboard responsive. When you run the benchmark, the split is grouped so repeated synthetic templates cannot leak into both train and test.")

if render_lazy_action("Run grouped benchmark", "comparison_load_metrics", "Loads the cached comparison artifacts for this session."):
    artifacts = get_comparison_artifacts()
    best_row = artifacts.metrics.sort_values("accuracy", ascending=False).iloc[0]
    k1, k2, k3 = st.columns(3)
    with k1:
        render_metric_card("Best Model", str(best_row["model"]), "Top department routing performer")
    with k2:
        render_metric_card("Best Accuracy", f"{best_row['accuracy']:.2f}", "Highest observed score under grouped holdout")
    with k3:
        render_metric_card("Average Confidence", f"{artifacts.metrics['avg_confidence'].mean():.2f}", "Mean confidence across models")

    st.write("")
    render_section_intro("Performance overview", "The comparison table and chart below present the experimental position of each model family across the key evaluation criteria expected in an MSc AI project.")
    st.dataframe(artifacts.metrics, use_container_width=True, hide_index=True)
    fig = ModelComparisonRunner.plot_metrics(artifacts.metrics)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.write("")
    render_section_intro("Handwritten validation", "This smaller external validation slice is intentionally written by hand. It gives you a more realistic narrative for the report than the synthetic warehouse alone.")
    st.dataframe(artifacts.external_validation_metrics, use_container_width=True, hide_index=True)

    st.write("")
    render_section_intro("Error analysis", "Confusion matrices make it easier to discuss where each approach confuses departments and where retrieval or learned classification offers a measurable benefit.")
    for model_name, confusion_df in artifacts.confusion.items():
        st.markdown(f"### {model_name} Confusion Matrix")
        fig = ModelComparisonRunner.plot_confusion(confusion_df, f"{model_name} Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
else:
    st.info("Benchmark results will appear here after you run the grouped evaluation.")
