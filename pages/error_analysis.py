"""Error analysis page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_utils import get_comparison_artifacts, inject_css, render_header, render_metric_card, render_section_intro
from src.model_comparison import ModelComparisonRunner

st.set_page_config(page_title="Error Analysis", page_icon=":material/bug_report:", layout="wide")
inject_css()
artifacts = get_comparison_artifacts()
render_header(
    "Error Analysis",
    "Inspect wrong predictions and confusion matrices so you can explain model weaknesses, failure patterns, and research trade-offs in your final project presentation.",
    chips=["Wrong predictions", "Department confusion", "Model weakness analysis"],
)

wrong_df = artifacts.predictions[artifacts.predictions["is_wrong"]].copy()
selected_model = st.selectbox("Model", artifacts.metrics["model"].tolist())
model_row = artifacts.metrics.loc[artifacts.metrics["model"] == selected_model].iloc[0]
model_wrong = wrong_df.loc[wrong_df["model"] == selected_model].copy()

k1, k2, k3 = st.columns(3)
with k1:
    render_metric_card("Accuracy", f"{model_row['accuracy']:.2f}", f"{selected_model} department accuracy")
with k2:
    render_metric_card("Wrong Predictions", f"{len(model_wrong):,}", "Count of incorrect routed complaints")
with k3:
    render_metric_card("F1 Score", f"{model_row['f1_score']:.2f}", "Balanced classification quality")

st.write("")
render_section_intro("Confusion matrix", "This view shows which departments the selected model confuses most often. It is useful for discussing weaknesses and motivating retrieval or stronger learning approaches.")
st.pyplot(ModelComparisonRunner.plot_confusion(artifacts.confusion[selected_model], f"{selected_model} Confusion Matrix"), use_container_width=True)

st.write("")
render_section_intro("Wrong predictions", "Reviewing specific failure cases gives your project a stronger research narrative than reporting accuracy alone.")
if model_wrong.empty:
    st.success("No wrong predictions were found for the selected model on the current evaluation split.")
else:
    st.dataframe(model_wrong[["complaint_text", "true_department", "predicted_department", "confidence_score"]], use_container_width=True, hide_index=True)
