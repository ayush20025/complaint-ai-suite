"""CFPB-style benchmark page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_utils import get_dataset, inject_css, render_header, render_lazy_action, render_metric_card, render_section_intro
from src.ml_model import MLComplaintAnalyzer
from src.rag_pipeline import AdvancedRAGComplaintAnalyzer
from src.rule_based_model import build_rule_based_analysis
from src.utils import load_cfpb_dataset

st.set_page_config(page_title="CFPB Benchmark", page_icon=":material/dataset:", layout="wide")
inject_css()
render_header(
    "CFPB-style Benchmark",
    "This page shows how the project models behave on a separate CFPB-style complaint dataset stored in an independent data folder for report-ready external benchmarking.",
    chips=["Separate dataset folder", "External-style evaluation", "Sample prediction review"],
)


@st.cache_data(show_spinner=False)
def get_cfpb_dataset() -> pd.DataFrame:
    return load_cfpb_dataset()


@st.cache_data(show_spinner=False)
def run_cfpb_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = get_dataset()
    cfpb_df = get_cfpb_dataset()
    ml_model = MLComplaintAnalyzer().fit(train_df, persist=False)
    rag_model = AdvancedRAGComplaintAnalyzer(knowledge_base=train_df, enable_generation=False)
    outputs = {
        "Rule-Based": cfpb_df["complaint_text"].apply(build_rule_based_analysis),
        "ML Baseline": cfpb_df["complaint_text"].apply(ml_model.analyze),
        "RAG + LLM": cfpb_df["complaint_text"].apply(lambda text: rag_model.analyze(text, enable_generation=False)),
    }

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    y_true = cfpb_df["true_department"].reset_index(drop=True)
    for model_name, series in outputs.items():
        y_pred = series.apply(lambda row: row.department)
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "f1_score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            }
        )
        prediction_rows.extend(
            {
                "model": model_name,
                "complaint_id": cfpb_df.iloc[idx]["complaint_id"],
                "product": cfpb_df.iloc[idx]["product"],
                "complaint_text": cfpb_df.iloc[idx]["complaint_text"],
                "true_department": cfpb_df.iloc[idx]["true_department"],
                "predicted_department": row.department,
                "priority": row.priority,
                "core_issue": row.core_issue,
                "is_correct": cfpb_df.iloc[idx]["true_department"] == row.department,
            }
            for idx, row in enumerate(series.tolist())
        )
    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


cfpb_df = get_cfpb_dataset()
k1, k2, k3 = st.columns(3)
with k1:
    render_metric_card("CFPB-style Rows", f"{len(cfpb_df):,}", "Independent benchmark set")
with k2:
    render_metric_card("Products", str(cfpb_df['product'].nunique()), "Complaint product categories")
with k3:
    render_metric_card("Departments", str(cfpb_df['true_department'].nunique()), "Mapped routing targets")

render_section_intro("Dataset framing", "This dataset is separate from the project warehouse. It uses CFPB-style product and narrative fields while still mapping each complaint into the four internal routing departments used by your system.")

if render_lazy_action("Run CFPB benchmark", "cfpb_benchmark_load", "Evaluates the three model families on the separate CFPB-style dataset."):
    metrics_df, predictions_df = run_cfpb_benchmark()
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    render_section_intro("Sample predictions", "These rows help explain whether the system's internal routing logic transfers cleanly to a CFPB-style complaint narrative distribution.")
    st.dataframe(predictions_df.head(30), use_container_width=True, hide_index=True)
else:
    st.info("Run the CFPB benchmark to view metrics and sample predictions.")
