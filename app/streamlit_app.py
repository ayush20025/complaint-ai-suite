"""Landing page for the upgraded complaint AI dashboard."""

from __future__ import annotations

import streamlit as st

from app.dashboard_utils import get_dataset, inject_css, render_header, render_lazy_action, render_metric_card, render_nav_cards, render_section_intro
from src.analytics import department_distribution, priority_distribution


def main() -> None:
    st.set_page_config(page_title="Complaint AI Research Dashboard", page_icon=":material/analytics:", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    dataset = get_dataset()

    render_header(
        "Complaint-to-Action Intelligence Hub",
        "Analyze complaints, compare model outputs, inspect similar cases, and benchmark the system from one dashboard.",
        chips=[f"Warehouse {len(dataset):,} complaints", "3 model families", "CFPB benchmark included"],
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Complaint Warehouse", f"{len(dataset):,}", "Expanded enterprise-scale complaint dataset")
    with c2:
        render_metric_card("Model Portfolio", "3", "Rule-based, ML baseline, and retrieval-augmented AI")
    with c3:
        render_metric_card("Evaluation Mode", "Grouped", "Near-duplicate complaint templates held out together")

    st.write("")
    left, right = st.columns((1.15, 0.85))
    with left:
        render_section_intro(
            "Research-grade workflow",
            "This interface is structured like a modern SaaS operations console: analysts can triage complaints, compare model outputs, inspect semantic neighbors, study distributional patterns, and review error cases from one workspace.",
        )
        st.write("")
        render_nav_cards([
            ("Complaint Analyzer", "Run multi-step structured analysis with core issue, entities, department, priority, actionable task, confidence, and AI reasoning."),
            ("Complaint Analytics", "Inspect complaint length, keyword activity, department and priority mixes, and unsupervised KMeans cluster patterns."),
            ("Model Comparison", "Benchmark rule-based, classical ML, and RAG pipelines with evaluation tables and confusion matrices."),
            ("Similar Complaints", "Search the complaint warehouse with FAISS retrieval and inspect the top semantic neighbors for any new case."),
            ("Error Analysis", "Review wrong predictions and confusion matrices to explain model weaknesses and research trade-offs."),
        ])
    with right:
        render_section_intro(
            "Quick start",
            "Use the sidebar pages to open the analyzer, analytics, model comparison, CFPB benchmark, similarity search, and error analysis views.",
        )
        st.info("Heavy benchmark views open only when requested, so the landing page stays lighter than the analysis pages.")

    st.write("")
    render_section_intro("Home insights", "These light-weight summaries load immediately so the home page stays responsive while still looking polished in demos.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.metric("Evaluation Protocol", "Grouped holdout", "Prevents template leakage between train and test")
    with chart_right:
        st.pyplot(department_distribution(dataset), use_container_width=True)

    chart_left2, chart_right2 = st.columns(2)
    with chart_left2:
        st.pyplot(priority_distribution(dataset), use_container_width=True)
    with chart_right2:
        if render_lazy_action("Load comparison snapshot", "home_load_comparison", "Runs the grouped benchmark once and caches the result."):
            from src.model_comparison import ModelComparisonRunner

            comparison = ModelComparisonRunner().run(dataset)
            st.dataframe(comparison.metrics[["model", "accuracy", "precision", "recall", "f1_score"]], use_container_width=True, hide_index=True)
        else:
            st.caption("Comparison metrics stay unloaded until requested.")


if __name__ == "__main__":
    main()
