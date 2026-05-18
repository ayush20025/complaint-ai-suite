"""Complaint analytics page."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_utils import (
    get_cfpb_data,
    get_cluster_artifacts,
    get_dataset,
    get_keyword_dataset,
    get_validation_data,
    inject_css,
    render_header,
    render_lazy_action,
    render_metric_card,
    render_section_intro,
)
from src.analytics import complaint_length_distribution, department_distribution, keyword_frequency, priority_distribution

st.set_page_config(page_title="Complaint Analytics", page_icon=":material/insights:", layout="wide")
inject_css()
dataset = get_dataset()
validation_df = get_validation_data()
cfpb_df = get_cfpb_data()
render_header(
    "Complaint Analytics",
    "Explore complaint distributions, keyword trends, and unsupervised clustering patterns through a more polished research dashboard lens.",
    chips=["150K raw warehouse", "5K handwritten validation", "CFPB-style external benchmark"],
)

k1, k2, k3 = st.columns(3)
with k1:
    render_metric_card("Rows Analysed", f"{len(dataset):,}", "Complaint warehouse volume")
with k2:
    render_metric_card("Departments", str(dataset['true_department'].nunique()), "Distinct routing targets")
with k3:
    render_metric_card("Priority Levels", str(dataset['true_priority'].nunique()), "Operational urgency bands")

st.write("")
render_section_intro(
    "Dataset roles",
    f"The 150K raw warehouse is the main training and retrieval corpus. The 5K handwritten validation set keeps the same label schema but uses more natural writing style for realism checks. The CFPB-style dataset is a separate external benchmark with product and narrative fields mapped into your internal departments. Current sizes: raw {len(dataset):,}, handwritten {len(validation_df):,}, CFPB-style {len(cfpb_df):,}.",
)

st.write("")
render_section_intro("Distribution views", "These plots summarize workload shape, departmental demand, urgency profile, and lexical patterns in the complaint warehouse.")

c1, c2 = st.columns(2)
with c1:
    fig = complaint_length_distribution(dataset)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with c2:
    fig = department_distribution(dataset)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

c3, c4 = st.columns(2)
with c3:
    fig = priority_distribution(dataset)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with c4:
    fig = keyword_frequency(get_keyword_dataset())
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.write("")
render_section_intro("Complaint clustering", "KMeans clustering is now lazy-loaded on a representative sample so this page opens faster while still showing meaningful emergent themes.")
cluster_count = st.slider("Number of clusters", min_value=4, max_value=10, value=6)
if render_lazy_action("Load cluster analysis", f"analytics_cluster_{cluster_count}", "Runs clustering on a sampled complaint slice and caches the result."):
    summary_df, assignments_df = get_cluster_artifacts(cluster_count)
    left, right = st.columns((0.9, 1.1))
    with left:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with right:
        from src.clustering import ComplaintClusterAnalyzer

        clusterer = ComplaintClusterAnalyzer(n_clusters=cluster_count)
        sampled = assignments_df.sample(min(700, len(assignments_df)), random_state=42)
        fig = clusterer.plot_clusters(sampled)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
else:
    st.info("Cluster analysis loads on demand so moving between dashboard pages stays smoother.")
