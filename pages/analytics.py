"""Complaint analytics page."""

from __future__ import annotations

import streamlit as st

from app.dashboard_utils import get_dataset, inject_css, render_header, render_metric_card, render_section_intro
from src.analytics import complaint_length_distribution, department_distribution, keyword_frequency, priority_distribution
from src.clustering import ComplaintClusterAnalyzer
from src.utils import load_cfpb_dataset, load_validation_dataset

st.set_page_config(page_title="Complaint Analytics", page_icon=":material/insights:", layout="wide")
inject_css()
dataset = get_dataset()
validation_df = load_validation_dataset()
cfpb_df = load_cfpb_dataset()
render_header(
    "Complaint Analytics",
    "Explore complaint distributions, keyword trends, and unsupervised clustering patterns through a more polished research dashboard lens.",
    chips=["50K raw warehouse", "5K handwritten validation", "CFPB-style external benchmark"],
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
    f"The 50K raw warehouse is the main training and retrieval corpus. The 5K handwritten validation set keeps the same label schema but uses more natural writing style for realism checks. The CFPB-style dataset is a separate external benchmark with product and narrative fields mapped into your internal departments. Current sizes: raw {len(dataset):,}, handwritten {len(validation_df):,}, CFPB-style {len(cfpb_df):,}.",
)

st.write("")
render_section_intro("Distribution views", "These plots summarize workload shape, departmental demand, urgency profile, and lexical patterns in the complaint warehouse.")

c1, c2 = st.columns(2)
with c1:
    st.pyplot(complaint_length_distribution(dataset), use_container_width=True)
with c2:
    st.pyplot(department_distribution(dataset), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.pyplot(priority_distribution(dataset), use_container_width=True)
with c4:
    st.pyplot(keyword_frequency(dataset), use_container_width=True)

st.write("")
render_section_intro("Complaint clustering", "KMeans clustering reveals latent complaint themes, helping position the project beyond supervised classification into exploratory AI analysis.")
cluster_count = st.slider("Number of clusters", min_value=4, max_value=10, value=6)
clusterer = ComplaintClusterAnalyzer(n_clusters=cluster_count)
artifacts = clusterer.fit_predict(dataset)
left, right = st.columns((0.9, 1.1))
with left:
    st.dataframe(artifacts.summary, use_container_width=True, hide_index=True)
with right:
    sampled = artifacts.assignments.sample(min(700, len(artifacts.assignments)), random_state=42)
    st.pyplot(clusterer.plot_clusters(sampled), use_container_width=True)
