"""Similarity search page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard_utils import get_similarity_engine, inject_css, render_header, render_metric_card, render_section_intro

st.set_page_config(page_title="Similar Complaints", page_icon=":material/travel_explore:", layout="wide")
inject_css()
render_header(
    "Similar Complaints",
    "Query the complaint warehouse with FAISS retrieval to surface the top semantic neighbors for any new complaint and support retrieval-augmented reasoning.",
    chips=["FAISS vector index", "Embedding-based retrieval", "Top-k semantic neighbor search"],
)

left, right = st.columns((1.15, 0.85))
with left:
    render_section_intro("Semantic retrieval", "Paste a complaint, choose the retrieval depth, and inspect the closest historical cases that would support a retrieval-augmented model decision.")
    query = st.text_area("Complaint text", height=200, placeholder="Describe the complaint you want to compare against the complaint warehouse.")
    top_k = st.slider("Top-k results", min_value=3, max_value=10, value=5)
    run_search = st.button("Retrieve similar complaints", type="primary")
with right:
    render_section_intro("Why this matters", "This page demonstrates the project's RAG foundation: complaints are embedded, indexed, retrieved, and then exposed directly for analyst inspection and model context grounding.")
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Index Backend", "FAISS", "CPU-friendly vector search")
    with m2:
        render_metric_card("Default Depth", "Top 5", "Analyst retrieval set")
    with m3:
        render_metric_card("Usage", "RAG", "Context retrieval for analysis")

if run_search:
    if not query.strip():
        st.warning("Enter a complaint to search.")
    else:
        engine = get_similarity_engine()
        results = engine.retrieve(query, top_k=top_k)
        st.write("")
        if not results:
            st.info("No similar complaints were retrieved.")
        else:
            render_section_intro("Retrieved complaint set", "The table and expandable cards below show the highest-scoring neighbors returned by the FAISS search stage.")
            st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in results]), use_container_width=True, hide_index=True)
            cols = st.columns(2)
            for idx, item in enumerate(results):
                with cols[idx % 2]:
                    with st.expander(f"#{item.complaint_id} | {item.core_issue} | similarity {item.similarity:.2f}", expanded=idx < 2):
                        st.write(item.complaint_text)
                        st.write(f"Department: {item.department} | Priority: {item.priority}")
