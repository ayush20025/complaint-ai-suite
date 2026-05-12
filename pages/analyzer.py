"""Complaint analyzer page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_utils import build_recommended_analysis, get_dataset, inject_css, render_analysis_block, render_header, render_metric_card, render_section_intro, sample_complaints, trio_analysis

st.set_page_config(page_title="Complaint Analyzer", page_icon=":material/psychology:", layout="wide")
inject_css()
dataset = get_dataset()
render_header(
    "Complaint Analyzer",
    "Run multi-step complaint analysis and compare rule-based, ML, and RAG outputs with reasoning, evidence, structured JSON, and optional multimodal LLM support.",
    chips=["Head-to-head model view", "Gemini toggle", "Text + image input"],
)

samples = ["Custom complaint"] + sample_complaints(dataset)
selected = st.sidebar.selectbox("Sample complaint", samples)
head_to_head = st.sidebar.toggle("Head-to-head comparison", value=True)
use_embedded_gemini = st.sidebar.toggle("Use Gemini vision", value=False, help="When enabled, the RAG + LLM panel uses the Gemini API key configured in environment variables or Streamlit secrets for generation.")

left, right = st.columns((1.2, 0.8))
with left:
    render_section_intro("Analyst input", "Paste a live complaint, optionally attach an image screenshot or bill, and then generate structured triage outputs across the project's three model families.")
    complaint_text = st.text_area(
        "Complaint text",
        value="" if selected == "Custom complaint" else selected,
        height=200,
        placeholder="Example: My order arrived damaged and customer support has not responded for three days.",
    )
    uploaded_image = st.file_uploader("Optional complaint image", type=["png", "jpg", "jpeg", "webp"], help="Useful for invoice screenshots, damaged product photos, or support chat captures.")
    analyze = st.button("Run complaint analysis", type="primary")
with right:
    render_section_intro("What this page shows", "Each result panel highlights the predicted issue, extracted entities, routing decision, priority, actionable task, and AI reasoning explanation for academic transparency.")
    k1, k2, k3 = st.columns(3)
    with k1:
        render_metric_card("Dataset Samples", f"{len(dataset):,}", "Historical complaints available")
    with k2:
        render_metric_card("Modes", "3", "Rule-based, ML, and RAG")
    with k3:
        render_metric_card("Input Mode", "Text + Image", "Multimodal when Gemini is enabled")
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded complaint image preview", use_container_width=True)

if analyze:
    if not complaint_text.strip() and uploaded_image is None:
        st.warning("Enter complaint text or upload a complaint image to analyze.")
    else:
        image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
        image_mime_type = uploaded_image.type if uploaded_image is not None else None
        results = trio_analysis(
            complaint_text or "Analyze the attached complaint image and infer the complaint details.",
            use_embedded_key=use_embedded_gemini,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
        recommended = build_recommended_analysis(results)
        st.write("")
        render_section_intro("Final recommended prediction", "This is the dashboard's final answer based on model agreement and confidence. Use this in demos when the three model families do not fully agree.")
        render_analysis_block("Recommended", recommended)
        st.write("")
        if head_to_head:
            cols = st.columns(3)
            for col, (name, analysis) in zip(cols, results.items()):
                with col:
                    render_analysis_block(name, analysis)
        else:
            best_name, best_result = max(results.items(), key=lambda item: item[1].confidence_score)
            render_analysis_block(best_name, best_result)

        rag_result = results["RAG + LLM"]
        st.write("")
        render_section_intro("Top 5 similar complaints", "These are the highest-scoring retrieval results used to ground the retrieval-augmented analysis. Showing them clearly strengthens the RAG story in demos and viva discussions.")
        if rag_result.retrieved_examples:
            retrieved_df = pd.DataFrame([item.model_dump(mode="json") for item in rag_result.retrieved_examples[:5]])
            st.dataframe(retrieved_df, use_container_width=True, hide_index=True)
            sim_cols = st.columns(2)
            for idx, item in enumerate(rag_result.retrieved_examples[:5]):
                with sim_cols[idx % 2]:
                    with st.expander(f"#{item.complaint_id} | {item.core_issue} | similarity {item.similarity:.2f}", expanded=idx < 2):
                        st.write(item.complaint_text)
                        st.write(f"Department: {item.department} | Priority: {item.priority}")
        else:
            st.info("No similar complaints were retrieved for this query.")
