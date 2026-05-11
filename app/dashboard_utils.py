"""Shared Streamlit dashboard helpers."""

from __future__ import annotations

import html
from collections import defaultdict

import pandas as pd
import streamlit as st

from config import EMBEDDED_GEMINI_API_KEY, GEMINI_OPENAI_BASE_URL
from src.model_comparison import ModelComparisonRunner
from src.ml_model import MLComplaintAnalyzer
from src.rag_pipeline import AdvancedRAGComplaintAnalyzer
from src.rule_based_model import build_rule_based_analysis
from src.schemas import ComplaintAnalysis
from src.similarity_search import ComplaintSimilaritySearch
from src.utils import load_dataset

PRIORITY_CLASS = {
    "Low": "pill-low",
    "Medium": "pill-medium",
    "High": "pill-high",
    "Critical": "pill-critical",
}

MODEL_TONES = {
    "Rule-Based": "#6ea8ff",
    "ML Baseline": "#43d9ad",
    "RAG + LLM": "#ffb454",
    "Recommended": "#8ce6d0",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07131f;
            --panel: rgba(12, 24, 40, 0.84);
            --panel-strong: rgba(10, 20, 34, 0.94);
            --line: rgba(151, 178, 214, 0.16);
            --text: #eef4fb;
            --muted: #9cb2ce;
            --blue: #6ea8ff;
            --teal: #43d9ad;
            --amber: #ffb454;
            --rose: #ff6b7d;
        }
        html, body, [class*="css"] { font-family: "Segoe UI", "Trebuchet MS", "Helvetica Neue", sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(110, 168, 255, 0.18), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(67, 217, 173, 0.14), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(255, 180, 84, 0.08), transparent 22%),
                linear-gradient(180deg, #07131f 0%, #0a1727 45%, #08111d 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 22, 38, 0.94), rgba(7, 16, 29, 0.9));
            border-right: 1px solid var(--line);
            backdrop-filter: blur(18px);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebarNav"] { padding-top: 0.6rem; }
        [data-testid="stSidebarNav"]::before {
            content: "Complaint AI Suite";
            display: block;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
            margin: 0.4rem 0 1rem 0.5rem;
            letter-spacing: 0.02em;
        }
        [data-testid="stSidebarNav"] li a {
            border-radius: 14px;
            margin: 0.15rem 0.4rem;
            background: rgba(255,255,255,0.02);
        }
        [data-testid="stSidebarNav"] li a:hover { background: rgba(110, 168, 255, 0.09); }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .hero-shell, .surface-card, .metric-card, .analysis-shell, .reasoning-box {
            background: linear-gradient(145deg, rgba(13, 24, 40, 0.9), rgba(8, 17, 30, 0.86));
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 24px 60px rgba(0,0,0,0.24);
            backdrop-filter: blur(14px);
        }
        .hero-shell { padding: 1.4rem 1.5rem 1.45rem; position: relative; overflow: hidden; }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -10% -45% auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(110,168,255,0.22), transparent 68%);
            pointer-events: none;
        }
        .eyebrow { font-size: 0.73rem; letter-spacing: 0.14em; text-transform: uppercase; color: #7fd8ff; font-weight: 700; margin-bottom: 0.75rem; }
        .hero-title { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 2.45rem; line-height: 1.03; font-weight: 700; color: var(--text); margin-bottom: 0.45rem; max-width: 12ch; }
        .hero-subtitle { color: var(--muted); line-height: 1.7; max-width: 56rem; font-size: 1rem; }
        .hero-statbar { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1rem; }
        .chip { display: inline-flex; align-items: center; gap: 0.35rem; border-radius: 999px; padding: 0.38rem 0.8rem; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); color: var(--muted); font-size: 0.8rem; font-weight: 600; }
        .metric-card { padding: 1.1rem 1.2rem 1rem; min-height: 132px; }
        .metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
        .metric-value { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 2rem; font-weight: 700; margin-top: 0.35rem; color: var(--text); }
        .metric-caption { color: #87d7c1; font-size: 0.9rem; margin-top: 0.25rem; }
        .surface-card { padding: 1.15rem 1.2rem; }
        .section-title { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }
        .section-copy { color: var(--muted); line-height: 1.65; margin-bottom: 0.4rem; }
        .analysis-shell { padding: 1rem 1.05rem 0.95rem; height: 100%; }
        .analysis-model { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.65rem; }
        .analysis-meta { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.8rem; }
        .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 0.34rem 0.78rem; font-size: 0.78rem; font-weight: 700; border: 1px solid transparent; }
        .pill-low { background: rgba(67, 217, 173, 0.12); color: #8af1ce; border-color: rgba(67, 217, 173, 0.22); }
        .pill-medium { background: rgba(110, 168, 255, 0.12); color: #9bc1ff; border-color: rgba(110, 168, 255, 0.24); }
        .pill-high { background: rgba(255, 180, 84, 0.14); color: #ffcc8b; border-color: rgba(255, 180, 84, 0.28); }
        .pill-critical { background: rgba(255, 107, 125, 0.14); color: #ff9cab; border-color: rgba(255, 107, 125, 0.28); }
        .field-label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.75rem; }
        .field-value { color: var(--text); line-height: 1.62; font-size: 0.97rem; }
        .reasoning-box { padding: 1rem 1.05rem; margin-top: 0.8rem; background: linear-gradient(135deg, rgba(67, 217, 173, 0.12), rgba(110, 168, 255, 0.08)); }
        .reasoning-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8ce6d0; font-weight: 700; margin-bottom: 0.45rem; }
        .reasoning-copy { color: var(--text); line-height: 1.68; }
        .stButton>button { background: linear-gradient(135deg, #6ea8ff, #4782ff); color: white; border-radius: 14px; border: 1px solid rgba(110, 168, 255, 0.32); font-weight: 700; padding: 0.7rem 1rem; }
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stSlider { background: rgba(14, 24, 40, 0.76) !important; border-radius: 16px !important; }
        .stDataFrame, div[data-testid="stJson"] { border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
        .nav-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
        .nav-card { background: linear-gradient(160deg, rgba(15, 27, 46, 0.95), rgba(8, 17, 32, 0.9)); border: 1px solid var(--line); border-radius: 22px; padding: 1.15rem 1.2rem; min-height: 160px; }
        .nav-title { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.45rem; }
        .nav-copy { color: var(--muted); line-height: 1.65; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_ml_model() -> MLComplaintAnalyzer:
    return MLComplaintAnalyzer().fit(get_dataset())


@st.cache_resource(show_spinner=False)
def get_similarity_engine() -> ComplaintSimilaritySearch:
    return ComplaintSimilaritySearch(dataset=get_dataset())


@st.cache_resource(show_spinner=False)
def get_rag_model(use_embedded_key: bool = False) -> AdvancedRAGComplaintAnalyzer:
    if use_embedded_key:
        return AdvancedRAGComplaintAnalyzer(
            knowledge_base=get_dataset(),
            enable_generation=True,
            provider="gemini",
            api_key=EMBEDDED_GEMINI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
            model_name="gemini-2.5-flash",
        )
    return AdvancedRAGComplaintAnalyzer(knowledge_base=get_dataset(), enable_generation=False)


@st.cache_data(show_spinner=False)
def get_comparison_artifacts():
    return ModelComparisonRunner().run(get_dataset())


def render_lazy_action(button_label: str, key: str, help_text: str | None = None) -> bool:
    if key not in st.session_state:
        st.session_state[key] = False
    clicked = st.button(button_label, key=f"{key}_button", help=help_text, type="primary")
    if clicked:
        st.session_state[key] = True
    return bool(st.session_state[key])


def render_header(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-statbar">' + ''.join(f'<span class="chip">{html.escape(chip)}</span>' for chip in chips) + '</div>'
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="eyebrow">MSc Complaint Intelligence</div>
            <div class="hero-title">{html.escape(title)}</div>
            <div class="hero-subtitle">{html.escape(subtitle)}</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="surface-card">
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav_cards(cards: list[tuple[str, str]]) -> None:
    card_html = ''.join(
        f'<div class="nav-card"><div class="nav-title">{html.escape(title)}</div><div class="nav-copy">{html.escape(copy)}</div></div>'
        for title, copy in cards
    )
    st.markdown(f'<div class="nav-grid">{card_html}</div>', unsafe_allow_html=True)


def render_reasoning_box(reasoning: str) -> None:
    st.markdown(
        f"""
        <div class="reasoning-box">
            <div class="reasoning-title">AI Reasoning</div>
            <div class="reasoning-copy">{html.escape(reasoning)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_block(title: str, analysis) -> None:
    tone = MODEL_TONES.get(title, "#6ea8ff")
    priority_class = PRIORITY_CLASS.get(analysis.priority, "pill-medium")
    st.markdown(
        f"""
        <div class="analysis-shell">
            <div class="analysis-model" style="color:{tone};">{html.escape(title)}</div>
            <div class="analysis-meta">
                <span class="pill {priority_class}">{html.escape(analysis.priority)}</span>
                <span class="chip">{html.escape(analysis.department)}</span>
                <span class="chip">Confidence {analysis.confidence_score:.2f}</span>
            </div>
            <div class="field-label">Core Issue</div>
            <div class="field-value">{html.escape(analysis.core_issue)}</div>
            <div class="field-label">Detected Entities</div>
            <div class="field-value">{html.escape(', '.join(analysis.detected_entities) if analysis.detected_entities else 'None')}</div>
            <div class="field-label">Actionable Task</div>
            <div class="field-value">{html.escape(analysis.actionable_task)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_reasoning_box(analysis.reasoning)
    if analysis.supporting_evidence:
        with st.expander("Supporting evidence"):
            for item in analysis.supporting_evidence:
                st.write(f"- {item}")
    if analysis.retrieved_examples:
        with st.expander("Retrieved similar complaints"):
            st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in analysis.retrieved_examples]), use_container_width=True, hide_index=True)
    with st.expander("Structured JSON"):
        st.json(analysis.model_dump(mode="json"), expanded=False)


def sample_complaints(df: pd.DataFrame) -> list[str]:
    return df["complaint_text"].head(12).astype(str).tolist()


def trio_analysis(
    complaint_text: str,
    use_embedded_key: bool = False,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
):
    rag_model = get_rag_model(use_embedded_key=use_embedded_key)
    normalized_text = complaint_text.strip()
    image_summary = ""

    if image_bytes and image_mime_type and use_embedded_key:
        try:
            image_summary = rag_model.summarize_image_complaint(
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                complaint_text=normalized_text,
            ).strip()
        except Exception:
            image_summary = ""

    if normalized_text and image_summary:
        shared_text = f"{normalized_text}\n\nImage findings: {image_summary}"
    else:
        shared_text = normalized_text or image_summary

    return {
        "Rule-Based": build_rule_based_analysis(shared_text),
        "ML Baseline": get_ml_model().analyze(shared_text),
        "RAG + LLM": rag_model.analyze(
            shared_text,
            enable_generation=use_embedded_key,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        ),
    }


def build_recommended_analysis(results: dict[str, ComplaintAnalysis]) -> ComplaintAnalysis:
    """Return a final recommended prediction based on agreement and confidence."""
    issue_scores: dict[str, float] = defaultdict(float)
    department_scores: dict[str, float] = defaultdict(float)
    priority_scores: dict[str, float] = defaultdict(float)

    for analysis in results.values():
        confidence = float(analysis.confidence_score)
        issue_scores[analysis.core_issue] += confidence
        department_scores[analysis.department] += confidence
        priority_scores[analysis.priority] += confidence

    recommended_issue = max(issue_scores, key=issue_scores.get)
    recommended_department = max(department_scores, key=department_scores.get)
    recommended_priority = max(priority_scores, key=priority_scores.get)

    agreeing_models = [
        name
        for name, analysis in results.items()
        if (
            analysis.core_issue == recommended_issue
            and analysis.department == recommended_department
            and analysis.priority == recommended_priority
        )
    ]

    highest_confidence_name, highest_confidence_result = max(results.items(), key=lambda item: item[1].confidence_score)
    fully_agreeing_results = [
        analysis
        for analysis in results.values()
        if analysis.core_issue == recommended_issue and analysis.department == recommended_department and analysis.priority == recommended_priority
    ]
    winner = max(fully_agreeing_results, key=lambda analysis: analysis.confidence_score, default=highest_confidence_result)

    if len(agreeing_models) >= 2:
        consensus_confidence = round(
            min(
                1.0,
                (sum(results[name].confidence_score for name in agreeing_models) / len(agreeing_models))
                + (0.05 * (len(agreeing_models) - 1)),
            ),
            3,
        )
        reasoning = (
            f"This final recommendation is based on model agreement. "
            f"{', '.join(agreeing_models)} aligned on '{recommended_issue}' for {recommended_department} with {recommended_priority} priority."
        )
        support = [
            f"Agreeing models: {', '.join(agreeing_models)}.",
            f"Consensus issue score: {issue_scores[recommended_issue]:.3f}.",
            f"Consensus department score: {department_scores[recommended_department]:.3f}.",
            f"Consensus priority score: {priority_scores[recommended_priority]:.3f}.",
        ]
    else:
        winner = highest_confidence_result
        consensus_confidence = round(max(0.2, min(0.75, winner.confidence_score * 0.85)), 3)
        reasoning = (
            f"The models did not reach strong consensus, so the dashboard is surfacing the highest-confidence result from "
            f"{highest_confidence_name} while flagging the disagreement for manual review."
        )
        support = [
            f"No multi-model consensus was found across issue, department, and priority.",
            f"Highest-confidence model: {highest_confidence_name} ({winner.confidence_score:.3f}).",
            f"Weighted issue leader: {recommended_issue}.",
            f"Weighted department leader: {recommended_department}.",
        ]

    return winner.model_copy(
        update={
            "confidence_score": consensus_confidence,
            "analysis_mode": "recommended-consensus",
            "reasoning": reasoning,
            "supporting_evidence": support + winner.supporting_evidence,
        }
    )
