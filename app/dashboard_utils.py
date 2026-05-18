"""Shared Streamlit dashboard helpers."""

from __future__ import annotations

import html
from collections import defaultdict

import pandas as pd
import streamlit as st

from config import EMBEDDED_GEMINI_API_KEY, GEMINI_OPENAI_BASE_URL
from src.clustering import ComplaintClusterAnalyzer
from src.model_comparison import ModelComparisonRunner
from src.ml_model import MLComplaintAnalyzer
from src.rag_pipeline import AdvancedRAGComplaintAnalyzer
from src.rule_based_model import build_rule_based_analysis
from src.schemas import ComplaintAnalysis
from src.similarity_search import ComplaintSimilaritySearch
from src.utils import load_cfpb_dataset, load_dataset, load_validation_dataset

COMPARISON_SAMPLE_SIZE = 6000
CLUSTERING_SAMPLE_SIZE = 12000
KEYWORD_SAMPLE_SIZE = 20000

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
            --bg: #f3f7fb;
            --bg-strong: #e9f0f7;
            --panel: rgba(255, 255, 255, 0.92);
            --panel-strong: rgba(255, 255, 255, 0.98);
            --line: rgba(24, 58, 92, 0.10);
            --line-strong: rgba(24, 58, 92, 0.18);
            --text: #10243c;
            --muted: #556b84;
            --muted-soft: #7f93a9;
            --blue: #1f5eff;
            --blue-soft: #dce7ff;
            --teal: #0f9d84;
            --teal-soft: #d7f5ee;
            --amber: #c67d16;
            --amber-soft: #ffedcf;
            --rose: #d6455d;
            --rose-soft: #ffe1e5;
            --ink: #0d2136;
        }
        html, body, [class*="css"] { font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(31, 94, 255, 0.14), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(15, 157, 132, 0.12), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(198, 125, 22, 0.10), transparent 26%),
                linear-gradient(180deg, #f6f9fc 0%, #eff4fa 44%, #edf2f8 100%);
            color: var(--text);
        }
        a { color: var(--blue); }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 253, 0.98)),
                radial-gradient(circle at top, rgba(31, 94, 255, 0.08), transparent 34%);
            border-right: 1px solid var(--line);
            box-shadow: 18px 0 50px rgba(16, 36, 60, 0.05);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebarNav"] { padding-top: 0.7rem; }
        [data-testid="stSidebarNav"]::before {
            content: "Complaint AI Suite";
            display: block;
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text);
            margin: 0.4rem 0 1rem 0.5rem;
            letter-spacing: 0.03em;
        }
        [data-testid="stSidebarNav"] li a {
            border-radius: 18px;
            margin: 0.18rem 0.4rem;
            background: rgba(255,255,255,0.55);
            border: 1px solid transparent;
        }
        [data-testid="stSidebarNav"] li a:hover {
            background: rgba(31, 94, 255, 0.08);
            border-color: rgba(31, 94, 255, 0.12);
        }
        [data-testid="stSidebarNav"] li [aria-current="page"] {
            background: linear-gradient(135deg, rgba(31, 94, 255, 0.12), rgba(15, 157, 132, 0.08));
            border-color: rgba(31, 94, 255, 0.18);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.4rem;
            max-width: 1320px;
        }
        .hero-shell, .surface-card, .metric-card, .analysis-shell, .reasoning-box {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(249, 251, 254, 0.96));
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: 0 22px 65px rgba(16, 36, 60, 0.08);
        }
        .hero-shell { padding: 1.65rem 1.7rem 1.6rem; position: relative; overflow: hidden; }
        .hero-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(118deg, rgba(31, 94, 255, 0.06), transparent 34%),
                linear-gradient(310deg, rgba(15, 157, 132, 0.05), transparent 30%);
            pointer-events: none;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -5% -32% auto;
            width: 290px;
            height: 290px;
            background: radial-gradient(circle, rgba(31, 94, 255, 0.14), transparent 68%);
            pointer-events: none;
        }
        .eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--blue);
            font-weight: 700;
            margin-bottom: 0.82rem;
            position: relative;
        }
        .hero-title {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: clamp(2.2rem, 5vw, 4rem);
            line-height: 0.98;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.55rem;
            max-width: 11ch;
            position: relative;
        }
        .hero-subtitle {
            color: var(--muted);
            line-height: 1.72;
            max-width: 60rem;
            font-size: 1rem;
            position: relative;
        }
        .hero-statbar { display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 1.1rem; position: relative; }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            background: rgba(243, 247, 251, 0.95);
            border: 1px solid rgba(24, 58, 92, 0.08);
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 700;
        }
        .metric-card {
            padding: 1.15rem 1.25rem 1.05rem;
            min-height: 140px;
            position: relative;
            overflow: hidden;
        }
        .metric-card::after {
            content: "";
            position: absolute;
            inset: auto auto 0 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, rgba(31, 94, 255, 0.95), rgba(15, 157, 132, 0.7));
        }
        .metric-label { font-size: 0.77rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted-soft); }
        .metric-value { font-family: "Georgia", "Times New Roman", serif; font-size: clamp(1.9rem, 3vw, 2.5rem); font-weight: 700; margin-top: 0.42rem; color: var(--ink); }
        .metric-caption { color: var(--teal); font-size: 0.92rem; margin-top: 0.28rem; line-height: 1.45; }
        .surface-card { padding: 1.2rem 1.25rem; }
        .section-title { font-family: "Georgia", "Times New Roman", serif; font-size: 1.22rem; font-weight: 700; color: var(--ink); margin-bottom: 0.35rem; }
        .section-copy { color: var(--muted); line-height: 1.7; margin-bottom: 0.3rem; }
        .analysis-shell { padding: 1.1rem 1.1rem 1rem; height: 100%; }
        .analysis-model { font-family: "Georgia", "Times New Roman", serif; font-size: 1.12rem; font-weight: 700; margin-bottom: 0.7rem; }
        .analysis-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.82rem; }
        .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 0.36rem 0.82rem; font-size: 0.78rem; font-weight: 800; border: 1px solid transparent; }
        .pill-low { background: var(--teal-soft); color: #0b6f5e; border-color: rgba(15, 157, 132, 0.18); }
        .pill-medium { background: var(--blue-soft); color: #224ea8; border-color: rgba(31, 94, 255, 0.18); }
        .pill-high { background: var(--amber-soft); color: #8a5410; border-color: rgba(198, 125, 22, 0.2); }
        .pill-critical { background: var(--rose-soft); color: #a53549; border-color: rgba(214, 69, 93, 0.22); }
        .field-label { color: var(--muted-soft); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.10em; margin-top: 0.8rem; font-weight: 700; }
        .field-value { color: var(--text); line-height: 1.66; font-size: 0.97rem; }
        .reasoning-box {
            padding: 1.02rem 1.08rem;
            margin-top: 0.82rem;
            background: linear-gradient(135deg, rgba(15, 157, 132, 0.08), rgba(31, 94, 255, 0.06));
            border-color: rgba(15, 157, 132, 0.12);
        }
        .reasoning-title { font-size: 0.77rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--teal); font-weight: 800; margin-bottom: 0.48rem; }
        .reasoning-copy { color: var(--text); line-height: 1.7; }
        .stButton>button {
            background: linear-gradient(135deg, #1848d8, #2868ff);
            color: white;
            border-radius: 16px;
            border: none;
            font-weight: 700;
            padding: 0.72rem 1.05rem;
            box-shadow: 0 14px 28px rgba(31, 94, 255, 0.2);
        }
        .stButton>button:hover { filter: brightness(1.03); }
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        [data-baseweb="textarea"] {
            background: rgba(255, 255, 255, 0.94) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(24, 58, 92, 0.12) !important;
            color: var(--text) !important;
        }
        .stFileUploader > div,
        .stDataFrame,
        div[data-testid="stJson"],
        div[data-testid="stMetric"] {
            border: 1px solid var(--line) !important;
            border-radius: 20px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.88);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.78);
        }
        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid var(--line);
        }
        .nav-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
        .nav-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 253, 0.96));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            min-height: 170px;
            position: relative;
            overflow: hidden;
        }
        .nav-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, rgba(31, 94, 255, 0.82), rgba(15, 157, 132, 0.55), rgba(198, 125, 22, 0.38));
        }
        .nav-title { font-family: "Georgia", "Times New Roman", serif; font-size: 1.08rem; font-weight: 700; color: var(--ink); margin: 0.45rem 0 0.45rem; }
        .nav-copy { color: var(--muted); line-height: 1.68; }
        [data-testid="stMetricValue"] { color: var(--ink); font-family: "Georgia", "Times New Roman", serif; }
        [data-testid="stMetricLabel"] { color: var(--muted-soft); }
        [data-testid="stImage"] img {
            border-radius: 22px;
            border: 1px solid rgba(24, 58, 92, 0.1);
            box-shadow: 0 18px 45px rgba(16, 36, 60, 0.08);
        }
        @media (max-width: 1100px) {
            .hero-title { max-width: 14ch; }
            .nav-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .block-container { padding-top: 0.8rem; padding-left: 0.7rem; padding-right: 0.7rem; }
            .hero-shell, .surface-card, .metric-card, .analysis-shell, .reasoning-box, .nav-card { border-radius: 22px; }
            .hero-shell { padding: 1.2rem 1rem 1.15rem; }
            .hero-title { font-size: 2rem; max-width: none; }
            .hero-subtitle { font-size: 0.95rem; }
            .metric-card { min-height: 120px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_validation_data() -> pd.DataFrame:
    return load_validation_dataset()


@st.cache_data(show_spinner=False)
def get_cfpb_data() -> pd.DataFrame:
    return load_cfpb_dataset()


@st.cache_data(show_spinner=False)
def get_dataset_sample(sample_size: int, random_state: int = 42) -> pd.DataFrame:
    dataset = get_dataset()
    limit = min(len(dataset), sample_size)
    if len(dataset) <= limit:
        return dataset.copy()
    return dataset.sample(n=limit, random_state=random_state).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_ml_model() -> MLComplaintAnalyzer:
    return MLComplaintAnalyzer()


@st.cache_resource(show_spinner=False)
def get_similarity_engine() -> ComplaintSimilaritySearch:
    return ComplaintSimilaritySearch()


@st.cache_resource(show_spinner=False)
def get_rag_model(use_embedded_key: bool = False) -> AdvancedRAGComplaintAnalyzer:
    if use_embedded_key:
        return AdvancedRAGComplaintAnalyzer(
            enable_generation=True,
            provider="gemini",
            api_key=EMBEDDED_GEMINI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
            model_name="gemini-2.5-flash",
        )
    return AdvancedRAGComplaintAnalyzer(enable_generation=False)


@st.cache_data(show_spinner=False)
def get_comparison_artifacts():
    return ModelComparisonRunner().run(get_dataset_sample(COMPARISON_SAMPLE_SIZE))


@st.cache_data(show_spinner=False)
def get_cluster_artifacts(cluster_count: int):
    sample = get_dataset_sample(CLUSTERING_SAMPLE_SIZE, random_state=cluster_count + 42)
    clusterer = ComplaintClusterAnalyzer(n_clusters=cluster_count)
    artifacts = clusterer.fit_predict(sample)
    return artifacts.summary, artifacts.assignments


@st.cache_data(show_spinner=False)
def get_keyword_dataset() -> pd.DataFrame:
    return get_dataset_sample(KEYWORD_SAMPLE_SIZE)


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
