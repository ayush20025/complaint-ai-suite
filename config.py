"""Global configuration for the complaint AI system."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_DATA_PATH = DATA_DIR / "raw" / "complaints.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_complaints.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation" / "handwritten_complaints.csv"
CFPB_DATA_PATH = DATA_DIR / "cfpb" / "consumer_complaints.csv"
VECTOR_INDEX_PATH = MODELS_DIR / "vector_index.faiss"
EMBEDDING_META_PATH = MODELS_DIR / "embedding_metadata.parquet"
ML_CLASSIFIER_PATH = MODELS_DIR / "ml_classifier.pkl"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
USE_SENTENCE_TRANSFORMER = os.getenv("USE_SENTENCE_TRANSFORMER", "false").strip().lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDED_GEMINI_API_KEY = os.getenv("EMBEDDED_GEMINI_API_KEY", GEMINI_API_KEY)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "45"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
DEFAULT_CLUSTER_COUNT = int(os.getenv("DEFAULT_CLUSTER_COUNT", "6"))
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def resolve_llm_settings(
    provider_override: str | None = None,
    gemini_api_key_override: str | None = None,
    openai_api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> dict[str, str]:
    """Resolve which OpenAI-compatible provider to use for generation."""
    provider = (provider_override or LLM_PROVIDER or "auto").strip().lower()
    gemini_api_key = gemini_api_key_override if gemini_api_key_override is not None else GEMINI_API_KEY
    openai_api_key = openai_api_key_override if openai_api_key_override is not None else OPENAI_API_KEY
    base_url = base_url_override if base_url_override is not None else OPENAI_BASE_URL

    if provider == "auto":
        if gemini_api_key:
            provider = "gemini"
        elif openai_api_key:
            provider = "openai"
        else:
            provider = "none"

    if provider == "gemini":
        return {
            "provider": provider,
            "api_key": gemini_api_key,
            "base_url": base_url or GEMINI_OPENAI_BASE_URL,
            "model_name": GEMINI_MODEL_NAME,
        }

    if provider == "openai":
        return {
            "provider": provider,
            "api_key": openai_api_key,
            "base_url": base_url,
            "model_name": MODEL_NAME,
        }

    return {
        "provider": "none",
        "api_key": "",
        "base_url": "",
        "model_name": GEMINI_MODEL_NAME if gemini_api_key else MODEL_NAME,
    }


ACTIVE_LLM_SETTINGS = resolve_llm_settings()
ACTIVE_LLM_PROVIDER = ACTIVE_LLM_SETTINGS["provider"]
ACTIVE_LLM_API_KEY = ACTIVE_LLM_SETTINGS["api_key"]
ACTIVE_LLM_BASE_URL = ACTIVE_LLM_SETTINGS["base_url"]
ACTIVE_MODEL_NAME = ACTIVE_LLM_SETTINGS["model_name"]
GENERATION_AVAILABLE = bool(ACTIVE_LLM_API_KEY)
LLM_PROVIDER_LABEL = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "none": "None",
}.get(ACTIVE_LLM_PROVIDER, ACTIVE_LLM_PROVIDER.title())

ALLOWED_DEPARTMENTS = [
    "Billing",
    "Logistics",
    "Technical Support",
    "Customer Service",
]
ALLOWED_PRIORITIES = ["Low", "Medium", "High", "Critical"]

DEPARTMENT_DEFAULT_TASKS = {
    "Billing": "Review the billing ledger, verify the disputed transaction, and update the customer.",
    "Logistics": "Audit shipment milestones, coordinate with the courier, and share a recovery timeline.",
    "Technical Support": "Reproduce the issue, isolate the root cause, and communicate the fix plan.",
    "Customer Service": "Open a manual triage ticket, capture missing context, and route the case to the right team.",
}

OUTPUT_KEYS = [
    "core_issue",
    "detected_entities",
    "department",
    "priority",
    "actionable_task",
    "confidence_score",
    "reasoning",
]
