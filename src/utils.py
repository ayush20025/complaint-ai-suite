"""Utility helpers for normalization, validation, confidence scoring, and feature extraction."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

from config import ALLOWED_DEPARTMENTS, ALLOWED_PRIORITIES, CFPB_DATA_PATH, OUTPUT_KEYS, PROCESSED_DATA_PATH, RAW_DATA_PATH, VALIDATION_DATA_PATH

PRIORITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
ENTITY_VOCAB = {
    "billing": ["invoice", "payment", "refund", "charge", "subscription", "billing"],
    "logistics": ["delivery", "shipment", "courier", "warehouse", "package", "tracking", "arrived", "dispatch", "replacement", "pickup"],
    "technical": ["app", "website", "server", "login", "api", "bug", "crash", "error"],
    "service": ["support", "agent", "response", "service", "callback", "ticket"],
    "urgency": ["urgent", "asap", "immediately", "escalate", "critical"],
    "product": ["product", "item", "device", "order", "replacement", "screen", "damage", "damaged", "broken", "glass", "shattered", "cracked", "fractured", "dented", "smashed"],
}


def get_logger(name: str = "complaint_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_dataset() -> pd.DataFrame:
    dataset_path = RAW_DATA_PATH if Path(RAW_DATA_PATH).exists() else PROCESSED_DATA_PATH
    df = pd.read_csv(dataset_path)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def load_validation_dataset() -> pd.DataFrame:
    validation_path = Path(VALIDATION_DATA_PATH)
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {validation_path}")
    return pd.read_csv(validation_path)


def load_cfpb_dataset() -> pd.DataFrame:
    cfpb_path = Path(CFPB_DATA_PATH)
    if not cfpb_path.exists():
        raise FileNotFoundError(f"CFPB dataset not found: {cfpb_path}")
    return pd.read_csv(cfpb_path)


def extract_entities(text: str) -> list[str]:
    normalized = normalize_text(text)
    entities: list[str] = []
    for label, keywords in ENTITY_VOCAB.items():
        if any(keyword in normalized for keyword in keywords):
            entities.append(label)
    return list(dict.fromkeys(entities))


def build_reasoning(core_issue: str, department: str, priority: str, entities: Sequence[str], source: str) -> str:
    entity_text = ", ".join(entities) if entities else "general complaint indicators"
    return (
        f"The complaint signals '{core_issue}' through entities such as {entity_text}. "
        f"This maps to the {department} department. Priority is {priority} based on the severity and urgency cues "
        f"identified by the {source} analysis stage."
    )


def safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object.")
    return value


def calculate_confidence(weighted_score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    value = max(0.0, min(weighted_score / max_score, 1.0))
    return round(value, 3)


def normalize_priority(value: object, fallback: str = "Medium") -> str:
    priority = str(value or "").strip().title()
    if priority not in ALLOWED_PRIORITIES:
        return fallback
    return priority


def highest_priority(*priorities: str, fallback: str = "Medium") -> str:
    normalized = [normalize_priority(priority, fallback=fallback) for priority in priorities if priority]
    if not normalized:
        return fallback
    return max(normalized, key=lambda priority: PRIORITY_RANK.get(priority, 0))


def weighted_vote(values: Sequence[str], weights: Sequence[float], fallback: str) -> str:
    if not values or not weights or len(values) != len(weights):
        return fallback
    scores: dict[str, float] = defaultdict(float)
    for value, weight in zip(values, weights):
        scores[str(value)] += float(weight)
    return max(scores, key=scores.get) if scores else fallback


def derive_template_group(text: str) -> str:
    """Collapse slot-filled synthetic complaints into a stable template signature."""
    normalized = normalize_text(text)
    normalized = re.sub(r"\$?\b\d+(?:\.\d+)?\b", "<num>", normalized)
    normalized = re.sub(r"\bcmp-\d+\b", "<ticket>", normalized)
    normalized = re.sub(r"\b\d{1,2}\s+[a-z]{3}\s+\d{4}\b", "<date>", normalized)
    normalized = re.sub(
        r"\bi am a\s+(student|smb|enterprise|premium|standard)\s+customer and this feels\s+[a-z]+\b\.?",
        "<segment>",
        normalized,
    )
    normalized = re.sub(r"please escalate this immediately\.?", "<escalate>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return md5(normalized.encode("utf-8")).hexdigest()


def validate_output_json(payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    if not isinstance(payload, dict):
        return False, {}, "Payload must be a JSON object."
    sanitized: Dict[str, Any] = {}
    for key in OUTPUT_KEYS:
        if key not in payload:
            return False, {}, f"Missing key: {key}"
    sanitized["core_issue"] = str(payload.get("core_issue", "")).strip()
    raw_entities = payload.get("detected_entities", [])
    sanitized["detected_entities"] = extract_entities(", ".join(raw_entities) if isinstance(raw_entities, list) else str(raw_entities))
    department = str(payload.get("department", "")).strip()
    if department not in ALLOWED_DEPARTMENTS:
        return False, {}, f"Invalid department: {department}"
    sanitized["department"] = department
    priority = normalize_priority(payload.get("priority", ""), fallback="")
    if priority not in ALLOWED_PRIORITIES:
        return False, {}, f"Invalid priority: {priority}"
    sanitized["priority"] = priority
    sanitized["actionable_task"] = str(payload.get("actionable_task", "")).strip()
    sanitized["reasoning"] = str(payload.get("reasoning", "")).strip()
    try:
        confidence = float(payload.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        return False, {}, "confidence_score must be numeric."
    if not 0.0 <= confidence <= 1.0:
        return False, {}, "confidence_score must be between 0 and 1."
    sanitized["confidence_score"] = round(confidence, 3)
    return True, sanitized, "valid"
