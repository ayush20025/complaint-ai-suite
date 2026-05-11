"""Typed response models for complaint analyzers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import ALLOWED_DEPARTMENTS, ALLOWED_PRIORITIES

DepartmentLiteral = Literal["Billing", "Logistics", "Technical Support", "Customer Service"]
PriorityLiteral = Literal["Low", "Medium", "High", "Critical"]


class RetrievedExample(BaseModel):
    """Retrieved knowledge-base item used to support RAG decisions."""

    model_config = ConfigDict(extra="forbid")

    complaint_id: str = Field(default="")
    complaint_text: str = Field(default="")
    department: DepartmentLiteral
    priority: PriorityLiteral
    core_issue: str = Field(default="")
    similarity: float = Field(ge=0.0, le=1.0)

    @field_validator("complaint_id", "complaint_text", "core_issue", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()


class ComplaintAnalysis(BaseModel):
    """Normalized output shared across rule-based, ML, and RAG analyzers."""

    model_config = ConfigDict(extra="forbid")

    core_issue: str = Field(min_length=1, max_length=180)
    detected_entities: list[str] = Field(default_factory=list)
    department: DepartmentLiteral
    priority: PriorityLiteral
    actionable_task: str = Field(min_length=1, max_length=300)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    analysis_mode: str = Field(default="deterministic")
    supporting_evidence: list[str] = Field(default_factory=list)
    retrieved_examples: list[RetrievedExample] = Field(default_factory=list)

    @field_validator("core_issue", "actionable_task", "analysis_mode", "reasoning", mode="before")
    @classmethod
    def _strip_strings(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("detected_entities", mode="before")
    @classmethod
    def _normalize_entities(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("department", mode="before")
    @classmethod
    def _validate_department(cls, value: object) -> str:
        department = str(value or "").strip()
        if department not in ALLOWED_DEPARTMENTS:
            return "Customer Service"
        return department

    @field_validator("priority", mode="before")
    @classmethod
    def _validate_priority(cls, value: object) -> str:
        priority = str(value or "").strip().title()
        if priority not in ALLOWED_PRIORITIES:
            return "Medium"
        return priority

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(round(confidence, 3), 1.0))
