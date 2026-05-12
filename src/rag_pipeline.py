"""Advanced RAG complaint analysis pipeline built on FAISS retrieval and optional LLM synthesis."""

from __future__ import annotations

import json
import time
from base64 import b64encode
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

from config import ACTIVE_LLM_API_KEY, ACTIVE_LLM_BASE_URL, ACTIVE_LLM_PROVIDER, ACTIVE_MODEL_NAME, API_TIMEOUT_SECONDS, MAX_RETRIES, RAW_DATA_PATH, RAG_TOP_K, TEMPERATURE
from src.rule_based_model import ISSUE_RULES, build_rule_based_analysis
from src.schemas import ComplaintAnalysis
from src.similarity_search import ComplaintSimilaritySearch
from src.utils import build_reasoning, calculate_confidence, extract_entities, get_logger, safe_json_loads, validate_output_json, weighted_vote

logger = get_logger(__name__)
PROMPT_TEMPLATE = (
    "You are an AI complaint triage architect. Return strict JSON with keys: "
    "core_issue, detected_entities, department, priority, actionable_task, confidence_score, reasoning."
)


class AdvancedRAGComplaintAnalyzer:
    def __init__(
        self,
        dataset_path: str | Path = RAW_DATA_PATH,
        knowledge_base: Optional[pd.DataFrame] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: str = ACTIVE_MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_retries: int = MAX_RETRIES,
        timeout_seconds: int = API_TIMEOUT_SECONDS,
        top_k: int = RAG_TOP_K,
        enable_generation: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.provider = provider or ACTIVE_LLM_PROVIDER
        self.api_key = api_key if api_key is not None else ACTIVE_LLM_API_KEY
        self.base_url = base_url if base_url is not None else ACTIVE_LLM_BASE_URL
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k
        self.enable_generation = enable_generation
        self.knowledge_base = knowledge_base.copy() if knowledge_base is not None else pd.read_csv(self.dataset_path)
        self.similarity = ComplaintSimilaritySearch(dataset=self.knowledge_base, top_k=self.top_k)
        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout_seconds}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs) if self.api_key else None

    def retrieve(self, complaint_text: str, top_k: Optional[int] = None):
        return self.similarity.retrieve(complaint_text, top_k=top_k)

    def summarize_image_complaint(
        self,
        image_bytes: bytes,
        image_mime_type: str,
        complaint_text: str = "",
    ) -> str:
        if not self.client:
            raise ValueError("No LLM API key configured for image analysis.")
        image_base64 = b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Extract the complaint details visible in this customer-provided image. "
            "Return a short plain-text complaint summary covering the product or service, "
            "the problem, and any urgency. If the image shows physical damage, explicitly say whether it looks like "
            "a damaged product, damaged shipment, broken screen, shattered glass, cracked item, dented product, or packaging damage. "
            "For visible broken or shattered product photos from shopping or delivery complaints, strongly prefer wording like "
            "'damaged product', 'damaged shipment', 'shattered glass', 'cracked screen', or 'product arrived broken'. "
            "Write it as complaint text that downstream classifiers can use. Do not return JSON."
        )
        if complaint_text.strip():
            prompt += f"\n\nExisting customer text:\n{complaint_text.strip()}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You convert complaint screenshots or photos into concise complaint text for downstream classifiers."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}},
                    ],
                },
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty response received from image summarization call.")
        return content

    def _build_messages(
        self,
        complaint_text: str,
        fallback: ComplaintAnalysis,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> list[dict[str, object]]:
        retrieved_payload = json.dumps([item.model_dump(mode="json") for item in fallback.retrieved_examples], indent=2)
        fallback_payload = json.dumps(fallback.model_dump(mode="json"), indent=2)
        user_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": f"Complaint:\n{complaint_text}\n\nRetrieval-informed baseline:\n{fallback_payload}\n\nSimilar complaints:\n{retrieved_payload}\n\nReturn JSON only.",
            }
        ]
        if image_bytes and image_mime_type:
            image_base64 = b64encode(image_bytes).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                }
            )
        return [
            {"role": "system", "content": PROMPT_TEMPLATE + " Use only allowed departments Billing, Logistics, Technical Support, Customer Service and priorities Low, Medium, High, Critical. If the complaint or image describes broken, cracked, shattered, dented, or physically damaged delivered products, prefer Logistics with a damaged-product or damaged-shipment issue unless strong contrary evidence exists."},
            {"role": "user", "content": user_content},
        ]

    def _fallback_from_retrieval(self, complaint_text: str) -> ComplaintAnalysis:
        baseline = build_rule_based_analysis(complaint_text)
        retrieved_examples = self.retrieve(complaint_text, top_k=self.top_k)
        if not retrieved_examples:
            return baseline.model_copy(update={"analysis_mode": "rag-retrieval"})
        weights = [item.similarity for item in retrieved_examples]
        average_weight = sum(weights) / len(weights)
        strong_matches = sum(1 for weight in weights[:3] if weight >= 0.6)
        core_issue = weighted_vote([item.core_issue for item in retrieved_examples], weights, fallback=baseline.core_issue)
        department = weighted_vote([item.department for item in retrieved_examples], weights, fallback=baseline.department)
        priority = weighted_vote([item.priority for item in retrieved_examples], weights, fallback=baseline.priority)
        confidence = calculate_confidence((0.7 * baseline.confidence_score) + (0.9 * max(weights)) + (0.7 * average_weight) + (0.15 * strong_matches), 2.8)
        entities = sorted(set(baseline.detected_entities + extract_entities(complaint_text)))
        actionable_task = ISSUE_RULES.get(core_issue, {}).get("task", baseline.actionable_task)
        return ComplaintAnalysis(
            core_issue=core_issue,
            detected_entities=entities,
            department=department,
            priority=priority,
            actionable_task=actionable_task,
            confidence_score=confidence,
            reasoning=build_reasoning(core_issue, department, priority, entities, "retrieval-augmented"),
            analysis_mode="rag-retrieval",
            supporting_evidence=baseline.supporting_evidence + [f"Retrieved {len(retrieved_examples)} semantically similar complaints via FAISS.", f"Top retrieval similarity: {max(weights):.2f}; average similarity: {average_weight:.2f}."],
            retrieved_examples=retrieved_examples,
        )

    def _call_llm(
        self,
        complaint_text: str,
        fallback: ComplaintAnalysis,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        if not self.client:
            raise ValueError("No LLM API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY.")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(complaint_text, fallback, image_bytes=image_bytes, image_mime_type=image_mime_type),
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from LLM.")
        return content

    def analyze(
        self,
        complaint_text: str,
        enable_generation: Optional[bool] = None,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> ComplaintAnalysis:
        fallback = self._fallback_from_retrieval(complaint_text)
        if enable_generation is None:
            enable_generation = self.enable_generation
        if not enable_generation or not self.client:
            return fallback
        last_error = "Unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_content = self._call_llm(complaint_text, fallback, image_bytes=image_bytes, image_mime_type=image_mime_type)
                payload = safe_json_loads(raw_content)
                is_valid, sanitized, message = validate_output_json(payload)
                if not is_valid:
                    raise ValueError(message)
                merged_entities = sorted(set(fallback.detected_entities + sanitized.get("detected_entities", [])))
                final_payload = {key: value for key, value in sanitized.items() if key not in {"detected_entities", "reasoning"}}
                return ComplaintAnalysis(
                    **final_payload,
                    detected_entities=merged_entities,
                    reasoning=sanitized.get("reasoning") or fallback.reasoning,
                    analysis_mode="hybrid-rag-llm",
                    supporting_evidence=fallback.supporting_evidence,
                    retrieved_examples=fallback.retrieved_examples,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Attempt %s/%s failed for provider %s: %s", attempt, self.max_retries, self.provider, exc)
                if attempt < self.max_retries:
                    time.sleep(1.2 * attempt)
        return fallback.model_copy(update={"supporting_evidence": fallback.supporting_evidence + [f"LLM synthesis fallback triggered: {last_error}"]})


HybridRAGComplaintAnalyzer = AdvancedRAGComplaintAnalyzer
