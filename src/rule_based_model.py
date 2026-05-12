"""Rule-based complaint analyzer using keyword and regex matching."""

from __future__ import annotations

import re
from typing import List, Tuple

from config import ALLOWED_DEPARTMENTS, DEPARTMENT_DEFAULT_TASKS
from src.schemas import ComplaintAnalysis
from src.utils import build_reasoning, calculate_confidence, extract_entities, highest_priority, normalize_text

ISSUE_RULES = {
    "Refund delay": {"department": "Billing", "keywords": ["refund", "reimbursement", "money back", "chargeback"], "task": "Review refund transaction, confirm approval status, and process the pending reimbursement.", "base_priority": "Medium"},
    "Wrong charge": {"department": "Billing", "keywords": ["overcharged", "double charged", "wrong amount", "unauthorized charge", "charged twice"], "task": "Audit the invoice, reverse the incorrect charge, and notify the customer of the correction window.", "base_priority": "High"},
    "Invoice error": {"department": "Billing", "keywords": ["invoice", "billing statement", "tax mismatch", "payment failed"], "task": "Validate the billing statement, correct any tax or payment mismatch, and re-issue the invoice.", "base_priority": "Medium"},
    "Delivery delay": {"department": "Logistics", "keywords": ["late delivery", "delayed", "not arrived", "stuck in transit"], "task": "Trace the shipment, coordinate with the courier, and send a revised delivery timeline.", "base_priority": "High"},
    "Damaged shipment": {"department": "Logistics", "keywords": ["damaged package", "broken on arrival", "crushed box", "arrived damaged", "damaged box", "packaging damage", "courier damage"], "task": "Arrange a replacement shipment, log a courier damage claim, and confirm return or disposal instructions.", "base_priority": "High"},
    "Damaged product": {"department": "Logistics", "keywords": ["shattered glass", "broken glass", "cracked screen", "smashed screen", "damaged product", "broken item", "physically damaged", "fractured glass", "dented product", "unsafe to use"], "task": "Approve a damaged-product replacement, inspect packaging and courier handling, and confirm return or disposal instructions.", "base_priority": "High"},
    "Wrong item delivered": {"department": "Logistics", "keywords": ["wrong item", "incorrect product", "received different", "wrong color"], "task": "Initiate reverse pickup, reserve the correct stock, and dispatch the replacement order.", "base_priority": "Medium"},
    "App or website failure": {"department": "Technical Support", "keywords": ["app crash", "website error", "500 error", "not loading", "bug"], "task": "Reproduce the failure, assign the defect to engineering, and confirm when service is restored.", "base_priority": "High"},
    "Login or account access issue": {"department": "Technical Support", "keywords": ["cannot login", "password reset", "account locked", "otp not received"], "task": "Resolve the account access blocker, validate identity, and restore login access for the customer.", "base_priority": "High"},
    "Device or integration issue": {"department": "Technical Support", "keywords": ["sync failed", "api error", "device disconnected", "integration"], "task": "Diagnose the integration failure, apply the compatibility fix, and monitor for repeat incidents.", "base_priority": "Medium"},
    "Poor support experience": {"department": "Customer Service", "keywords": ["rude", "unhelpful", "no response", "kept waiting", "ignored", "not responding", "no reply"], "task": "Escalate the case to the support quality lead, review the interaction trail, and arrange a callback.", "base_priority": "High"},
    "Cancellation or return issue": {"department": "Customer Service", "keywords": ["cancel", "return request", "policy unclear", "no pickup"], "task": "Review the cancellation or return case, apply policy exceptions if needed, and confirm next steps.", "base_priority": "Medium"},
    "Security or fraud concern": {"department": "Technical Support", "keywords": ["fraud", "hacked", "security breach", "data leak", "stolen account"], "task": "Open a security incident, freeze risky activity, and escalate the case for immediate containment.", "base_priority": "Critical"},
}

PATTERN_RULES: List[Tuple[str, str, int]] = [
    (r"\b(double|duplicate)\s+charg(ed|e)\b", "Wrong charge", 3),
    (r"\b(refund|money\s*back)\b", "Refund delay", 2),
    (r"\b(late|delay(ed)?|not\s+arrived)\b", "Delivery delay", 2),
    (r"\b(shattered|smashed)\s+(glass|screen)\b", "Damaged product", 4),
    (r"\b(cracked|broken|damaged|defective)\s+(product|device|item|screen|glass)\b", "Damaged product", 3),
    (r"\b(damaged|broken|cracked|defective)\b", "Damaged shipment", 2),
    (r"\b(login|locked\s+out|otp)\b", "Login or account access issue", 2),
    (r"\b(crash|bug|error\s*500|not\s+loading)\b", "App or website failure", 2),
    (r"\b(rude|ignored|no\s+response|unhelpful|not\s+responding)\b", "Poor support experience", 2),
    (r"\b(fraud|breach|hacked|data\s+leak)\b", "Security or fraud concern", 4),
]

CRITICAL_PRIORITY_TERMS = {"fraud", "hacked", "security breach", "data leak", "stolen", "legal action", "lawsuit"}
HIGH_PRIORITY_TERMS = {"urgent", "immediately", "asap", "critical", "frustrated", "angry", "legal", "escalate"}
MEDIUM_PRIORITY_TERMS = {"soon", "pending", "waiting", "follow up", "reminder"}


def _detect_priority(text: str, default_priority: str) -> tuple[str, float]:
    critical_hits = sum(1 for term in CRITICAL_PRIORITY_TERMS if term in text)
    high_hits = sum(1 for term in HIGH_PRIORITY_TERMS if term in text)
    medium_hits = sum(1 for term in MEDIUM_PRIORITY_TERMS if term in text)
    if critical_hits >= 1:
        return "Critical", 1.0
    if "!!!" in text or high_hits >= 2:
        return "High", 0.98
    if high_hits == 1:
        return highest_priority(default_priority, "High"), 0.9
    if medium_hits >= 1:
        return highest_priority(default_priority, "Medium"), 0.75
    return default_priority, 0.65 if default_priority == "Critical" else 0.6


def build_rule_based_analysis(complaint_text: str) -> ComplaintAnalysis:
    text = normalize_text(complaint_text)
    entities = extract_entities(text)
    if not text:
        return ComplaintAnalysis(
            core_issue="Insufficient complaint details",
            detected_entities=entities,
            department="Customer Service",
            priority="Low",
            actionable_task=DEPARTMENT_DEFAULT_TASKS["Customer Service"],
            confidence_score=0.1,
            reasoning="The complaint text was empty, so manual triage is required.",
            analysis_mode="rule-based",
            supporting_evidence=["No usable complaint text was provided."],
        )

    issue_scores: dict[str, float] = {issue: 0.0 for issue in ISSUE_RULES}
    evidence: list[str] = []
    raw_matches = 0
    for issue, rule in ISSUE_RULES.items():
        for keyword in rule["keywords"]:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                issue_scores[issue] += 1.5
                raw_matches += 1
                evidence.append(f"Matched keyword '{keyword}' to {issue}.")
    for pattern, mapped_issue, weight in PATTERN_RULES:
        if re.search(pattern, text):
            issue_scores[mapped_issue] += weight
            raw_matches += 1
            evidence.append(f"Matched pattern '{pattern}' to {mapped_issue}.")

    best_issue = max(issue_scores, key=issue_scores.get)
    best_score = issue_scores[best_issue]
    if best_score <= 0:
        priority, priority_conf = _detect_priority(text, "Low")
        return ComplaintAnalysis(
            core_issue="General complaint requiring manual triage",
            detected_entities=entities,
            department="Customer Service",
            priority=priority,
            actionable_task=DEPARTMENT_DEFAULT_TASKS["Customer Service"],
            confidence_score=calculate_confidence(0.8 * priority_conf, 3.0),
            reasoning=build_reasoning("General complaint requiring manual triage", "Customer Service", priority, entities, "rule-based"),
            analysis_mode="rule-based",
            supporting_evidence=["No direct rule matched, so the case was routed to manual triage."],
        )

    selected_rule = ISSUE_RULES[best_issue]
    department = selected_rule["department"] if selected_rule["department"] in ALLOWED_DEPARTMENTS else "Customer Service"
    priority, priority_conf = _detect_priority(text, selected_rule["base_priority"])
    confidence = calculate_confidence(best_score + (0.8 * priority_conf) + (0.3 * min(raw_matches, 4)), 8.0)
    return ComplaintAnalysis(
        core_issue=best_issue,
        detected_entities=entities,
        department=department,
        priority=priority,
        actionable_task=selected_rule["task"],
        confidence_score=confidence,
        reasoning=build_reasoning(best_issue, department, priority, entities, "rule-based"),
        analysis_mode="rule-based",
        supporting_evidence=evidence[:4] or ["Rule engine routed the complaint based on deterministic patterns."],
    )


def analyze_complaint_rule_based(complaint_text: str) -> dict[str, object]:
    return build_rule_based_analysis(complaint_text).model_dump(mode="json")
