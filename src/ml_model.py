"""Classical machine-learning complaint analyzer built on TF-IDF and logistic regression."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import DEPARTMENT_DEFAULT_TASKS, ML_CLASSIFIER_PATH, RAW_DATA_PATH
from src.rule_based_model import ISSUE_RULES
from src.schemas import ComplaintAnalysis
from src.utils import build_reasoning, calculate_confidence, extract_entities, normalize_text


def _make_text_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1400, class_weight="balanced", random_state=42)),
        ]
    )


class MLComplaintAnalyzer:
    def __init__(self, dataset_path: str | Path = RAW_DATA_PATH, model_path: str | Path = ML_CLASSIFIER_PATH) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.issue_model = _make_text_classifier()
        self.department_model = _make_text_classifier()
        self.priority_model = _make_text_classifier()
        self._is_fitted = False

    def fit(self, df: Optional[pd.DataFrame] = None, persist: bool = True) -> "MLComplaintAnalyzer":
        training_df = df.copy() if df is not None else pd.read_csv(self.dataset_path)
        required_columns = {"complaint_text", "true_department", "true_priority", "true_core_issue"}
        missing = required_columns - set(training_df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        text_series = training_df["complaint_text"].fillna("").astype(str)
        self.issue_model.fit(text_series, training_df["true_core_issue"].astype(str))
        self.department_model.fit(text_series, training_df["true_department"].astype(str))
        self.priority_model.fit(text_series, training_df["true_priority"].astype(str))
        self._is_fitted = True
        if persist:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"issue_model": self.issue_model, "department_model": self.department_model, "priority_model": self.priority_model}, self.model_path)
        return self

    def _ensure_fitted(self) -> None:
        if self._is_fitted:
            return
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            self.issue_model = payload["issue_model"]
            self.department_model = payload["department_model"]
            self.priority_model = payload["priority_model"]
            self._is_fitted = True
            return
        self.fit()

    @staticmethod
    def _predict_with_confidence(model: Pipeline, text: str) -> tuple[str, float]:
        probabilities = model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        return str(model.classes_[best_index]), round(float(probabilities[best_index]), 3)

    def analyze(self, complaint_text: str) -> ComplaintAnalysis:
        self._ensure_fitted()
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
                reasoning="The classifier received an empty complaint payload, so it defaulted to manual triage.",
                analysis_mode="ml-baseline",
                supporting_evidence=["The classifier received an empty complaint payload."],
            )
        issue, issue_conf = self._predict_with_confidence(self.issue_model, text)
        department, dept_conf = self._predict_with_confidence(self.department_model, text)
        priority, priority_conf = self._predict_with_confidence(self.priority_model, text)
        actionable_task = ISSUE_RULES.get(issue, {}).get("task", DEPARTMENT_DEFAULT_TASKS.get(department, DEPARTMENT_DEFAULT_TASKS["Customer Service"]))
        confidence = calculate_confidence(issue_conf + dept_conf + priority_conf, 3.0)
        return ComplaintAnalysis(
            core_issue=issue,
            detected_entities=entities,
            department=department,
            priority=priority,
            actionable_task=actionable_task,
            confidence_score=confidence,
            reasoning=build_reasoning(issue, department, priority, entities, "machine-learning"),
            analysis_mode="ml-baseline",
            supporting_evidence=[f"Issue confidence: {issue_conf:.2f}", f"Department confidence: {dept_conf:.2f}", f"Priority confidence: {priority_conf:.2f}"],
        )


def analyze_complaint_ml(complaint_text: str, analyzer: Optional[MLComplaintAnalyzer] = None) -> dict[str, object]:
    analyzer = analyzer or MLComplaintAnalyzer()
    return analyzer.analyze(complaint_text).model_dump(mode="json")
