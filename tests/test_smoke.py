"""Smoke tests for the complaint AI project."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.dashboard_utils import build_recommended_analysis, trio_analysis
from src.ml_model import MLComplaintAnalyzer
from src.model_comparison import ModelComparisonRunner
from src.rag_pipeline import AdvancedRAGComplaintAnalyzer
from src.rule_based_model import build_rule_based_analysis
from src.schemas import ComplaintAnalysis
from src.utils import derive_template_group, load_cfpb_dataset, load_dataset, load_validation_dataset


class ComplaintAISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.validation = load_validation_dataset()
        cls.cfpb = load_cfpb_dataset()

    def test_schema_validation_normalizes_priority(self) -> None:
        payload = ComplaintAnalysis(
            core_issue="Refund delay",
            detected_entities=["billing"],
            department="Billing",
            priority="High",
            actionable_task="Process refund.",
            confidence_score=0.81,
            reasoning="Billing complaint.",
        )
        self.assertEqual(payload.department, "Billing")
        self.assertEqual(payload.priority, "High")

    def test_rule_routing_for_security_case(self) -> None:
        result = build_rule_based_analysis("My account was hacked and there is suspicious payment activity on it.")
        self.assertEqual(result.department, "Technical Support")
        self.assertEqual(result.priority, "Critical")

    def test_ml_analyzer_runs(self) -> None:
        analyzer = MLComplaintAnalyzer().fit(self.dataset.head(300), persist=False)
        result = analyzer.analyze("The mobile app crashes at checkout and I cannot complete payment.")
        self.assertTrue(result.department in {"Technical Support", "Billing"})
        self.assertGreaterEqual(result.confidence_score, 0.0)

    def test_rag_fallback_runs_without_generation(self) -> None:
        analyzer = AdvancedRAGComplaintAnalyzer(knowledge_base=self.dataset.head(500), enable_generation=False)
        result = analyzer.analyze("My order arrived late and damaged.", enable_generation=False)
        self.assertEqual(result.analysis_mode, "rag-retrieval")
        self.assertTrue(len(result.retrieved_examples) > 0)

    def test_recommended_analysis_falls_back_to_highest_confidence_on_disagreement(self) -> None:
        results = {
            "Rule-Based": ComplaintAnalysis(
                core_issue="Refund delay",
                detected_entities=["billing"],
                department="Billing",
                priority="Medium",
                actionable_task="Process refund.",
                confidence_score=0.62,
                reasoning="Rule output.",
            ),
            "ML Baseline": ComplaintAnalysis(
                core_issue="Damaged shipment",
                detected_entities=["logistics"],
                department="Logistics",
                priority="High",
                actionable_task="Arrange replacement.",
                confidence_score=0.74,
                reasoning="ML output.",
            ),
            "RAG + LLM": ComplaintAnalysis(
                core_issue="Poor support experience",
                detected_entities=["service"],
                department="Customer Service",
                priority="Medium",
                actionable_task="Escalate support quality review.",
                confidence_score=0.68,
                reasoning="RAG output.",
            ),
        }
        recommended = build_recommended_analysis(results)
        self.assertEqual(recommended.core_issue, "Damaged shipment")
        self.assertEqual(recommended.department, "Logistics")
        self.assertLess(recommended.confidence_score, 0.74)

    def test_image_only_without_generation_does_not_use_placeholder_prompt(self) -> None:
        with patch("app.dashboard_utils.get_rag_model") as mock_get_rag_model, patch("app.dashboard_utils.get_ml_model") as mock_get_ml_model:
            mock_rag_model = mock_get_rag_model.return_value
            mock_rag_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Insufficient complaint details",
                detected_entities=[],
                department="Customer Service",
                priority="Low",
                actionable_task="Open a manual triage ticket, capture missing context, and route the case to the right team.",
                confidence_score=0.1,
                reasoning="No text available.",
            )
            mock_ml_model = mock_get_ml_model.return_value
            mock_ml_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Insufficient complaint details",
                detected_entities=[],
                department="Customer Service",
                priority="Low",
                actionable_task="Open a manual triage ticket, capture missing context, and route the case to the right team.",
                confidence_score=0.1,
                reasoning="No text available.",
            )

            results = trio_analysis("", use_embedded_key=False, image_bytes=b"fake-image", image_mime_type="image/png")

            mock_ml_model.analyze.assert_called_once_with("")
            mock_rag_model.analyze.assert_called_once_with("", enable_generation=False, image_bytes=b"fake-image", image_mime_type="image/png")
            self.assertEqual(results["Rule-Based"].core_issue, "Insufficient complaint details")
            self.assertEqual(results["ML Baseline"].core_issue, "Insufficient complaint details")

    def test_image_summary_is_shared_with_all_models_when_generation_is_enabled(self) -> None:
        with patch("app.dashboard_utils.get_rag_model") as mock_get_rag_model, patch("app.dashboard_utils.get_ml_model") as mock_get_ml_model:
            mock_rag_model = mock_get_rag_model.return_value
            mock_rag_model.summarize_image_complaint.return_value = "The invoice shows a duplicate charge and refund request."
            mock_rag_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Wrong charge",
                detected_entities=["billing"],
                department="Billing",
                priority="High",
                actionable_task="Reverse the duplicate charge.",
                confidence_score=0.83,
                reasoning="Vision-assisted RAG output.",
            )
            mock_ml_model = mock_get_ml_model.return_value
            mock_ml_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Wrong charge",
                detected_entities=["billing"],
                department="Billing",
                priority="High",
                actionable_task="Reverse the duplicate charge.",
                confidence_score=0.8,
                reasoning="ML output.",
            )

            results = trio_analysis("", use_embedded_key=True, image_bytes=b"fake-image", image_mime_type="image/png")

            mock_rag_model.summarize_image_complaint.assert_called_once()
            mock_ml_model.analyze.assert_called_once_with("The invoice shows a duplicate charge and refund request.")
            self.assertEqual(results["Rule-Based"].department, "Billing")

    def test_image_summary_is_merged_with_user_text_when_both_exist(self) -> None:
        with patch("app.dashboard_utils.get_rag_model") as mock_get_rag_model, patch("app.dashboard_utils.get_ml_model") as mock_get_ml_model:
            mock_rag_model = mock_get_rag_model.return_value
            mock_rag_model.summarize_image_complaint.return_value = "The image shows shattered glass and a damaged delivered item."
            mock_rag_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Damaged shipment",
                detected_entities=["product", "logistics"],
                department="Logistics",
                priority="High",
                actionable_task="Arrange a replacement shipment.",
                confidence_score=0.84,
                reasoning="Vision-assisted RAG output.",
            )
            mock_ml_model = mock_get_ml_model.return_value
            mock_ml_model.analyze.return_value = ComplaintAnalysis(
                core_issue="Damaged shipment",
                detected_entities=["product", "logistics"],
                department="Logistics",
                priority="High",
                actionable_task="Arrange a replacement shipment.",
                confidence_score=0.78,
                reasoning="ML output.",
            )

            trio_analysis(
                "please check the attached image and tell",
                use_embedded_key=True,
                image_bytes=b"fake-image",
                image_mime_type="image/png",
            )

            expected_text = "please check the attached image and tell\n\nImage findings: The image shows shattered glass and a damaged delivered item."
            mock_ml_model.analyze.assert_called_once_with(expected_text)
            mock_rag_model.analyze.assert_called_once_with(
                expected_text,
                enable_generation=True,
                image_bytes=b"fake-image",
                image_mime_type="image/png",
            )

    def test_grouped_split_holds_out_templates(self) -> None:
        runner = ModelComparisonRunner(test_size=0.25, random_state=42)
        train_df, test_df = runner._split_dataset(self.dataset)
        train_groups = {derive_template_group(text) for text in train_df["complaint_text"].astype(str)}
        test_groups = {derive_template_group(text) for text in test_df["complaint_text"].astype(str)}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_handwritten_validation_dataset_exists(self) -> None:
        self.assertGreaterEqual(len(self.validation), 5_000)
        self.assertTrue({"complaint_text", "true_department", "true_priority", "true_core_issue"}.issubset(self.validation.columns))

    def test_main_and_cfpb_datasets_are_scaled(self) -> None:
        self.assertGreaterEqual(len(self.dataset), 150_000)
        self.assertGreaterEqual(len(self.cfpb), 2_500)
        self.assertTrue({"product", "consumer_complaint_narrative", "complaint_text", "true_department"}.issubset(self.cfpb.columns))


if __name__ == "__main__":
    unittest.main()
