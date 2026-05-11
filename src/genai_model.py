"""Backward-compatible accessors for the advanced RAG complaint analyzer."""

from src.rag_pipeline import AdvancedRAGComplaintAnalyzer, HybridRAGComplaintAnalyzer

GenAIComplaintAnalyzer = AdvancedRAGComplaintAnalyzer


def analyze_complaint_rag(complaint_text: str, analyzer=None, enable_generation=None) -> dict[str, object]:
    analyzer = analyzer or AdvancedRAGComplaintAnalyzer()
    return analyzer.analyze(complaint_text, enable_generation=enable_generation).model_dump(mode="json")
