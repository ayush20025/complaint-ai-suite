"""Compatibility wrapper for the upgraded rule-based analyzer."""

from src.rule_based_model import ISSUE_RULES, analyze_complaint_rule_based, build_rule_based_analysis

__all__ = ["ISSUE_RULES", "analyze_complaint_rule_based", "build_rule_based_analysis"]
