"""Backward-compatible evaluation wrappers for the upgraded model comparison pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model_comparison import ModelComparisonRunner


def evaluate_models(df: pd.DataFrame, include_ml: bool = True, include_rag: bool = True, rag_enable_generation: bool = False, **_: object):
    artifacts = ModelComparisonRunner().run(df)
    tables = {"metrics": artifacts.metrics, "rule_based_confusion": artifacts.confusion["Rule-Based"]}
    tables["external_validation_metrics"] = artifacts.external_validation_metrics
    if include_ml:
        tables["ml_confusion"] = artifacts.confusion["ML Baseline"]
    if include_rag:
        tables["rag_confusion"] = artifacts.confusion["RAG + LLM"]
    return tables


def save_evaluation_tables(tables: dict[str, pd.DataFrame], output_excel_path: str | Path) -> Path:
    output_excel_path = Path(output_excel_path)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel_path) as writer:
        for sheet_name, table_df in tables.items():
            table_df.to_excel(writer, sheet_name=sheet_name[:31])
    return output_excel_path
