"""Model comparison experiments across rule-based, ML, and RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from src.ml_model import MLComplaintAnalyzer
from src.rag_pipeline import AdvancedRAGComplaintAnalyzer
from src.rule_based_model import build_rule_based_analysis
from src.utils import derive_template_group, load_dataset, load_validation_dataset

sns.set_theme(style="whitegrid")


@dataclass
class ComparisonArtifacts:
    metrics: pd.DataFrame
    confusion: dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    external_validation_metrics: pd.DataFrame


class ModelComparisonRunner:
    def __init__(self, random_state: int = 42, test_size: float = 0.25) -> None:
        self.random_state = random_state
        self.test_size = test_size

    def _split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        grouped = dataset.copy()
        grouped["evaluation_group"] = grouped["complaint_text"].astype(str).map(derive_template_group)
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(splitter.split(grouped, groups=grouped["evaluation_group"]))
        train_df = grouped.iloc[train_idx].drop(columns=["evaluation_group"]).reset_index(drop=True)
        test_df = grouped.iloc[test_idx].drop(columns=["evaluation_group"]).reset_index(drop=True)
        return train_df, test_df

    @staticmethod
    def _metric_row(model_name: str, y_true: pd.Series, y_pred: pd.Series, confidence: pd.Series) -> dict[str, float | str]:
        return {
            "model": model_name,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "avg_confidence": round(float(confidence.mean()), 4),
        }

    def run(self, df: Optional[pd.DataFrame] = None) -> ComparisonArtifacts:
        dataset = df.copy() if df is not None else load_dataset()
        train_df, test_df = self._split_dataset(dataset)
        ml_model = MLComplaintAnalyzer().fit(train_df)
        rag_model = AdvancedRAGComplaintAnalyzer(knowledge_base=train_df, enable_generation=False)
        labels = sorted(dataset["true_department"].dropna().unique().tolist())
        outputs = {
            "Rule-Based": test_df["complaint_text"].apply(build_rule_based_analysis),
            "ML Baseline": test_df["complaint_text"].apply(ml_model.analyze),
            "RAG + LLM": test_df["complaint_text"].apply(lambda text: rag_model.analyze(text, enable_generation=False)),
        }
        records = []
        confusion: dict[str, pd.DataFrame] = {}
        prediction_frames = []
        for model_name, series in outputs.items():
            dept_pred = series.apply(lambda row: row.department)
            confidence_series = series.apply(lambda row: row.confidence_score)
            true_series = test_df["true_department"].reset_index(drop=True)
            records.append(self._metric_row(model_name, true_series, dept_pred, confidence_series))
            confusion[model_name] = pd.DataFrame(confusion_matrix(true_series, dept_pred, labels=labels), index=labels, columns=labels)
            frame = pd.DataFrame({
                "model": model_name,
                "complaint_text": test_df["complaint_text"].values,
                "true_department": true_series.values,
                "predicted_department": dept_pred.values,
                "confidence_score": confidence_series.values,
            })
            frame["is_wrong"] = frame["true_department"] != frame["predicted_department"]
            prediction_frames.append(frame)
        return ComparisonArtifacts(
            metrics=pd.DataFrame(records),
            confusion=confusion,
            predictions=pd.concat(prediction_frames, ignore_index=True),
            external_validation_metrics=self._run_external_validation(ml_model, rag_model),
        )

    def _run_external_validation(self, ml_model: MLComplaintAnalyzer, rag_model: AdvancedRAGComplaintAnalyzer) -> pd.DataFrame:
        validation_df = load_validation_dataset()
        outputs = {
            "Rule-Based": validation_df["complaint_text"].apply(build_rule_based_analysis),
            "ML Baseline": validation_df["complaint_text"].apply(ml_model.analyze),
            "RAG + LLM": validation_df["complaint_text"].apply(lambda text: rag_model.analyze(text, enable_generation=False)),
        }
        rows = []
        y_true = validation_df["true_department"].reset_index(drop=True)
        for model_name, series in outputs.items():
            y_pred = series.apply(lambda row: row.department)
            confidence = series.apply(lambda row: row.confidence_score)
            rows.append(self._metric_row(model_name, y_true, y_pred, confidence))
        external_metrics = pd.DataFrame(rows)
        external_metrics.insert(1, "evaluation_set", "handwritten_validation")
        return external_metrics

    @staticmethod
    def plot_metrics(metrics_df: pd.DataFrame, value_vars: list[str] | None = None):
        metrics = value_vars or ["accuracy", "precision", "recall", "f1_score", "avg_confidence"]
        melted = metrics_df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="score")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Performance Comparison")
        ax.tick_params(axis="x", rotation=20)
        return fig

    @staticmethod
    def plot_confusion(confusion_df: pd.DataFrame, title: str):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        return fig
