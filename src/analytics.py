"""Analytics module for complaint warehouse exploration using matplotlib and seaborn."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils import load_dataset, normalize_text

sns.set_theme(style="whitegrid")


def complaint_length_distribution(df: pd.DataFrame | None = None):
    dataset = df.copy() if df is not None else load_dataset()
    lengths = dataset["complaint_text"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(lengths, bins=25, kde=True, color="#5b8def", ax=ax)
    ax.set_title("Complaint Length Distribution")
    ax.set_xlabel("Word count")
    return fig


def department_distribution(df: pd.DataFrame | None = None):
    dataset = df.copy() if df is not None else load_dataset()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=dataset, x="true_department", palette="crest", ax=ax)
    ax.set_title("Department Distribution")
    ax.set_xlabel("Department")
    ax.set_ylabel("Complaints")
    ax.tick_params(axis="x", rotation=15)
    return fig


def priority_distribution(df: pd.DataFrame | None = None):
    dataset = df.copy() if df is not None else load_dataset()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=dataset, x="true_priority", order=["Low", "Medium", "High", "Critical"], palette="magma", ax=ax)
    ax.set_title("Priority Distribution")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Complaints")
    return fig


def keyword_frequency(df: pd.DataFrame | None = None, top_n: int = 15):
    dataset = df.copy() if df is not None else load_dataset()
    tokens = []
    for text in dataset["complaint_text"].fillna("").astype(str):
        cleaned = normalize_text(text)
        tokens.extend([token for token in cleaned.split() if token not in ENGLISH_STOP_WORDS and len(token) > 3])
    plot_df = pd.DataFrame(Counter(tokens).most_common(top_n), columns=["keyword", "count"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="count", y="keyword", palette="flare", ax=ax)
    ax.set_title("Most Common Complaint Keywords")
    return fig
