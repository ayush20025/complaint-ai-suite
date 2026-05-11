"""Complaint clustering analysis using TF-IDF and KMeans."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DEFAULT_CLUSTER_COUNT
from src.utils import load_dataset

sns.set_theme(style="whitegrid")


@dataclass
class ClusteringArtifacts:
    assignments: pd.DataFrame
    summary: pd.DataFrame


class ComplaintClusterAnalyzer:
    def __init__(self, n_clusters: int = DEFAULT_CLUSTER_COUNT, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1200)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)

    def fit_predict(self, df: pd.DataFrame | None = None) -> ClusteringArtifacts:
        dataset = df.copy() if df is not None else load_dataset()
        features = self.vectorizer.fit_transform(dataset["complaint_text"].fillna("").astype(str))
        dataset = dataset.copy()
        dataset["cluster_id"] = self.model.fit_predict(features)
        terms = self.vectorizer.get_feature_names_out()
        cluster_terms = []
        for cluster_id in range(self.n_clusters):
            centroid = self.model.cluster_centers_[cluster_id]
            top_terms = [terms[index] for index in centroid.argsort()[-5:][::-1]]
            cluster_terms.append({"cluster_id": cluster_id, "cluster_label": ", ".join(top_terms), "size": int((dataset["cluster_id"] == cluster_id).sum())})
        summary = pd.DataFrame(cluster_terms).sort_values("size", ascending=False)
        return ClusteringArtifacts(assignments=dataset, summary=summary)

    def plot_clusters(self, assignments: pd.DataFrame):
        features = self.vectorizer.transform(assignments["complaint_text"].fillna("").astype(str))
        reduced = PCA(n_components=2, random_state=self.random_state).fit_transform(features.toarray())
        plot_df = assignments[["cluster_id"]].copy()
        plot_df["x"] = reduced[:, 0]
        plot_df["y"] = reduced[:, 1]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x="x", y="y", hue="cluster_id", palette="viridis", ax=ax, s=55)
        ax.set_title("Complaint Cluster Map")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        return fig
