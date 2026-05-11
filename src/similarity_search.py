"""FAISS-backed complaint similarity search."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from config import EMBEDDING_META_PATH, RAW_DATA_PATH, SIMILARITY_TOP_K, VECTOR_INDEX_PATH
from src.embeddings import ComplaintEmbedder
from src.schemas import RetrievedExample
from src.utils import get_logger, load_dataset

logger = get_logger(__name__)


class ComplaintSimilaritySearch:
    """Build and query a FAISS index over complaint embeddings."""

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        dataset_path: str | Path = RAW_DATA_PATH,
        index_path: str | Path = VECTOR_INDEX_PATH,
        metadata_path: str | Path = EMBEDDING_META_PATH,
        top_k: int = SIMILARITY_TOP_K,
        persist_artifacts: Optional[bool] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.top_k = top_k
        self.embedder = ComplaintEmbedder()
        self.dataset = dataset.copy() if dataset is not None else load_dataset()
        self.metadata = self.dataset.copy()
        self.index: faiss.Index | None = None
        self.persist_artifacts = persist_artifacts if persist_artifacts is not None else dataset is None
        self._load_or_build_index()

    def _build_index(self) -> None:
        self.metadata = self.dataset.copy()
        texts = self.metadata["complaint_text"].fillna("").astype(str).tolist()
        embedding_result = self.embedder.encode(texts)
        vectors = embedding_result.vectors.astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        if self.persist_artifacts:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.metadata.to_parquet(self.metadata_path, index=False)
        logger.info("Built FAISS index with %s complaints using %s backend", len(texts), embedding_result.backend)

    def _load_or_build_index(self) -> None:
        if not self.persist_artifacts:
            self._build_index()
            return
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = pd.read_parquet(self.metadata_path)
            if len(self.metadata) != len(self.dataset):
                logger.info(
                    "Existing FAISS metadata size %s does not match dataset size %s. Rebuilding index.",
                    len(self.metadata),
                    len(self.dataset),
                )
                self._build_index()
            return
        self._build_index()

    def retrieve(self, complaint_text: str, top_k: Optional[int] = None) -> list[RetrievedExample]:
        if self.index is None:
            self._build_index()
        limit = top_k or self.top_k
        query_vector = self.embedder.encode([complaint_text]).vectors.astype("float32")
        scores, indices = self.index.search(query_vector, limit)
        results: list[RetrievedExample] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[int(idx)]
            results.append(
                RetrievedExample(
                    complaint_id=str(row.get("complaint_id", "")),
                    complaint_text=str(row.get("complaint_text", "")),
                    department=str(row.get("true_department", "Customer Service")),
                    priority=str(row.get("true_priority", "Medium")),
                    core_issue=str(row.get("true_core_issue", "General complaint requiring manual triage")),
                    similarity=round(float(np.clip(score, 0.0, 1.0)), 3),
                )
            )
        return results
