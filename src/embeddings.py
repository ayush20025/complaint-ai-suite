"""Embedding utilities for sentence-transformer and FAISS workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from config import EMBEDDING_MODEL_NAME, USE_SENTENCE_TRANSFORMER
from src.utils import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: str
    backend: str


class ComplaintEmbedder:
    """CPU-friendly sentence embedder with offline-safe fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._backend = "hashing"
        self._model = None
        self._fallback = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)
        if USE_SENTENCE_TRANSFORMER and SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
                self._backend = "sentence-transformers"
            except Exception as exc:
                logger.warning("SentenceTransformer model unavailable, falling back to hashing embeddings: %s", exc)

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, texts: Iterable[str]) -> EmbeddingResult:
        payload = [str(text or "") for text in texts]
        if self._model is not None:
            vectors = self._model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
            return EmbeddingResult(vectors=np.asarray(vectors, dtype="float32"), model_name=self.model_name, backend=self._backend)
        sparse = self._fallback.transform(payload)
        dense = normalize(sparse, norm="l2", copy=False).toarray().astype("float32")
        return EmbeddingResult(vectors=dense, model_name="hashing-fallback", backend=self._backend)
