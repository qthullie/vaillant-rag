"""Embedding model wrapper.

Handles the asymmetric query/passage prefixes required by E5-family models
(``intfloat/*e5*``): documents must be embedded as ``"passage: <text>"`` and
queries as ``"query: <text>"``. Other models are used as-is.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_id: str) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    logger.info("Loading embedding model: %s", model_id)
    return SentenceTransformer(model_id)


class Embedder:
    """Embeds documents and queries with model-appropriate prefixes.

    Args:
        model_id: Hugging Face model id understood by sentence-transformers.
        batch_size: Encoding batch size.
    """

    def __init__(self, model_id: str, batch_size: int = 64) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._is_e5 = "e5" in model_id.lower()

    @property
    def model(self) -> SentenceTransformer:
        return _load_model(self.model_id)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed document chunks. Returns a float32 array of shape (n, dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts, show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query. Returns a float32 array of shape (1, dim)."""
        text = f"query: {query}" if self._is_e5 else query
        return self._encode([text])
