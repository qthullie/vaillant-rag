"""Cross-encoder re-ranking.

A cross-encoder scores each (query, chunk) pair jointly, which is far more
accurate than embedding similarity but too slow to run on the whole corpus.
Standard pattern: retrieve ``top_k`` candidates cheaply, re-rank them with
the cross-encoder, keep the best ``top_n``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from .retrieval import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_id: str) -> CrossEncoder:
    """Load (and cache) a cross-encoder model."""
    from sentence_transformers import CrossEncoder  # lazy: heavy import

    logger.info("Loading cross-encoder model: %s", model_id)
    return CrossEncoder(model_id)


class Reranker:
    """Re-orders retrieval candidates with a cross-encoder.

    Args:
        model_id: sentence-transformers cross-encoder model id, e.g.
            ``cross-encoder/ms-marco-MiniLM-L6-v2`` (English) or
            ``BAAI/bge-reranker-base`` (multilingual).
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @property
    def model(self) -> CrossEncoder:
        return _load_cross_encoder(self.model_id)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Score all candidates against ``query`` and keep the best ``top_n``.

        Returned chunks carry the cross-encoder score (higher is better).
        """
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        reranked = [
            RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=float(s))
            for c, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda c: -c.score)
        return reranked[:top_n]
