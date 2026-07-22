"""Retrieval: dense, sparse (BM25), and hybrid search.

Hybrid search fuses dense cosine-similarity ranking with BM25 keyword
ranking using Reciprocal Rank Fusion (RRF), which is robust because it
only depends on ranks, not on incompatible score scales.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .ann import make_searcher
from .embeddings import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Standard RRF dampening constant (Cormack et al., 2009).
RRF_K = 60

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval result."""

    chunk_id: str
    text: str
    score: float


class Retriever:
    """Searches a :class:`VectorStore` with dense or hybrid ranking.

    Args:
        store: The loaded vector store.
        embedder: Embedder matching the model used to build the index.
        use_hybrid: When True, fuse dense results with BM25 via RRF.
        vector_backend: Dense backend: ``auto``, ``numpy``, or ``faiss``.
        faiss_min_chunks: ``auto`` mode threshold for switching to FAISS.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        use_hybrid: bool = True,
        vector_backend: str = "auto",
        faiss_min_chunks: int = 50_000,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.use_hybrid = use_hybrid
        self._searcher = make_searcher(store.vectors, vector_backend, faiss_min_chunks)
        self._bm25: BM25Okapi | None = None
        if use_hybrid and not store.is_empty:
            self._bm25 = BM25Okapi([_tokenize(t) for t in store.texts])

    def _bm25_ranking(self, query: str, top_k: int) -> list[int]:
        """Return chunk indices ranked by BM25 score (best first)."""
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return ranked[:top_k]

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve the ``top_k`` most relevant chunks for ``query``.

        Returns an empty list when the index is empty.
        """
        if self.store.is_empty:
            logger.warning("Search on empty index — did you run `vaillant-rag index`?")
            return []

        query_vector = self.embedder.embed_query(query)
        dense = self._searcher.search(query_vector, top_k)

        if not (self.use_hybrid and self._bm25):
            return [
                RetrievedChunk(self.store.ids[i], self.store.texts[i], score) for i, score in dense
            ]

        # Reciprocal Rank Fusion of dense and BM25 rankings.
        fused: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(self._bm25_ranking(query, top_k)):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievedChunk(self.store.ids[i], self.store.texts[i], score) for i, score in ranked
        ]
