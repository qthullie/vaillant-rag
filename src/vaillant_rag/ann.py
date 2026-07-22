"""Dense search backends: exact numpy or approximate FAISS (HNSW).

``make_searcher`` picks the backend:

- ``numpy`` — exact brute-force cosine (dot product on normalized vectors).
  Best up to ~10^5 chunks; zero extra dependencies.
- ``faiss`` — approximate HNSW index (inner product). Requires the
  ``[faiss]`` extra. Worth it beyond ~10^5 chunks.
- ``auto`` (default) — FAISS when installed *and* the collection is at
  least ``faiss_min_chunks`` large; numpy otherwise.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("auto", "numpy", "faiss")

# HNSW parameters: good recall/speed defaults for text embeddings.
_HNSW_NEIGHBORS = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 128


class DenseSearcher(Protocol):
    """Search interface shared by all backends."""

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return ``top_k`` (chunk_index, score) pairs, best first."""
        ...


class NumpySearcher:
    """Exact brute-force inner-product search."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if not self.vectors.size:
            return []
        scores = (query_vector @ self.vectors.T).flatten()
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices]


class FaissSearcher:
    """Approximate HNSW search via FAISS (inner product metric)."""

    def __init__(self, vectors: np.ndarray) -> None:
        import faiss  # lazy: optional dependency

        dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, _HNSW_NEIGHBORS, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = _HNSW_EF_SEARCH
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.index = index
        logger.info("FAISS HNSW index built: %d vectors, dim=%d", vectors.shape[0], dim)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i != -1]


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def make_searcher(
    vectors: np.ndarray, backend: str = "auto", faiss_min_chunks: int = 50_000
) -> DenseSearcher:
    """Build the dense search backend for ``vectors``.

    Args:
        vectors: float32 array of shape (n, dim); may be empty.
        backend: ``auto``, ``numpy``, or ``faiss``.
        faiss_min_chunks: In ``auto`` mode, minimum collection size before
            FAISS is preferred over exact numpy search.

    Raises:
        ValueError: For an unknown backend name.
        ImportError: When ``faiss`` is requested explicitly but not installed.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown vector_backend {backend!r}; expected one of {VALID_BACKENDS}")
    if not vectors.size:
        return NumpySearcher(vectors)

    if backend == "faiss":
        if not _faiss_available():
            raise ImportError(
                "vector_backend is set to 'faiss' but faiss is not installed. "
                "Install it with: pip install vaillant-rag[faiss]"
            )
        return FaissSearcher(vectors)

    if backend == "auto" and vectors.shape[0] >= faiss_min_chunks and _faiss_available():
        return FaissSearcher(vectors)

    return NumpySearcher(vectors)
