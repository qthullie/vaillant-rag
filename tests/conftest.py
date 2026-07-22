"""Shared test fixtures.

Tests never download real embedding models: ``fake_embedder`` produces
deterministic hash-based vectors so the suite runs offline and fast.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vaillant_rag.config import Settings
from vaillant_rag.embeddings import Embedder

EMBEDDING_DIM = 32


def _fake_vector(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding derived from the text's hash."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder(Embedder):
    """Embedder replacement that needs no model download."""

    def __init__(self) -> None:
        super().__init__(model_id="fake-model", batch_size=8)

    def embed_documents(self, texts, show_progress=False):  # noqa: ARG002
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([_fake_vector(t) for t in texts])

    def embed_query(self, query):
        return _fake_vector(query).reshape(1, -1)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing every path at a temp directory."""
    return Settings(
        docs_dir=str(tmp_path / "docs"),
        index_dir=str(tmp_path / "index"),
        log_dir=str(tmp_path / "logs"),
        system_prompt_path="You are a test assistant.",
        chunk_size_chars=120,
        chunk_overlap_chars=20,
        top_k=5,
        top_n_contexts=2,
    )
