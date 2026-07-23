"""Regression tests for the pre-publication audit fixes.

M1 — retrieval knobs must be positive (config + searcher guard).
M2 — the log file must rotate (bounded disk usage).
M3 — the API must not leak upstream LLM error bodies to clients.
M5 — index saves are atomic and fail loudly on internal desync.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from vaillant_rag.ann import NumpySearcher
from vaillant_rag.config import Settings
from vaillant_rag.logging_utils import setup_logging
from vaillant_rag.vector_store import VectorStore

# --- M1: retrieval knobs must be positive ------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_k", 0), ("top_k", -3), ("top_n_contexts", 0)],
)
def test_settings_reject_non_positive_retrieval_knobs(field, value):
    with pytest.raises(ValueError, match="positive"):
        replace(Settings(), **{field: value}).validate()


def test_numpy_searcher_top_k_zero_returns_empty():
    vectors = np.eye(4, dtype=np.float32)
    assert NumpySearcher(vectors).search(vectors[:1], top_k=0) == []


# --- M2: log file must rotate ------------------------------------------


def test_logging_uses_rotating_handler(settings):
    setup_logging(settings)
    handlers = logging.getLogger().handlers
    rotating = [h for h in handlers if type(h).__name__ == "RotatingFileHandler"]
    assert rotating, "file handler must rotate to bound disk usage"
    assert rotating[0].maxBytes > 0
    assert rotating[0].backupCount > 0


# --- M3: API must not leak upstream error bodies -----------------------


def test_ask_hides_llm_error_details(settings, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vaillant_rag import api as api_module
    from vaillant_rag.llm import LLMError

    class _FakeStore:
        doc_hashes: dict[str, str] = {}

        def __len__(self) -> int:
            return 0

    class _BoomPipeline:
        store = _FakeStore()

        def ask(self, question):
            raise LLMError("endpoint error — SECRET-UPSTREAM-BODY")

    monkeypatch.setattr(api_module, "RagPipeline", lambda s: _BoomPipeline())
    client = TestClient(api_module.create_app(settings))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "SECRET-UPSTREAM-BODY" not in response.text  # generic detail only


# --- M5: save must be atomic and strict --------------------------------


def test_save_leaves_no_tmp_files_and_roundtrips(tmp_path):
    store = VectorStore()
    store.add_document("a.txt", "ha", ["one", "two"], np.eye(2, dtype=np.float32))
    store.save(tmp_path)
    assert not list(tmp_path.glob("*.tmp*"))  # everything was published
    loaded = VectorStore.load(tmp_path)
    assert loaded.ids == store.ids
    assert loaded.texts == store.texts


def test_save_raises_on_ids_texts_desync(tmp_path):
    store = VectorStore()
    store.add_document("a.txt", "ha", ["one"], np.eye(1, dtype=np.float32))
    store.texts.append("orphan")  # simulate internal corruption
    with pytest.raises(ValueError):  # strict zip must fail loudly
        store.save(tmp_path)
