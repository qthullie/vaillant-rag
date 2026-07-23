import numpy as np
import pytest

from vaillant_rag.ann import NumpySearcher
from vaillant_rag.vector_store import VectorStore


def _unit_vectors(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_empty_store():
    store = VectorStore()
    assert store.is_empty
    assert len(store) == 0
    assert NumpySearcher(store.vectors).search(np.ones((1, 8), dtype=np.float32), top_k=5) == []


def test_add_and_search():
    store = VectorStore()
    vectors = _unit_vectors(3)
    store.add_document("a.txt", "hash-a", ["one", "two", "three"], vectors)
    assert len(store) == 3
    results = NumpySearcher(store.vectors).search(vectors[1:2], top_k=2)
    assert results[0][0] == 1  # most similar to itself
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_add_replaces_existing_document():
    store = VectorStore()
    store.add_document("a.txt", "h1", ["old1", "old2"], _unit_vectors(2))
    store.add_document("a.txt", "h2", ["new1"], _unit_vectors(1))
    assert store.texts == ["new1"]
    assert store.doc_hashes["a.txt"] == "h2"


def test_remove_document():
    store = VectorStore()
    store.add_document("a.txt", "ha", ["a1", "a2"], _unit_vectors(2))
    store.add_document("b.txt", "hb", ["b1"], _unit_vectors(1))
    removed = store.remove_document("a.txt")
    assert removed == 2
    assert store.texts == ["b1"]
    assert "a.txt" not in store.doc_hashes
    assert store.remove_document("missing.txt") == 0


def test_dimension_mismatch_raises():
    store = VectorStore()
    store.add_document("a.txt", "ha", ["a1"], _unit_vectors(1, dim=8))
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add_document("b.txt", "hb", ["b1"], _unit_vectors(1, dim=16))


def test_chunk_vector_count_mismatch_raises():
    store = VectorStore()
    with pytest.raises(ValueError, match="chunks"):
        store.add_document("a.txt", "ha", ["one", "two"], _unit_vectors(3))


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add_document("a.txt", "ha", ["hello", "world"], _unit_vectors(2))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.ids == store.ids
    assert loaded.texts == store.texts
    assert loaded.doc_hashes == store.doc_hashes
    np.testing.assert_allclose(loaded.vectors, store.vectors)


def test_load_missing_dir_returns_empty(tmp_path):
    store = VectorStore.load(tmp_path / "nope")
    assert store.is_empty
