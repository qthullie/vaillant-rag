import numpy as np
import pytest

from vaillant_rag.ann import FaissSearcher, NumpySearcher, make_searcher


def _unit_vectors(n: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_numpy_searcher_exact_top1():
    vectors = _unit_vectors(50)
    searcher = NumpySearcher(vectors)
    results = searcher.search(vectors[10:11], top_k=3)
    assert results[0][0] == 10
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_make_searcher_empty_vectors():
    searcher = make_searcher(np.zeros((0, 0), dtype=np.float32), "auto")
    assert searcher.search(np.ones((1, 8), dtype=np.float32), top_k=5) == []


def test_make_searcher_numpy_backend():
    searcher = make_searcher(_unit_vectors(10), "numpy")
    assert isinstance(searcher, NumpySearcher)


def test_make_searcher_unknown_backend():
    with pytest.raises(ValueError, match="Unknown vector_backend"):
        make_searcher(_unit_vectors(10), "milvus")


def test_make_searcher_auto_small_collection_uses_numpy():
    # Below the threshold, auto must pick numpy even if faiss is installed.
    searcher = make_searcher(_unit_vectors(10), "auto", faiss_min_chunks=1000)
    assert isinstance(searcher, NumpySearcher)


def test_faiss_searcher_agrees_with_numpy():
    pytest.importorskip("faiss")
    vectors = _unit_vectors(200)
    query = vectors[42:43]
    exact = NumpySearcher(vectors).search(query, top_k=5)
    approx = FaissSearcher(vectors).search(query, top_k=5)
    # HNSW recall on a 200-vector set is effectively perfect for top-1.
    assert approx[0][0] == exact[0][0] == 42


def test_make_searcher_faiss_missing_raises(monkeypatch):
    import vaillant_rag.ann as ann

    monkeypatch.setattr(ann, "_faiss_available", lambda: False)
    with pytest.raises(ImportError, match=r"vaillant-rag\[faiss\]"):
        make_searcher(_unit_vectors(10), "faiss")
