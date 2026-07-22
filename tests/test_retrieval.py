from vaillant_rag.retrieval import Retriever
from vaillant_rag.vector_store import VectorStore

DOCS = [
    "The Eiffel Tower is located in Paris, France.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Python programming language emphasizes readability.",
    "Paris is the capital of France and hosts many museums.",
]


def _build_store(fake_embedder) -> VectorStore:
    store = VectorStore()
    vectors = fake_embedder.embed_documents(DOCS)
    for i, (text, vector) in enumerate(zip(DOCS, vectors, strict=False)):
        store.add_document(f"doc{i}.txt", f"hash{i}", [text], vector.reshape(1, -1))
    return store


def test_empty_index_returns_no_results(fake_embedder):
    retriever = Retriever(VectorStore(), fake_embedder, use_hybrid=True)
    assert retriever.search("anything", top_k=3) == []


def test_hybrid_search_finds_keyword_match(fake_embedder):
    """BM25 leg must surface exact keyword matches even with random dense vectors."""
    store = _build_store(fake_embedder)
    retriever = Retriever(store, fake_embedder, use_hybrid=True)
    results = retriever.search("Python programming readability", top_k=4)
    assert results
    assert any("Python" in r.text for r in results[:2])


def test_dense_only_search_runs(fake_embedder):
    store = _build_store(fake_embedder)
    retriever = Retriever(store, fake_embedder, use_hybrid=False)
    results = retriever.search("museums in Paris", top_k=2)
    assert len(results) == 2
    assert all(r.chunk_id and r.text for r in results)


def test_top_k_respected(fake_embedder):
    store = _build_store(fake_embedder)
    retriever = Retriever(store, fake_embedder, use_hybrid=True)
    assert len(retriever.search("France", top_k=2)) <= 2
