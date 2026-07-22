"""End-to-end pipeline test: ingest -> index -> retrieve -> generate.

The embedding model is faked (deterministic vectors) and the LLM endpoint
is mocked, so this runs offline. Everything else — loaders, chunking,
vector store persistence, hybrid retrieval, prompt assembly — is real.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from vaillant_rag.indexing import sync_index
from vaillant_rag.pipeline import RagPipeline
from vaillant_rag.vector_store import VectorStore


def _write_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True)
    (docs_dir / "pigeons.txt").write_text(
        "Homing pigeons were used to carry military messages during World War I. "
        "The pigeon Cher Ami saved nearly 200 soldiers in 1918.",
        encoding="utf-8",
    )
    (docs_dir / "python.md").write_text(
        "# Python\n\nPython is a programming language created by Guido van Rossum. "
        "It was first released in 1991.",
        encoding="utf-8",
    )


class _FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "Cher Ami saved soldiers in 1918."}}]}


def test_full_pipeline_end_to_end(settings, fake_embedder):
    _write_docs(Path(settings.docs_dir))

    # --- Ingestion & indexing (with the fake embedder) ---
    with patch("vaillant_rag.indexing.Embedder", return_value=fake_embedder):
        report = sync_index(settings)
    assert sorted(report.added_or_updated) == ["pigeons.txt", "python.md"]
    assert not report.failed

    # Index persisted to disk.
    store = VectorStore.load(settings.index_dir)
    assert len(store) > 0
    assert set(store.doc_hashes) == {"pigeons.txt", "python.md"}

    # --- Incremental sync: nothing changed, nothing re-indexed ---
    with patch("vaillant_rag.indexing.Embedder", return_value=fake_embedder):
        report2 = sync_index(settings)
    assert sorted(report2.skipped_unchanged) == ["pigeons.txt", "python.md"]
    assert not report2.added_or_updated

    # --- Document removal is propagated ---
    (Path(settings.docs_dir) / "python.md").unlink()
    with patch("vaillant_rag.indexing.Embedder", return_value=fake_embedder):
        report3 = sync_index(settings)
    assert report3.removed == ["python.md"]
    store = VectorStore.load(settings.index_dir)
    assert set(store.doc_hashes) == {"pigeons.txt"}

    # --- Question answering with mocked LLM ---
    with (
        patch("vaillant_rag.pipeline.Embedder", return_value=fake_embedder),
        patch("vaillant_rag.llm.requests.post", return_value=_FakeResponse()) as mock_post,
    ):
        pipeline = RagPipeline(settings)
        result = pipeline.ask("Which pigeon saved soldiers?")

    assert result.answer == "Cher Ami saved soldiers in 1918."
    assert result.contexts  # grounded on retrieved chunks
    # The retrieved context actually reached the LLM prompt (BM25 leg
    # guarantees the keyword match lands in the top contexts).
    sent_payload = mock_post.call_args.kwargs["json"]
    user_message = sent_payload["messages"][1]["content"]
    assert "Cher Ami" in user_message
    assert "Which pigeon saved soldiers?" in user_message
