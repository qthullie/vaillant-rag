"""Index construction and incremental synchronization.

Two index kinds, selected by ``retrieval_mode``:

- ``vector`` — documents are chunked and embedded into the vector store.
- ``markdown`` — documents are rendered to a Markdown corpus (no
  embeddings; see :mod:`vaillant_rag.markdown_store`).

Incremental strategy (both kinds): each document's raw bytes are hashed
(SHA-256). On sync, unchanged documents are skipped, changed documents
are re-processed, and documents deleted from the source directory are
removed from the index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .loaders import extract_markdown, extract_text, is_supported
from .markdown_store import HASHES_FILE, markdown_dir
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Summary of an index synchronization run."""

    added_or_updated: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _hash_file(path: Path) -> str:
    """SHA-256 of the file content."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_documents(docs_dir: Path) -> list[Path]:
    """Supported documents directly inside ``docs_dir``, sorted by name."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}. "
            "Create it and add documents, or set DOCS_DIR."
        )
    return sorted(p for p in docs_dir.iterdir() if p.is_file() and is_supported(p))


def index_document(store: VectorStore, embedder: Embedder, settings: Settings, path: Path) -> int:
    """Extract, chunk, embed, and upsert one document into ``store``.

    Returns:
        Number of chunks produced (0 when the document yields no text).
    """
    started = time.perf_counter()
    text = extract_text(path, ocr_images=settings.ocr_images)
    chunks = chunk_text(text, settings.chunk_size_chars, settings.chunk_overlap_chars)
    vectors = embedder.embed_documents(chunks)
    store.add_document(path.name, _hash_file(path), chunks, vectors)
    logger.info(
        "Indexed %s: %d chars, %d chunks in %.2fs",
        path.name,
        len(text),
        len(chunks),
        time.perf_counter() - started,
    )
    if not chunks:
        logger.warning(
            "No text extracted from %s (image-only document? try ocr_images=true)", path.name
        )
    return len(chunks)


def sync_index(settings: Settings, full_rebuild: bool = False) -> SyncReport:
    """Synchronize the index with the documents directory.

    Dispatches on ``settings.retrieval_mode``: the vector store or the
    Markdown corpus.

    Args:
        settings: Application settings.
        full_rebuild: When True, ignore stored hashes and re-index everything.

    Returns:
        A :class:`SyncReport` describing what changed.
    """
    if settings.retrieval_mode == "markdown":
        return _sync_markdown_corpus(settings, full_rebuild)
    return _sync_vector_store(settings, full_rebuild)


def _sync_vector_store(settings: Settings, full_rebuild: bool) -> SyncReport:
    """Chunk + embed changed documents into the vector store."""
    docs_dir = Path(settings.docs_dir)
    documents = _list_documents(docs_dir)
    store = VectorStore() if full_rebuild else VectorStore.load(settings.index_dir)
    embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
    report = SyncReport()

    present_names = {p.name for p in documents}
    for stale_name in sorted(set(store.doc_hashes) - present_names):
        removed_chunks = store.remove_document(stale_name)
        logger.info("Removed %s from index (%d chunks)", stale_name, removed_chunks)
        report.removed.append(stale_name)

    for path in documents:
        try:
            if not full_rebuild and store.doc_hashes.get(path.name) == _hash_file(path):
                logger.debug("Unchanged, skipping: %s", path.name)
                report.skipped_unchanged.append(path.name)
                continue
            index_document(store, embedder, settings, path)
            report.added_or_updated.append(path.name)
        except Exception:
            logger.exception("Failed to index %s", path.name)
            report.failed.append(path.name)

    store.save(settings.index_dir)
    _log_sync_summary(report)
    return report


def _sync_markdown_corpus(settings: Settings, full_rebuild: bool) -> SyncReport:
    """Render changed documents to the Markdown corpus (no embeddings)."""
    docs_dir = Path(settings.docs_dir)
    documents = _list_documents(docs_dir)
    directory = markdown_dir(settings.index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    hashes_path = directory / HASHES_FILE

    hashes: dict[str, str] = {}
    if full_rebuild:
        for md_path in directory.glob("*.md"):
            md_path.unlink()
    elif hashes_path.is_file():
        try:
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring corrupt %s: %s", hashes_path, exc)

    report = SyncReport()
    present_names = {p.name for p in documents}
    for stale_name in sorted(set(hashes) - present_names):
        (directory / f"{stale_name}.md").unlink(missing_ok=True)
        hashes.pop(stale_name)
        logger.info("Removed %s from markdown corpus", stale_name)
        report.removed.append(stale_name)

    for path in documents:
        try:
            digest = _hash_file(path)
            if hashes.get(path.name) == digest:
                logger.debug("Unchanged, skipping: %s", path.name)
                report.skipped_unchanged.append(path.name)
                continue
            started = time.perf_counter()
            markdown = extract_markdown(path, ocr_images=settings.ocr_images)
            (directory / f"{path.name}.md").write_text(markdown, encoding="utf-8")
            hashes[path.name] = digest
            logger.info(
                "Rendered %s to markdown: %d chars in %.2fs",
                path.name,
                len(markdown),
                time.perf_counter() - started,
            )
            report.added_or_updated.append(path.name)
        except Exception:
            logger.exception("Failed to render %s", path.name)
            report.failed.append(path.name)

    hashes_path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_sync_summary(report)
    return report


def _log_sync_summary(report: SyncReport) -> None:
    logger.info(
        "Sync done: %d added/updated, %d unchanged, %d removed, %d failed",
        len(report.added_or_updated),
        len(report.skipped_unchanged),
        len(report.removed),
        len(report.failed),
    )
