"""Index construction and incremental synchronization.

Incremental strategy: each document's raw bytes are hashed (SHA-256).
On sync, unchanged documents are skipped, changed documents are
re-chunked and re-embedded, and documents deleted from the source
directory are removed from the index.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .loaders import extract_text, is_supported
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

    Args:
        settings: Application settings.
        full_rebuild: When True, ignore stored hashes and re-index everything.

    Returns:
        A :class:`SyncReport` describing what changed.
    """
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
    logger.info(
        "Sync done: %d added/updated, %d unchanged, %d removed, %d failed",
        len(report.added_or_updated),
        len(report.skipped_unchanged),
        len(report.removed),
        len(report.failed),
    )
    return report
