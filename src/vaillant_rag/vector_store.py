"""On-disk vector store.

Storage layout inside the index directory:

- ``index.npz``   — dense vectors (float32, L2-normalized) and chunk ids.
- ``store.jsonl`` — one JSON record per chunk: ``{"id": ..., "text": ...}``.
- ``doc_hashes.json`` — SHA-256 content hash per indexed document, used for
  incremental updates.

Search is brute-force cosine similarity (dot product on normalized vectors),
which is exact and fast enough up to roughly 10^5–10^6 chunks. Beyond that,
swap in an ANN index (FAISS, hnswlib) behind this same interface.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = "index.npz"
STORE_FILE = "store.jsonl"
HASHES_FILE = "doc_hashes.json"


@dataclass
class VectorStore:
    """In-memory chunk store with persistence helpers.

    Attributes:
        vectors: float32 array of shape (n, dim); (0, 0) when empty.
        ids: Chunk ids, format ``"<doc_name>#chunk-<i>"``.
        texts: Chunk texts, parallel to ``ids``.
        doc_hashes: SHA-256 content hash per document name.
    """

    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    doc_hashes: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def is_empty(self) -> bool:
        return not self.ids

    # --- persistence -----------------------------------------------------

    @classmethod
    def load(cls, index_dir: str | Path) -> VectorStore:
        """Load a store from disk; returns an empty store if files are absent."""
        index_dir = Path(index_dir)
        index_path = index_dir / INDEX_FILE
        store_path = index_dir / STORE_FILE
        hashes_path = index_dir / HASHES_FILE
        if not (index_path.is_file() and store_path.is_file()):
            return cls()

        arrays = np.load(index_path)
        vectors = arrays["vectors"].astype(np.float32)
        ids = [str(x) for x in arrays["ids"]]
        texts: list[str] = []
        with open(store_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Corrupt store file {store_path} at line {line_number}: {exc}"
                    ) from exc
        if len(texts) != len(ids):
            raise ValueError(
                f"Index inconsistency: {len(ids)} ids in {index_path} "
                f"but {len(texts)} texts in {store_path}"
            )

        doc_hashes: dict[str, str] = {}
        if hashes_path.is_file():
            try:
                doc_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Ignoring corrupt %s: %s", hashes_path, exc)

        return cls(vectors=vectors, ids=ids, texts=texts, doc_hashes=doc_hashes)

    def save(self, index_dir: str | Path) -> None:
        """Persist the store to ``index_dir`` (created if missing)."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            index_dir / INDEX_FILE,
            vectors=np.asarray(self.vectors, dtype=np.float32),
            ids=np.array(self.ids),
        )
        with open(index_dir / STORE_FILE, "w", encoding="utf-8") as f:
            for chunk_id, text in zip(self.ids, self.texts, strict=False):
                f.write(json.dumps({"id": chunk_id, "text": text}, ensure_ascii=False) + "\n")
        (index_dir / HASHES_FILE).write_text(
            json.dumps(self.doc_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Index saved: %d chunks, %d documents", len(self.ids), len(self.doc_hashes))

    # --- mutation --------------------------------------------------------

    def remove_document(self, doc_name: str) -> int:
        """Remove all chunks of a document. Returns the number removed."""
        prefix = f"{doc_name}#"
        kept = [i for i, cid in enumerate(self.ids) if not cid.startswith(prefix)]
        removed = len(self.ids) - len(kept)
        if removed:
            self.vectors = self.vectors[kept] if kept else np.zeros((0, 0), dtype=np.float32)
            self.ids = [self.ids[i] for i in kept]
            self.texts = [self.texts[i] for i in kept]
        self.doc_hashes.pop(doc_name, None)
        return removed

    def add_document(
        self, doc_name: str, doc_hash: str, chunks: list[str], vectors: np.ndarray
    ) -> None:
        """Add (or replace) a document's chunks.

        Args:
            doc_name: Document file name (used as id prefix).
            doc_hash: SHA-256 hash of the document content.
            chunks: Chunk texts.
            vectors: Embeddings for ``chunks``, shape (len(chunks), dim).

        Raises:
            ValueError: On chunk/vector count mismatch or embedding
                dimension mismatch with the existing index.
        """
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(chunks)} chunks but {vectors.shape[0]} vectors for {doc_name}"
            )
        self.remove_document(doc_name)
        if chunks:
            if self.vectors.size and self.vectors.shape[1] != vectors.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: index has {self.vectors.shape[1]}, "
                    f"new vectors have {vectors.shape[1]}. "
                    "Rebuild the index after changing the embedding model."
                )
            self.vectors = (
                np.concatenate([self.vectors, vectors], axis=0) if self.vectors.size else vectors
            )
            self.ids.extend(f"{doc_name}#chunk-{i}" for i in range(len(chunks)))
            self.texts.extend(chunks)
        self.doc_hashes[doc_name] = doc_hash

    # --- search ----------------------------------------------------------

    def dense_search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return the ``top_k`` (chunk_index, cosine_score) pairs for a query.

        Exact numpy search; see :mod:`vaillant_rag.ann` for the FAISS backend.
        Returns an empty list when the store is empty.
        """
        from .ann import NumpySearcher  # local import to avoid a module cycle

        return NumpySearcher(self.vectors).search(query_vector, top_k)
