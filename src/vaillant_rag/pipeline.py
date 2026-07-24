"""High-level RAG pipeline: retrieval + optional re-ranking + generation.

Retrieval is hybrid dense/BM25 search over embedded chunks (with
optional cross-encoder re-ranking). The final answer is generated
strictly from the retrieved document excerpts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .llm import ChatClient, build_rag_prompt
from .loaders import extract_text
from .rerank import Reranker
from .retrieval import RetrievedChunk, Retriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

EMPTY_INDEX_MESSAGE = "The index is empty — run `vaillant-rag index` first."


@dataclass(frozen=True)
class RagAnswer:
    """An answer together with the contexts it was grounded on."""

    answer: str
    contexts: list[RetrievedChunk]


class RagPipeline:
    """End-to-end question answering over an indexed document collection.

    Loads the index and models once; reuse a single instance for multiple
    questions (the FastAPI app and the CLI both do).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
        self.reranker = Reranker(settings.reranker_model) if settings.use_reranker else None
        self.llm = ChatClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self.store: VectorStore
        self.retriever: Retriever | None = None
        self.reload_index()
        logger.info(
            "Pipeline ready: %d chunks, embedding=%s, reranker=%s, llm=%s @ %s",
            len(self.store),
            settings.embedding_model,
            settings.reranker_model if self.reranker else "off",
            settings.llm_model,
            settings.llm_base_url,
        )

    def reload_index(self) -> None:
        """Re-read the index from disk and rebuild the retriever.

        Call after :func:`vaillant_rag.indexing.sync_index` changed the index.
        """
        self.store = VectorStore.load(self.settings.index_dir)
        self.retriever = Retriever(
            self.store,
            self.embedder,
            use_hybrid=self.settings.use_hybrid_search,
            vector_backend=self.settings.vector_backend,
            faiss_min_chunks=self.settings.faiss_min_chunks,
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve the contexts for a question."""
        assert self.retriever is not None
        candidates = self.retriever.search(question, self.settings.top_k)
        if self.reranker:
            contexts = self.reranker.rerank(question, candidates, self.settings.top_n_contexts)
        else:
            contexts = candidates[: self.settings.top_n_contexts]
        if logger.isEnabledFor(logging.DEBUG):  # the text join below is not free
            for rank, chunk in enumerate(contexts, start=1):
                logger.debug(
                    "Context #%d | id=%s score=%.4f | %.120s",
                    rank,
                    chunk.chunk_id,
                    chunk.score,
                    " ".join(chunk.text.split()),
                )
        return contexts

    def ask(self, question: str) -> RagAnswer:
        """Answer a question grounded on the indexed documents."""
        contexts = self.retrieve(question)
        if not contexts:
            return RagAnswer(answer=EMPTY_INDEX_MESSAGE, contexts=[])
        prompt = build_rag_prompt(question, [c.text for c in contexts])
        answer = self.llm.chat(self.settings.system_prompt, prompt)
        return RagAnswer(answer=answer, contexts=contexts)

    def ask_stream(self, question: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Streaming variant of :meth:`ask`.

        Returns:
            The retrieved contexts and an iterator of answer text fragments.
            For an empty index, the iterator yields a single explanatory
            message.
        """
        contexts = self.retrieve(question)
        if not contexts:
            return [], iter([EMPTY_INDEX_MESSAGE])
        prompt = build_rag_prompt(question, [c.text for c in contexts])
        return contexts, self.llm.chat_stream(self.settings.system_prompt, prompt)

    def analyze_document(self, path: str) -> str:
        """Run the system prompt against a single document's full text.

        The text is truncated to ``max_analysis_chars`` to bound token usage.
        """
        text = extract_text(path, ocr_images=self.settings.ocr_images)
        if not text.strip():
            raise ValueError(f"No text could be extracted from {path}")
        truncated = text[: self.settings.max_analysis_chars]
        if len(text) > len(truncated):
            logger.info("Document text truncated from %d to %d chars", len(text), len(truncated))
        return self.llm.chat(self.settings.system_prompt, truncated)
