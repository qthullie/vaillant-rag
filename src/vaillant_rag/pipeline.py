"""High-level RAG pipeline: retrieval + optional re-ranking + generation.

Two retrieval modes (``settings.retrieval_mode``):

- ``vector`` — hybrid dense/BM25 search over embedded chunks (default).
- ``markdown`` — the LLM picks sections from an outline of the Markdown
  corpus (no embeddings; see :mod:`vaillant_rag.markdown_store`).

Either way, the final answer is generated strictly from the retrieved
document excerpts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .llm import ChatClient, LLMError, build_rag_prompt
from .loaders import extract_text
from .markdown_store import (
    SECTION_SELECT_SYSTEM,
    MarkdownStore,
    build_selection_prompt,
    format_outline,
    parse_selection,
)
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
        self.markdown_mode = settings.retrieval_mode == "markdown"
        if self.markdown_mode:
            # No embeddings in markdown mode: skip the heavy models entirely.
            self.embedder = None
            self.reranker = None
            if settings.use_reranker:
                logger.info("use_reranker is ignored in markdown retrieval mode")
        else:
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
        self.store: VectorStore | MarkdownStore
        self.retriever: Retriever | None = None
        self.reload_index()
        logger.info(
            "Pipeline ready (%s mode): %d %s, embedding=%s, reranker=%s, llm=%s @ %s",
            settings.retrieval_mode,
            len(self.store),
            "sections" if self.markdown_mode else "chunks",
            "off" if self.markdown_mode else settings.embedding_model,
            settings.reranker_model if self.reranker else "off",
            settings.llm_model,
            settings.llm_base_url,
        )

    def reload_index(self) -> None:
        """Re-read the index from disk and rebuild the retriever.

        Call after :func:`vaillant_rag.indexing.sync_index` changed the index.
        """
        if self.markdown_mode:
            self.store = MarkdownStore.load(
                self.settings.index_dir, self.settings.markdown_section_max_chars
            )
            self.retriever = None
            return
        assert self.embedder is not None
        self.store = VectorStore.load(self.settings.index_dir)
        self.retriever = Retriever(
            self.store,
            self.embedder,
            use_hybrid=self.settings.use_hybrid_search,
            vector_backend=self.settings.vector_backend,
            faiss_min_chunks=self.settings.faiss_min_chunks,
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve the contexts for a question (mode-dependent)."""
        if self.markdown_mode:
            contexts = self._select_markdown_sections(question)
        else:
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

    def _select_markdown_sections(self, question: str) -> list[RetrievedChunk]:
        """Markdown mode: let the LLM pick sections from the corpus outline.

        The outline is BM25-prefiltered when the corpus exceeds
        ``markdown_outline_limit`` sections. If the selection call fails
        or returns nothing usable, falls back to plain BM25 ranking.
        """
        store = self.store
        assert isinstance(store, MarkdownStore)
        if store.is_empty:
            return []
        limit = self.settings.markdown_outline_limit
        candidates = store.sections if len(store) <= limit else store.bm25_select(question, limit)
        max_picks = self.settings.top_n_contexts
        try:
            reply = self.llm.chat(
                SECTION_SELECT_SYSTEM,
                build_selection_prompt(question, format_outline(candidates), max_picks),
            )
            picks = parse_selection(reply, len(candidates))[:max_picks]
            sections = [candidates[number - 1] for number in picks]
            logger.info("LLM selected sections %s from %d candidates", picks, len(candidates))
        except LLMError as exc:
            logger.warning("Section selection failed (%s); falling back to BM25", exc)
            sections = []
        if not sections:
            sections = store.bm25_select(question, max_picks)
        return [
            RetrievedChunk(
                chunk_id=section.section_id,
                text=f"[{section.doc_name} — {section.heading}]\n{section.text}",
                score=round(1.0 / rank, 4),
            )
            for rank, section in enumerate(sections, start=1)
        ]

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
