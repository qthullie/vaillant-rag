"""FastAPI web service exposing the RAG pipeline.

Run with ``vaillant-rag serve`` (requires the ``[api]`` extra) or mount
:func:`create_app` under any ASGI server.

Endpoints:
    GET  /health        Liveness + index stats.
    POST /ask           Grounded answer with its source contexts.
    POST /ask/stream    Same, streamed as Server-Sent Events.
    POST /reindex       Sync the index with the docs directory.

Concurrency model: one shared :class:`RagPipeline`; retrieval and
generation run in the server's thread pool (endpoints are sync ``def``).
Re-indexing is serialized by a lock, and the retriever is atomically
swapped when it finishes.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .indexing import sync_index
from .llm import LLMError
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)


class ContextOut(BaseModel):
    chunk_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    contexts: list[ContextOut]


class ReindexRequest(BaseModel):
    full_rebuild: bool = False


class ReindexResponse(BaseModel):
    added_or_updated: list[str]
    skipped_unchanged: list[str]
    removed: list[str]
    failed: list[str]
    total_chunks: int


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application around a shared pipeline instance."""
    settings = settings or load_settings()
    app = FastAPI(
        title="vaillant-rag",
        version=__version__,
        description="Retrieval-Augmented Generation over local documents.",
    )
    pipeline = RagPipeline(settings)
    reindex_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "chunks": len(pipeline.store),
            "documents": len(pipeline.store.doc_hashes),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = pipeline.ask(request.question)
        except LLMError as exc:
            # Full detail stays server-side: upstream error bodies may leak
            # internal URLs or provider internals to API clients.
            logger.error("LLM backend failure: %s", exc)
            raise HTTPException(
                status_code=502, detail="LLM backend error; see server logs."
            ) from exc
        return AskResponse(
            answer=result.answer,
            contexts=[
                ContextOut(chunk_id=c.chunk_id, text=c.text, score=c.score) for c in result.contexts
            ],
        )

    @app.post("/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        """Server-Sent Events stream.

        Emits ``data: {"contexts": [...]}`` first, then one
        ``data: {"delta": "..."}`` event per fragment, and finally
        ``data: [DONE]``. LLM failures mid-stream are reported as a
        ``data: {"error": "..."}`` event.
        """
        try:
            contexts, fragments = pipeline.ask_stream(request.question)
        except LLMError as exc:
            logger.error("LLM backend failure: %s", exc)
            raise HTTPException(
                status_code=502, detail="LLM backend error; see server logs."
            ) from exc

        def event_source() -> Iterator[str]:
            head = {
                "contexts": [
                    {"chunk_id": c.chunk_id, "text": c.text, "score": c.score} for c in contexts
                ]
            }
            yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
            try:
                for fragment in fragments:
                    yield f"data: {json.dumps({'delta': fragment}, ensure_ascii=False)}\n\n"
            except LLMError as exc:
                logger.error("LLM failure mid-stream: %s", exc)
                payload = {"error": "LLM backend error; see server logs."}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/reindex", response_model=ReindexResponse)
    def reindex(request: ReindexRequest) -> ReindexResponse:
        if not reindex_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A re-index is already running")
        try:
            report = sync_index(settings, full_rebuild=request.full_rebuild)
            pipeline.reload_index()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            reindex_lock.release()
        return ReindexResponse(
            added_or_updated=report.added_or_updated,
            skipped_unchanged=report.skipped_unchanged,
            removed=report.removed,
            failed=report.failed,
            total_chunks=len(pipeline.store),
        )

    return app
