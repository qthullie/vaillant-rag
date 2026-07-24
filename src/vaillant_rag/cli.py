"""Command-line interface.

Commands:
    vaillant-rag index            Build or rebuild the full index.
    vaillant-rag update           Incrementally sync the index with the docs directory.
    vaillant-rag qa               Interactive question-answering session (streams answers).
    vaillant-rag analyze <file>   Run the system prompt against one document.
    vaillant-rag eval             Measure retrieval quality (recall@k, MRR, latency).
    vaillant-rag serve            Start the FastAPI web service (requires the [api] extra).

Output follows the demo palette (see ``assets/demo.svg``); while an
answer is being retrieved and generated, the logo pigeon flies across
the line. Both degrade gracefully on non-TTY output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from .config import Settings, load_settings
from .evaluation import DEFAULT_KS, EvalSummary, RetrieveFn, evaluate, load_questions
from .indexing import sync_index
from .llm import LLMError
from .logging_utils import setup_logging
from .pipeline import RagPipeline
from .term import PigeonFlight, accent, dim, error, ok, prompt_input

logger = logging.getLogger(__name__)

QUIT_COMMANDS = {"quit", "exit", "q"}

DEFAULT_QUESTIONS_PATH = "eval/questions.yaml"


def _retrieval_summary(settings: Settings) -> str:
    """Human-readable retrieval description for the timing line."""
    summary = "hybrid search: dense + BM25, RRF" if settings.use_hybrid_search else "dense search"
    if settings.use_reranker:
        summary += ", reranked"
    return summary


def _cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    report = sync_index(settings, full_rebuild=not args.incremental)
    summary = (
        f"Index updated: {len(report.added_or_updated)} added/updated, "
        f"{len(report.skipped_unchanged)} unchanged, "
        f"{len(report.removed)} removed, {len(report.failed)} failed."
    )
    print(error(summary) if report.failed else summary)
    return 1 if report.failed else 0


def _stream_answer(pipeline: RagPipeline, question: str) -> bool:
    """Stream one answer with the pigeon flying until the first token.

    Returns False when the LLM failed (already reported on stderr).
    """
    flight = PigeonFlight()
    flight.start()
    try:
        _contexts, fragments = pipeline.ask_stream(question)
        first = next(fragments, "")
    except LLMError as exc:
        flight.stop()
        print(error(f"LLM error: {exc}"), file=sys.stderr)
        return False
    finally:
        flight.stop()
    print(ok("\n=== Answer ===") + "\n")
    try:
        if first:
            print(first, end="", flush=True)
        for fragment in fragments:
            print(fragment, end="", flush=True)
    except LLMError as exc:
        print("\n" + error(f"LLM error: {exc}"), file=sys.stderr)
        return False
    return True


def _cmd_qa(settings: Settings, args: argparse.Namespace) -> int:
    pipeline = RagPipeline(settings)
    if pipeline.store.is_empty:
        print(error("The index is empty. Run `vaillant-rag index` first."), file=sys.stderr)
        return 1
    print(f"{accent('vaillant-rag')} {dim('— qa')}")
    print(dim(f"Loaded {len(pipeline.store)} chunks. Type 'quit' to exit.") + "\n")
    while True:
        try:
            question = prompt_input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not question or question.lower() in QUIT_COMMANDS:
            print("Bye.")
            return 0
        started = time.perf_counter()
        if not _stream_answer(pipeline, question):
            continue
        elapsed = time.perf_counter() - started
        print("\n\n" + dim(f"({elapsed:.1f}s — streamed, {_retrieval_summary(settings)})") + "\n")


def _cmd_analyze(settings: Settings, args: argparse.Namespace) -> int:
    pipeline = RagPipeline(settings)
    flight = PigeonFlight()
    flight.start()
    try:
        answer = pipeline.analyze_document(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(error(f"Error: {exc}"), file=sys.stderr)
        return 1
    except LLMError as exc:
        print(error(f"LLM error: {exc}"), file=sys.stderr)
        return 1
    finally:
        flight.stop()
    print(ok("\n=== Document analysis ===") + "\n")
    print(answer)
    return 0


def _build_retrieve_fn(
    store, embedder, settings: Settings, *, use_hybrid: bool, reranker=None
) -> RetrieveFn:
    """Compose a ranked-retrieval function for one evaluation configuration.

    The reranker (when given) re-scores the full candidate set so that
    recall@k can be measured over the reranked order, not just the top-n.
    """
    from .retrieval import Retriever

    retriever = Retriever(
        store,
        embedder,
        use_hybrid=use_hybrid,
        vector_backend=settings.vector_backend,
        faiss_min_chunks=settings.faiss_min_chunks,
    )
    top_k = settings.top_k
    if reranker is None:
        return lambda q: retriever.search(q, top_k)
    return lambda q: reranker.rerank(q, retriever.search(q, top_k), top_k)


def _run_config(name: str, retrieve_fn: RetrieveFn, questions, ks) -> EvalSummary:
    """Warm up (excludes model load from timing) then evaluate one config."""
    if questions:
        retrieve_fn(questions[0].question)
    return evaluate(name, retrieve_fn, questions, ks)


def _print_eval_table(summaries: list[EvalSummary], ks) -> None:
    """Print a side-by-side table: one column per configuration."""
    rows = [f"recall@{k}" for k in ks] + ["MRR", "latency p50 (ms)", "latency p95 (ms)"]
    label_w = max(len(r) for r in rows) + 2
    col_w = max(14, max(len(s.config) for s in summaries) + 2)

    header = " " * label_w + "".join(accent(s.config.ljust(col_w)) for s in summaries)
    print("\n" + header)
    print(dim("-" * (label_w + col_w * len(summaries))))

    def line(label: str, cells: list[str]) -> str:
        return dim(label.ljust(label_w)) + "".join(c.ljust(col_w) for c in cells)

    for k in ks:
        print(line(f"recall@{k}", [f"{s.recall_at_k[k]:.3f}" for s in summaries]))
    print(line("MRR", [f"{s.mrr:.3f}" for s in summaries]))
    print(line("latency p50 (ms)", [f"{s.latency_p50_ms:.1f}" for s in summaries]))
    print(line("latency p95 (ms)", [f"{s.latency_p95_ms:.1f}" for s in summaries]))
    first = summaries[0]
    print(
        dim(
            f"\n{first.num_answerable} answerable + {first.num_unanswerable} "
            "unanswerable questions (unanswerable excluded from quality metrics)."
        )
    )


def _cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from .embeddings import Embedder
    from .vector_store import VectorStore

    ks = tuple(int(k) for k in args.k.split(",")) if args.k else DEFAULT_KS
    try:
        questions = load_questions(args.questions)
    except (FileNotFoundError, ValueError) as exc:
        print(error(f"Error loading questions: {exc}"), file=sys.stderr)
        return 1

    embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
    store = VectorStore.load(settings.index_dir)
    if store.is_empty:
        print(error("The index is empty. Run `vaillant-rag index` first."), file=sys.stderr)
        return 1

    print(f"{accent('vaillant-rag')} {dim('— eval')}")
    print(dim(f"{len(store)} chunks, embedding={settings.embedding_model}, top_k={settings.top_k}"))

    summaries: list[EvalSummary] = []
    if args.ablate:
        summaries.append(
            _run_config(
                "dense",
                _build_retrieve_fn(store, embedder, settings, use_hybrid=False),
                questions,
                ks,
            )
        )
        summaries.append(
            _run_config(
                "hybrid",
                _build_retrieve_fn(store, embedder, settings, use_hybrid=True),
                questions,
                ks,
            )
        )
        from .rerank import Reranker

        summaries.append(
            _run_config(
                "hybrid+rerank",
                _build_retrieve_fn(
                    store,
                    embedder,
                    settings,
                    use_hybrid=True,
                    reranker=Reranker(settings.reranker_model),
                ),
                questions,
                ks,
            )
        )
    else:
        reranker = None
        if settings.use_reranker:
            from .rerank import Reranker

            reranker = Reranker(settings.reranker_model)
        name = "hybrid" if settings.use_hybrid_search else "dense"
        if reranker is not None:
            name += "+rerank"
        summaries.append(
            _run_config(
                name,
                _build_retrieve_fn(
                    store,
                    embedder,
                    settings,
                    use_hybrid=settings.use_hybrid_search,
                    reranker=reranker,
                ),
                questions,
                ks,
            )
        )

    if args.chunk_sizes:
        summaries.extend(_ablate_chunk_sizes(settings, embedder, questions, ks, args.chunk_sizes))

    _print_eval_table(summaries, ks)

    if args.json:
        payload = {
            "corpus_chunks": len(store),
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "ks": list(ks),
            "configs": [s.to_dict() for s in summaries],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(ok(f"\nWrote {args.json}"))
    return 0


def _ablate_chunk_sizes(
    settings: Settings, embedder, questions, ks, chunk_sizes: list[int]
) -> list[EvalSummary]:
    """Re-index the corpus at each chunk size (temp index) and evaluate hybrid.

    Each size gets a throwaway index directory so the current index on disk
    is left untouched. Embeddings are recomputed, so this is the slow path.
    """
    from .vector_store import VectorStore

    summaries: list[EvalSummary] = []
    for size in chunk_sizes:
        with tempfile.TemporaryDirectory(prefix="vaillant-eval-") as tmp:
            variant = dataclasses.replace(settings, chunk_size_chars=size, index_dir=tmp)
            sync_index(variant, full_rebuild=True)
            store = VectorStore.load(tmp)
            fn = _build_retrieve_fn(store, embedder, variant, use_hybrid=True)
            summaries.append(_run_config(f"hybrid chunk={size}", fn, questions, ks))
    return summaries


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .api import create_app
    except ImportError:
        print(
            "The web service requires the API extra: pip install vaillant-rag[api]",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="vaillant-rag",
        description="Retrieval-Augmented Generation over local documents.",
    )
    parser.add_argument(
        "--config", default=None, help="Path to a YAML config file (default: config.yaml)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Build the index from the docs directory")
    p_index.add_argument(
        "--incremental",
        action="store_true",
        help="Only re-index changed documents (same as `update`)",
    )

    subparsers.add_parser("update", help="Incrementally sync the index with the docs directory")
    subparsers.add_parser("qa", help="Start an interactive question-answering session")

    p_analyze = subparsers.add_parser(
        "analyze", help="Run the system prompt against a single document"
    )
    p_analyze.add_argument("file", help="Path to the document to analyze")

    p_eval = subparsers.add_parser(
        "eval", help="Measure retrieval quality (recall@k, MRR, latency)"
    )
    p_eval.add_argument(
        "--ablate",
        action="store_true",
        help="Compare dense vs hybrid vs hybrid+reranker on the same corpus",
    )
    p_eval.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_PATH,
        help=f"Question set (default: {DEFAULT_QUESTIONS_PATH})",
    )
    p_eval.add_argument("--json", default=None, help="Also write results to this JSON file")
    p_eval.add_argument(
        "--k", default=None, help="Comma-separated recall cut-offs (default: 1,3,5,10)"
    )
    p_eval.add_argument(
        "--chunk-sizes",
        default=None,
        type=lambda s: [int(x) for x in s.split(",")],
        help="Also ablate these chunk sizes (rebuilds a temp index per size, slow)",
    )

    p_serve = subparsers.add_parser("serve", help="Start the FastAPI web service")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(settings)

    try:
        if args.command == "index":
            return _cmd_index(settings, args)
        if args.command == "update":
            args.incremental = True
            return _cmd_index(settings, args)
        if args.command == "qa":
            return _cmd_qa(settings, args)
        if args.command == "analyze":
            return _cmd_analyze(settings, args)
        if args.command == "eval":
            return _cmd_eval(settings, args)
        if args.command == "serve":
            return _cmd_serve(settings, args)
    except FileNotFoundError as exc:
        print(error(f"Error: {exc}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
