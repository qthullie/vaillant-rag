"""Command-line interface.

Commands:
    vaillant-rag index            Build or rebuild the full index.
    vaillant-rag update           Incrementally sync the index with the docs directory.
    vaillant-rag qa               Interactive question-answering session (streams answers).
    vaillant-rag analyze <file>   Run the system prompt against one document.
    vaillant-rag serve            Start the FastAPI web service (requires the [api] extra).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import Settings, load_settings
from .indexing import sync_index
from .llm import LLMError
from .logging_utils import setup_logging
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

QUIT_COMMANDS = {"quit", "exit", "q"}


def _cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    report = sync_index(settings, full_rebuild=not args.incremental)
    print(
        f"Index updated: {len(report.added_or_updated)} added/updated, "
        f"{len(report.skipped_unchanged)} unchanged, "
        f"{len(report.removed)} removed, {len(report.failed)} failed."
    )
    return 1 if report.failed else 0


def _cmd_qa(settings: Settings, args: argparse.Namespace) -> int:
    pipeline = RagPipeline(settings)
    if pipeline.store.is_empty:
        print("The index is empty. Run `vaillant-rag index` first.", file=sys.stderr)
        return 1
    print(f"Loaded {len(pipeline.store)} chunks. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not question or question.lower() in QUIT_COMMANDS:
            print("Bye.")
            return 0
        started = time.perf_counter()
        try:
            _contexts, fragments = pipeline.ask_stream(question)
            print("\n=== Answer ===\n")
            for fragment in fragments:
                print(fragment, end="", flush=True)
        except LLMError as exc:
            print(f"\nLLM error: {exc}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - started
        print(f"\n\n({elapsed:.1f}s)\n")


def _cmd_analyze(settings: Settings, args: argparse.Namespace) -> int:
    pipeline = RagPipeline(settings)
    try:
        answer = pipeline.analyze_document(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    print("\n=== Document analysis ===\n")
    print(answer)
    return 0


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
        if args.command == "serve":
            return _cmd_serve(settings, args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
