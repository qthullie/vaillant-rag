# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-24

First public release.

### Added

- Hybrid retrieval (dense embeddings + BM25, fused with Reciprocal Rank
  Fusion) over a local numpy vector store, with an optional FAISS HNSW
  backend for large collections.
- Optional cross-encoder re-ranking of retrieval candidates.
- Multi-format ingestion: PDF (PyMuPDF, optional OCR), plain text,
  Markdown, reStructuredText, HTML, Word (`.docx`), and Excel
  (`.xlsx`/`.xlsm`).
- Provider-agnostic LLM client for any OpenAI-compatible endpoint
  (Ollama, LM Studio, vLLM, OpenAI, ...), with streaming answers.
- Interactive Q&A CLI, single-document analysis, and a FastAPI web
  service (`/ask`, `/ask/stream`, `/reindex`, `/health`).
- Incremental indexing based on per-document SHA-256 hashing.
- Redistributable example corpus in `data/docs/` (Universal Declaration
  of Human Rights, a chapter of *The Rust Programming Language*, and a
  solar-system fact sheet) with documented provenance and licensing.
- `vaillant-rag eval`: offline retrieval evaluation reporting recall@k,
  MRR, and per-query latency, with an `--ablate` mode comparing dense vs
  hybrid vs hybrid + re-ranker and an optional `--chunk-sizes` ablation.
  Results for the bundled corpus are reported in the README.

### Removed

- **Breaking:** the Markdown retrieval mode (LLM-navigated section
  selection over a rendered Markdown corpus) has been removed. Hybrid
  vector retrieval is now the single retrieval strategy. The
  `retrieval_mode`, `markdown_section_max_chars`, and
  `markdown_outline_limit` configuration keys no longer exist; remove
  them from any `config.yaml` or environment configuration. Markdown
  (`.md`, `.markdown`) remains a fully supported *input* format.

[0.1.0]: https://github.com/qthullie/vaillant-rag/releases/tag/v0.1.0
