<p align="center">
  <img src="assets/logo.svg" alt="vaillant-rag logo — pixel-art homing pigeon" width="160"/>
</p>

<h1 align="center">Vaillant RAG</h1>

<p align="center">
  <a href="https://github.com/qthullie/vaillant-rag/actions/workflows/ci.yml"><img src="https://github.com/qthullie/vaillant-rag/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"/>
</p>

> Named after **Vaillant**, the homing pigeon that flew the last message out
> of Fort Vaux through the shellfire of Verdun in 1916 — and delivered it.
> That is retrieval under pressure: one messenger, the right message,
> against all the noise. This project does the same for your documents.

Retrieval-Augmented Generation (RAG) over your local documents, with **any
OpenAI-compatible LLM endpoint** — Ollama, LM Studio, vLLM, llama.cpp server,
OpenAI, Mistral, and more.

Point it at a folder of documents (PDF, Word, Excel, Markdown, plain text,
HTML), build a local index, and ask questions answered strictly from your
documents' content — fully offline if you use a local model. Retrieval is a
hybrid vector index (dense embeddings + BM25, fused with reciprocal rank
fusion), with optional cross-encoder re-ranking.

<p align="center">
  <img src="assets/demo.svg" alt="Animated demo of a vaillant-rag qa session" width="760"/>
</p>

## Table of contents

- [Why](#why)
- [Two-minute tour](#two-minute-tour)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
  - [1. Install](#1-install)
  - [2. Point at an LLM](#2-point-at-an-llm)
  - [3. Add documents and build the index](#3-add-documents-and-build-the-index)
  - [4. Ask questions](#4-ask-questions)
- [Web API](#web-api)
- [Scaling retrieval](#scaling-retrieval)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development](#development)
- [Scalability notes](#scalability-notes)
- [Security notes](#security-notes)
- [License](#license)

## Why

Most RAG examples are tied to a single provider SDK or a heavyweight
framework. `vaillant-rag` is a small, readable, dependency-light pipeline that:

- works with **any chat endpoint** speaking the OpenAI `/chat/completions` API;
- runs **fully local** (local embeddings + Ollama) with zero API cost;
- uses **hybrid retrieval** (dense embeddings + BM25, reciprocal rank fusion)
  — measured against dense-only search on the bundled corpus (see
  [Evaluation](#evaluation));
- supports **incremental indexing**: only changed documents are re-embedded.

## Two-minute tour

The repository ships with a small [example corpus](data/docs/) (the Universal
Declaration of Human Rights, a chapter of *The Rust Programming Language*, and
a solar-system fact sheet), so you can go from clone to answers immediately.

```bash
# 1. Clone and install (CPU-only PyTorch is fine)
git clone https://github.com/qthullie/vaillant-rag.git
cd vaillant-rag
pip install -e ".[office]"

# 2. Build the index over the bundled documents (downloads the embedding
#    model on first run, then works offline)
vaillant-rag index
# -> Index updated: 3 added/updated, 0 unchanged, 0 removed, 0 failed.

# 3a. Reproduce the retrieval evaluation — no LLM needed, fully offline
vaillant-rag eval --ablate
# -> a recall@k / MRR / latency table (see the Evaluation section)

# 3b. Or ask questions (needs an LLM endpoint; e.g. `ollama serve` with phi4-mini)
vaillant-rag qa
```

A `qa` session looks like this:

```text
vaillant-rag — qa
Loaded 50 chunks. Type 'quit' to exit.

Your question: What is the tallest volcano in the Solar System?

=== Answer ===
Olympus Mons, on Mars, is the tallest volcano in the Solar System, rising
about 22 kilometers above the surrounding plains.

(1.4s — streamed, hybrid search: dense + BM25, RRF)
```

The exact wording depends on your LLM; the retrieved context does not.

## Features

- **Multi-format ingestion**: `.pdf` (PyMuPDF, optional OCR of embedded
  images), `.txt`, `.md`, `.rst`, `.html`, `.docx` (Word), `.xlsx`/`.xlsm`
  (Excel) — Office formats via the `[office]` extra; legacy `.doc`/`.xls`
  must be converted first
- **Sentence-aware chunking** with configurable size and overlap
- **Hybrid search**: cosine similarity + BM25, fused with RRF
- **FAISS backend** (optional): approximate HNSW search for large
  collections; auto-selected above a configurable size threshold
- **Cross-encoder re-ranking** (optional): re-scores retrieval candidates
  for higher precision before they reach the LLM
- **Streaming answers**: tokens appear as the model generates them, in the
  CLI and over the API (Server-Sent Events)
- **FastAPI web service**: `/ask`, `/ask/stream`, `/reindex`, `/health` —
  multi-user ready with a shared pipeline instance
- **Incremental sync**: SHA-256 content hashing per document; deleted files
  are removed from the index
- **Provider-agnostic LLM client**: one `LLM_BASE_URL` away from switching
  between local and hosted models
- **Interactive Q&A CLI** and single-document analysis mode

## Architecture

```
documents (.pdf/.txt/.md/.html/.docx/.xlsx)
        │  loaders.py (extraction, optional OCR)
        ▼
   chunking.py (sentence-aware, size/overlap from config)
        │
        ▼
  embeddings.py (sentence-transformers, E5 prefixes handled)
        │
        ▼
 vector_store.py (numpy index + JSONL text store on disk)
        │
        ▼                            ┌──────────────────────────┐
  retrieval.py  ◄──── question ──────┤ pipeline.py (RAG)        │
  dense (numpy or FAISS) + BM25      │  ▲ cli.py    ▲ api.py    │
        │  RRF fusion, top-k         │  (terminal)  (FastAPI)   │
        ▼                            └────────────┬─────────────┘
  rerank.py (optional cross-encoder, top-n)       │
        │                                         ▼
        └──────────► llm.py — any OpenAI-compatible endpoint
                     (blocking or streamed via SSE)
```

## Quick start

### 1. Install

Requires Python ≥ 3.10.

```bash
git clone https://github.com/qthullie/vaillant-rag.git
cd vaillant-rag
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .
```

Optional OCR support for image-only PDFs (also requires a
[Tesseract](https://github.com/tesseract-ocr/tesseract) install):

```bash
pip install -e ".[ocr]"
```

Optional Word/Excel support (`.docx`, `.xlsx`, `.xlsm`):

```bash
pip install -e ".[office]"
```

### 2. Point at an LLM

Any OpenAI-compatible endpoint works. For a free, fully local setup with
[Ollama](https://ollama.com):

```bash
ollama pull phi4-mini
```

Copy the environment template and adjust if needed:

```bash
cp .env.example .env
```

```env
LLM_BASE_URL=http://localhost:11434/v1   # Ollama default
LLM_MODEL=phi4-mini
LLM_API_KEY=                             # only needed for hosted providers
```

Switching to OpenAI, Mistral, vLLM, LM Studio… is just a different
`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`.

### 3. Add documents and build the index

```bash
# The repo already ships an example corpus in data/docs/.
# Add your own files there too (or replace them), then:
vaillant-rag index
```

### 4. Ask questions

```bash
vaillant-rag qa
```

```
Your question: What does chapter 3 say about pricing?

=== Answer (4.2s) ===
...
```

Other commands:

```bash
vaillant-rag update            # incremental re-index (only changed files)
vaillant-rag analyze file.pdf  # run the system prompt against one document
vaillant-rag --config other.yaml qa
```

## Web API

Install the API extra and start the server:

```bash
pip install -e ".[api]"
vaillant-rag serve --host 127.0.0.1 --port 8000
```

Endpoints (interactive docs at `http://127.0.0.1:8000/docs`):

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + index stats |
| `POST` | `/ask` | `{"question": "..."}` → answer + source contexts |
| `POST` | `/ask/stream` | Same, streamed as Server-Sent Events |
| `POST` | `/reindex` | `{"full_rebuild": false}` → sync index with docs dir |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does chapter 3 say about pricing?"}'
```

The SSE stream emits one `data: {"contexts": [...]}` event, then
`data: {"delta": "..."}` events per token group, then `data: [DONE]`.

## Scaling retrieval

Two opt-in features improve quality and speed as your corpus grows:

**FAISS (speed, large collections).** Exact numpy search is fine up to
~10⁵ chunks. Beyond that:

```bash
pip install -e ".[faiss]"
```

With `vector_backend: auto` (default), FAISS HNSW kicks in automatically
once the collection exceeds `faiss_min_chunks` (50 000). Force it with
`vector_backend: faiss`.

**Cross-encoder re-ranking (quality).** Set `use_reranker: true` to
re-score the `top_k` candidates with a cross-encoder before selecting the
final `top_n_contexts`. On the bundled corpus it lifts recall@1 and MRR
markedly but adds ~1.5 s of CPU latency per query — see
[Evaluation](#evaluation) for the exact trade-off. Default model is English
(`cross-encoder/ms-marco-MiniLM-L6-v2`); for multilingual corpora use
`reranker_model: BAAI/bge-reranker-base`.

## Configuration

All tunables live in [`config.yaml`](config.yaml); every key can be
overridden by an environment variable of the same name in upper case
(`CHUNK_SIZE_CHARS=800`). Secrets belong in `.env` only.

| Key | Default | Description |
|---|---|---|
| `docs_dir` | `data/docs` | Source documents folder |
| `index_dir` | `data/index` | Generated index (gitignored) |
| `llm_base_url` | `http://localhost:11434/v1` | OpenAI-compatible API root |
| `llm_model` | `phi4-mini` | Model name at the endpoint |
| `llm_temperature` | `0.2` | Sampling temperature |
| `embedding_model` | `intfloat/multilingual-e5-small` | sentence-transformers model |
| `chunk_size_chars` | `1000` | Max chunk length |
| `chunk_overlap_chars` | `150` | Overlap between chunks |
| `top_k` | `20` | Retrieval candidates |
| `top_n_contexts` | `5` | Contexts sent to the LLM |
| `use_hybrid_search` | `true` | Dense + BM25 fusion |
| `vector_backend` | `auto` | `auto` / `numpy` / `faiss` |
| `faiss_min_chunks` | `50000` | Auto-switch threshold for FAISS |
| `use_reranker` | `false` | Cross-encoder re-ranking |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Re-ranking model |
| `ocr_images` | `false` | OCR images inside PDFs |
| `system_prompt_path` | `prompts/system_prompt.txt` | System prompt file (or inline text) |

Customize the assistant's behavior by editing
[`prompts/system_prompt.txt`](prompts/system_prompt.txt).

## Evaluation

`vaillant-rag eval` measures **retrieval** quality on a fixed question set
([`eval/questions.yaml`](eval/questions.yaml)) whose gold passages are known.
It scores nothing with the LLM, so it runs fully offline. `--ablate` compares
retrieval strategies on the same index; `--chunk-sizes` additionally rebuilds
temporary indexes to compare chunk sizes.

```bash
vaillant-rag index                              # build the index first
vaillant-rag eval --ablate --chunk-sizes 500,1500 --json eval/results.json
```

**What the metrics mean.** A retrieved chunk is *relevant* when it contains
one of the question's gold snippets. **recall@k** is the fraction of
answerable questions for which a relevant chunk appears in the top *k*
results. **MRR** (mean reciprocal rank) averages `1/rank` of the first
relevant chunk — it rewards ranking the right passage higher, not just
retrieving it. Latency is wall-clock time per query (query embedding +
search + any re-ranking).

### Results

Reference run committed at [`eval/results.json`](eval/results.json).
Embedding model `intfloat/multilingual-e5-small`; corpus of **3 documents
(50 chunks)**; **27 answerable + 3 unanswerable** questions; `top_k=20`;
Intel (Alder Lake) laptop CPU, **CPU-only**, Windows 11.

| Metric | dense | hybrid | hybrid + reranker | hybrid chunk=500 | hybrid chunk=1500 |
|---|---|---|---|---|---|
| recall@1 | 0.630 | 0.741 | **0.889** | 0.704 | 0.815 |
| recall@3 | 0.852 | 0.852 | **0.963** | 0.852 | 0.963 |
| recall@5 | 0.889 | **1.000** | 0.963 | 1.000 | 0.963 |
| recall@10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | 0.767 | 0.823 | **0.919** | 0.799 | 0.889 |
| latency p50 (ms) | 24 | 24 | 1555 | 33 | 37 |
| latency p95 (ms) | 29 | 26 | 1872 | 44 | 51 |

**Reading the numbers honestly:**

- **Hybrid vs dense** is a *real but modest* win on this corpus: recall@1
  rises 0.63 → 0.74 and MRR 0.77 → 0.82, and hybrid finds every answer within
  the top 5 (recall@5 = 1.00) where dense does not — at essentially no extra
  latency. But recall@3 is identical, and both retrieve every answer by
  rank 10. This is a weaker effect than the previous README asserted.
- **The re-ranker is the largest quality lever**: recall@1 0.74 → 0.89 and
  MRR 0.82 → 0.92. It costs ~65× the latency (24 ms → ~1.5 s on CPU),
  and it even nudges recall@5 *down* (1.00 → 0.96) by demoting one gold
  passage — a reminder that re-ranking optimizes the top of the list, not
  coverage.
- **Chunk size matters**: bigger chunks help here (recall@1 0.70 / 0.74 /
  0.82 at 500 / 1000 / 1500 chars), because more context lands in a single
  chunk on a small, prose-heavy corpus.

### Limitations

This is a **tiny benchmark** (50 chunks, 27 answerable questions), written by
the maintainer from the corpus itself. recall@10 saturates at 1.0 for every
configuration — a ceiling effect that a larger corpus would remove. Treat the
numbers as an honest, reproducible snapshot of *these* documents and this
embedding model, not as a general claim about hybrid search or re-ranking.
Retrieval quality is also not answer quality: the eval deliberately stops at
the retrieval layer. Metric definitions and edge cases are unit-tested in
[`tests/test_evaluation.py`](tests/test_evaluation.py).

## Development

```bash
pip install -e ".[dev]"
pytest            # offline test suite (embeddings faked, LLM mocked)
ruff check .      # lint
ruff format .     # format
```

CI (GitHub Actions) runs linting and the test suite on every push and PR.

## Scalability notes

- Default search is exact (numpy + BM25) — ideal up to ~10⁵ chunks; the
  FAISS backend covers larger collections (see *Scaling retrieval*).
- The API serves concurrent users from one shared pipeline; requests run
  in the server thread pool and re-indexing is serialized with a lock.
- BM25 and the FAISS index are rebuilt in memory at startup/reload; for
  very large corpora persist them separately or move to a vector database.

## Security notes

- No secrets are stored in the repository; `.env` is gitignored. Use
  `.env.example` as a template.
- Retrieved document text is sent verbatim to the LLM: treat untrusted
  documents as potential prompt-injection vectors and review answers
  accordingly.
- The web API has **no authentication**: keep it bound to `127.0.0.1`
  (the default) or put it behind a reverse proxy that handles auth
  before exposing it beyond your machine.
- Documents are parsed with PyMuPDF/python-docx/openpyxl: only index
  files you trust, and keep dependencies current (a `pip-audit` job runs
  in CI). The `[office]` extra includes `defusedxml`, which openpyxl
  picks up automatically to harden XML parsing.

## License

[MIT](LICENSE)
