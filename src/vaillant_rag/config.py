"""Application settings.

Configuration is resolved in three layers, lowest to highest priority:

1. Built-in defaults (defined on :class:`Settings`).
2. An optional YAML file (``config.yaml`` by default).
3. Environment variables (optionally loaded from a ``.env`` file),
   using the upper-cased field name (e.g. ``CHUNK_SIZE_CHARS``).

Secrets (``LLM_API_KEY``) should only ever be provided via environment
variables, never committed to the YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_FILE = "config.yaml"

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Settings:
    """All tunable parameters for the RAG pipeline."""

    # --- Paths ---
    docs_dir: str = "data/docs"
    index_dir: str = "data/index"
    system_prompt_path: str = "prompts/system_prompt.txt"
    log_dir: str = "logs"
    log_file: str = "vaillant_rag.log"
    log_level: str = "INFO"

    # --- LLM (any OpenAI-compatible chat completions endpoint) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""  # optional; required by hosted providers
    llm_model: str = "phi4-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 180

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 64

    # --- Chunking ---
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150

    # --- Retrieval ---
    top_k: int = 20
    top_n_contexts: int = 5
    use_hybrid_search: bool = True
    vector_backend: str = "auto"  # auto | numpy | faiss
    faiss_min_chunks: int = 50_000  # auto mode: switch to FAISS above this size
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # --- Document extraction ---
    ocr_images: bool = False  # OCR images embedded in PDFs (requires the [ocr] extra)
    max_analysis_chars: int = 20000  # cap on document text sent to the LLM by `analyze`

    def validate(self) -> None:
        """Raise :class:`ValueError` when parameter combinations are unusable."""
        if self.chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than chunk_size_chars "
                f"(got overlap={self.chunk_overlap_chars}, size={self.chunk_size_chars})"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive (got {self.top_k})")
        if self.top_n_contexts <= 0:
            raise ValueError(f"top_n_contexts must be positive (got {self.top_n_contexts})")
        if self.top_n_contexts > self.top_k:
            raise ValueError(
                "top_n_contexts cannot exceed top_k "
                f"(got top_n_contexts={self.top_n_contexts}, top_k={self.top_k})"
            )
        if not self.llm_base_url:
            raise ValueError("llm_base_url must be set (e.g. http://localhost:11434/v1)")
        if self.vector_backend not in ("auto", "numpy", "faiss"):
            raise ValueError(
                f"vector_backend must be one of auto/numpy/faiss, got {self.vector_backend!r}"
            )

    @property
    def system_prompt(self) -> str:
        """Return the system prompt, loaded from file if the path exists."""
        path = Path(self.system_prompt_path)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        # Fall back to treating the value as an inline prompt string.
        return self.system_prompt_path


def load_settings(config_file: str | None = None) -> Settings:
    """Build a validated :class:`Settings` from defaults, YAML file, and environment.

    Args:
        config_file: Path to a YAML config file. Defaults to ``config.yaml``
            in the current working directory; silently skipped if absent.

    Returns:
        A validated, immutable :class:`Settings` instance.

    Raises:
        ValueError: If a value cannot be cast to the expected type or the
            resulting configuration fails validation.
    """
    load_dotenv()

    values: dict[str, object] = {}

    path = Path(config_file or DEFAULT_CONFIG_FILE)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            file_values = yaml.safe_load(f) or {}
        if not isinstance(file_values, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")
        values.update(file_values)

    for field in fields(Settings):
        env_value = os.environ.get(field.name.upper())
        if env_value is not None:
            values[field.name] = env_value

    known = {f.name: f for f in fields(Settings)}
    kwargs: dict[str, object] = {}
    for name, raw in values.items():
        if name not in known:
            raise ValueError(f"Unknown configuration key: {name!r}")
        target_type = known[name].type
        try:
            if target_type == "bool":
                kwargs[name] = _as_bool(raw)  # type: ignore[arg-type]
            elif target_type == "int":
                kwargs[name] = int(raw)  # type: ignore[call-overload]
            elif target_type == "float":
                kwargs[name] = float(raw)  # type: ignore[arg-type]
            else:
                kwargs[name] = str(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for configuration key {name!r}: {raw!r} ({exc})"
            ) from exc

    settings = Settings(**kwargs)  # type: ignore[arg-type]
    settings.validate()
    return settings
