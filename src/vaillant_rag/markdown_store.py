"""Markdown corpus store for LLM-navigated ("agentic") retrieval.

Optional alternative to the embedding index (``retrieval_mode: markdown``).
Documents are converted to Markdown at index time and split into
heading-delimited sections. At question time the LLM reads a numbered
outline of the corpus, picks the sections worth consulting, and the
answer is generated from their full text — navigate-then-read, the way
agentic assistants browse a codebase, instead of similarity search over
embedded chunks. No embeddings are involved: this mode needs neither
sentence-transformers nor FAISS at query time.

Storage layout inside ``<index_dir>/markdown/``:

- ``<source name>.md`` — one Markdown rendering per source document
  (e.g. ``report.pdf.md``).
- ``doc_hashes.json``  — SHA-256 per source document, for incremental sync.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

MARKDOWN_SUBDIR = "markdown"
HASHES_FILE = "doc_hashes.json"

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)  # same tokenizer as retrieval.py

SECTION_SELECT_SYSTEM = (
    "You are a retrieval planner for a document question-answering system. "
    "Given a question and a numbered outline of document sections, reply with "
    "the numbers of the sections most likely to contain the answer, "
    "comma-separated, most relevant first. Reply with numbers only."
)


def markdown_dir(index_dir: str | Path) -> Path:
    """The Markdown corpus directory inside the index directory."""
    return Path(index_dir) / MARKDOWN_SUBDIR


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Section:
    """A heading-delimited slice of one document's Markdown rendering."""

    section_id: str  # "<doc name>#<ordinal>"
    doc_name: str
    heading: str
    text: str


def split_sections(doc_name: str, markdown_text: str, max_chars: int) -> list[Section]:
    """Split a document's Markdown into heading-delimited sections.

    Text before the first heading becomes a ``(start)`` section; sections
    longer than ``max_chars`` are split into ``(part n)`` pieces so one
    selected section can never flood the prompt.
    """
    raw: list[tuple[str, list[str]]] = []
    heading = "(start)"
    accumulated: list[str] = []
    for line in markdown_text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            if any(part.strip() for part in accumulated):
                raw.append((heading, accumulated))
            heading = match.group(2)
            accumulated = []
        else:
            accumulated.append(line)
    if any(part.strip() for part in accumulated):
        raw.append((heading, accumulated))

    sections: list[Section] = []
    for heading, lines in raw:
        text = "\n".join(lines).strip()
        pieces = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        for part_number, piece in enumerate(pieces, start=1):
            title = heading if len(pieces) == 1 else f"{heading} (part {part_number})"
            sections.append(
                Section(
                    section_id=f"{doc_name}#{len(sections)}",
                    doc_name=doc_name,
                    heading=title,
                    text=piece,
                )
            )
    return sections


@dataclass
class MarkdownStore:
    """In-memory section catalog parsed from the on-disk Markdown corpus."""

    sections: list[Section] = field(default_factory=list)
    doc_hashes: dict[str, str] = field(default_factory=dict)
    _bm25: BM25Okapi | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def is_empty(self) -> bool:
        return not self.sections

    @classmethod
    def load(cls, index_dir: str | Path, section_max_chars: int) -> MarkdownStore:
        """Parse ``<index_dir>/markdown/*.md``; empty store when absent."""
        directory = markdown_dir(index_dir)
        sections: list[Section] = []
        if directory.is_dir():
            for md_path in sorted(directory.glob("*.md")):
                doc_name = md_path.name[: -len(".md")]
                text = md_path.read_text(encoding="utf-8", errors="replace")
                sections.extend(split_sections(doc_name, text, section_max_chars))

        doc_hashes: dict[str, str] = {}
        hashes_path = directory / HASHES_FILE
        if hashes_path.is_file():
            try:
                doc_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Ignoring corrupt %s: %s", hashes_path, exc)
        return cls(sections=sections, doc_hashes=doc_hashes)

    def bm25_select(self, question: str, n: int) -> list[Section]:
        """Rank sections lexically (BM25) — outline pre-filter and fallback."""
        if not self.sections:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi(
                [_tokenize(f"{s.doc_name}\n{s.heading}\n{s.text}") for s in self.sections]
            )
        scores = self._bm25.get_scores(_tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.sections[i] for i in ranked]


def format_outline(sections: list[Section], preview_chars: int = 90) -> str:
    """Numbered one-line-per-section outline shown to the LLM.

    Headings and previews are capped so one pathological document cannot
    inflate the selection prompt.
    """
    lines = []
    for number, section in enumerate(sections, start=1):
        heading = section.heading[:120]
        preview = " ".join(section.text.split())[:preview_chars]
        lines.append(f"{number}. [{section.doc_name}] {heading} — {preview}")
    return "\n".join(lines)


def build_selection_prompt(question: str, outline: str, max_picks: int) -> str:
    """User message for the section-selection call."""
    return (
        f"Question: {question}\n\n"
        f"Sections:\n{outline}\n\n"
        f"Pick at most {max_picks} section numbers. Numbers only."
    )


def parse_selection(reply: str, n_sections: int) -> list[int]:
    """Extract valid 1-based outline numbers from the LLM reply.

    Deduplicates while preserving order; out-of-range numbers are
    dropped. An empty result means the caller should fall back to BM25.
    """
    picks: list[int] = []
    for token in re.findall(r"\d+", reply):
        number = int(token)
        if 1 <= number <= n_sections and number not in picks:
            picks.append(number)
    return picks
