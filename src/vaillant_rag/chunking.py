"""Sentence-aware text chunking.

Splits text into sentences, then greedily packs sentences into chunks of at
most ``chunk_size`` characters with a configurable character overlap between
consecutive chunks. Sentences longer than ``chunk_size`` are hard-split.
"""

from __future__ import annotations

import re

# End-of-sentence punctuation followed by whitespace, or a newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units, preserving all non-whitespace content."""
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text)]
    return [p for p in parts if p]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping, sentence-aware chunks.

    Args:
        text: Input text; may be empty.
        chunk_size: Maximum chunk length in characters. Must be positive.
        overlap: Approximate number of characters repeated between
            consecutive chunks. Must satisfy ``0 <= overlap < chunk_size``.

    Returns:
        List of non-empty chunk strings. Empty list for empty input.

    Raises:
        ValueError: If ``chunk_size`` or ``overlap`` are out of range
            (an ``overlap >= chunk_size`` would otherwise loop forever).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")
    if not text or not text.strip():
        return []

    # Hard-split any sentence that alone exceeds chunk_size.
    sentences: list[str] = []
    for sentence in split_sentences(text):
        while len(sentence) > chunk_size:
            sentences.append(sentence[:chunk_size])
            sentence = sentence[chunk_size - overlap if overlap else chunk_size :]
        if sentence:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        # +1 accounts for the joining space.
        added = len(sentence) + (1 if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences into the next chunk as overlap.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) + (1 if carried else 0) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + (1 if carried_len else 0)
            current = carried
            current_len = carried_len
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += added
    if current:
        chunks.append(" ".join(current))
    return chunks
