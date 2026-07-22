import pytest

from vaillant_rag.chunking import chunk_text, split_sentences


def test_empty_text_returns_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n  ", 100, 10) == []


def test_short_text_single_chunk():
    assert chunk_text("Hello world.", 100, 10) == ["Hello world."]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i}." for i in range(100))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def test_no_content_lost():
    text = " ".join(f"Word{i}." for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    joined = " ".join(chunks)
    for i in range(200):
        assert f"Word{i}." in joined


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence number {i} here." for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=40)
    # With sentence-carry overlap, the start of each chunk repeats the
    # end of the previous one.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.split(".")[0] + "."
        assert first_sentence in previous


def test_overlap_geq_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=60)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_oversized_sentence_hard_split():
    text = "x" * 500  # no sentence boundary at all
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c.replace(" ", "")) for c in chunks) >= 500


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
