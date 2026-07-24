"""Unit tests for the retrieval evaluation metrics.

The metric layer is pure, so these tests use hand-built ranked lists and a
fake retrieve function — no models or embeddings are loaded.
"""

from __future__ import annotations

import pytest

from vaillant_rag.evaluation import (
    EvalQuestion,
    _percentile,
    chunk_is_relevant,
    evaluate,
    load_questions,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)
from vaillant_rag.retrieval import RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="x", text=text, score=0.0)


def test_chunk_is_relevant_normalizes_whitespace_and_case():
    assert chunk_is_relevant("The  QUICK\nbrown fox", ["quick brown"])
    assert not chunk_is_relevant("nothing here", ["quick brown"])


def test_rank_of_first_relevant():
    texts = ["irrelevant", "still no", "here is the answer passage", "later"]
    assert rank_of_first_relevant(texts, ["the answer passage"]) == 3
    assert rank_of_first_relevant(texts, ["absent"]) is None


def test_rank_matches_any_source():
    texts = ["alpha content", "beta content"]
    # second source matches the first chunk -> rank 1
    assert rank_of_first_relevant(texts, ["zzz", "alpha content"]) == 1


def test_recall_at_k_boundaries():
    assert recall_at_k(3, 3) == 1.0
    assert recall_at_k(3, 2) == 0.0
    assert recall_at_k(None, 10) == 0.0
    assert recall_at_k(1, 1) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0


def test_percentile():
    assert _percentile([], 50) == 0.0
    assert _percentile([5.0], 95) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert _percentile([10.0, 20.0, 30.0], 100) == 30.0


def test_evaluate_recall_and_mrr_on_known_inputs():
    # A fake retriever returning a fixed ranking keyed by which question asks.
    rankings = {
        "q1": ["gold one is here", "noise", "noise"],  # hit at rank 1
        "q2": ["noise", "noise", "gold two here"],  # hit at rank 3
        "q3": ["noise", "noise", "noise"],  # miss
    }

    def retrieve(question: str):
        return [_chunk(t) for t in rankings[question]]

    questions = [
        EvalQuestion("q1", "q1", answerable=True, sources=["gold one is here"]),
        EvalQuestion("q2", "q2", answerable=True, sources=["gold two here"]),
        EvalQuestion("q3", "q3", answerable=True, sources=["never appears"]),
        EvalQuestion("q4", "q1", answerable=False, sources=[]),  # excluded from quality
    ]

    summary = evaluate("fake", retrieve, questions, ks=(1, 3))

    assert summary.num_answerable == 3
    assert summary.num_unanswerable == 1
    # recall@1: only q1 hits within top-1 -> 1/3
    assert summary.recall_at_k[1] == pytest.approx(1 / 3)
    # recall@3: q1 and q2 hit -> 2/3
    assert summary.recall_at_k[3] == pytest.approx(2 / 3)
    # MRR: (1/1 + 1/3 + 0) / 3
    assert summary.mrr == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_load_questions_rejects_answerable_without_sources(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- id: bad\n  question: q\n  answerable: true\n  sources: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no sources"):
        load_questions(path)


def test_load_questions_reads_real_set():
    questions = load_questions("eval/questions.yaml")
    assert len(questions) == 30
    assert sum(1 for q in questions if not q.answerable) == 3
    assert all(q.sources for q in questions if q.answerable)
