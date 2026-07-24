"""Offline retrieval evaluation.

Measures retrieval quality (recall@k, MRR) and latency against a fixed set
of questions whose gold passages are known (``eval/questions.yaml``). No LLM
is involved: everything is scored at the retrieval layer, so the evaluation
runs fully offline, exactly like the rest of the test suite.

A question's gold passages are short verbatim snippets. A retrieved chunk is
*relevant* when its text contains any of those snippets after whitespace
normalization; a question is *hit* at rank *r* when the highest-ranked
relevant chunk sits at position *r* (1-based). recall@k is the fraction of
answerable questions hit within the first *k* results; MRR averages ``1/r``
(0 when the question is missed). Unanswerable questions carry no gold passage
and are excluded from the quality metrics.

The metric functions are deliberately pure (ranked text lists in, numbers
out) so they can be unit-tested with hand-built inputs and fake embeddings.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .retrieval import RetrievedChunk

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvalQuestion:
    """A single evaluation question and its retrieval ground truth."""

    id: str
    question: str
    answerable: bool
    sources: list[str] = field(default_factory=list)
    type: str = ""
    doc: str | None = None


@dataclass
class QuestionResult:
    """Per-question outcome for one configuration."""

    id: str
    answerable: bool
    rank: int | None  # 1-based rank of first relevant chunk; None if missed
    latency_ms: float


@dataclass
class EvalSummary:
    """Aggregate metrics for one configuration over the whole question set."""

    config: str
    recall_at_k: dict[int, float]
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    num_answerable: int
    num_unanswerable: int
    results: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr": round(self.mrr, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "num_answerable": self.num_answerable,
            "num_unanswerable": self.num_unanswerable,
            "per_question": [
                {"id": r.id, "rank": r.rank, "latency_ms": round(r.latency_ms, 2)}
                for r in self.results
            ],
        }


def _normalize(text: str) -> str:
    """Lowercase and collapse all runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_is_relevant(chunk_text: str, sources: Sequence[str]) -> bool:
    """True when ``chunk_text`` contains any gold snippet (normalized match)."""
    haystack = _normalize(chunk_text)
    return any(_normalize(s) in haystack for s in sources)


def rank_of_first_relevant(
    ranked_texts: Sequence[str], sources: Sequence[str]
) -> int | None:
    """1-based rank of the first relevant chunk, or None if none match."""
    for rank, text in enumerate(ranked_texts, start=1):
        if chunk_is_relevant(text, sources):
            return rank
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    """1.0 when a relevant chunk was found within the first ``k`` results."""
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """``1/rank`` for a hit, 0.0 for a miss."""
    return 1.0 / rank if rank else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 100]); 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (pct / 100) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Load and validate the evaluation questions from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a YAML list of questions")
    questions = []
    for entry in raw:
        answerable = bool(entry["answerable"])
        sources = list(entry.get("sources") or [])
        if answerable and not sources:
            raise ValueError(f"Answerable question {entry['id']!r} has no sources")
        questions.append(
            EvalQuestion(
                id=entry["id"],
                question=entry["question"],
                answerable=answerable,
                sources=sources,
                type=entry.get("type", ""),
                doc=entry.get("doc"),
            )
        )
    return questions


# A retrieval function maps a question to a ranked list of candidate chunks.
RetrieveFn = Callable[[str], list[RetrievedChunk]]


def evaluate(
    config: str,
    retrieve_fn: RetrieveFn,
    questions: Sequence[EvalQuestion],
    ks: Sequence[int] = DEFAULT_KS,
) -> EvalSummary:
    """Run one configuration over every question and aggregate the metrics.

    Args:
        config: Human-readable name of the configuration being measured.
        retrieve_fn: Returns the ranked candidate chunks for a question.
        questions: The evaluation questions.
        ks: The cut-offs at which to report recall.
    """
    results: list[QuestionResult] = []
    latencies: list[float] = []
    for q in questions:
        started = time.perf_counter()
        chunks = retrieve_fn(q.question)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        rank = rank_of_first_relevant([c.text for c in chunks], q.sources) if q.answerable else None
        results.append(QuestionResult(q.id, q.answerable, rank, latency_ms))

    answerable = [r for r in results if r.answerable]
    n = len(answerable)
    recall = {
        k: (sum(recall_at_k(r.rank, k) for r in answerable) / n if n else 0.0) for k in ks
    }
    mrr = sum(reciprocal_rank(r.rank) for r in answerable) / n if n else 0.0
    return EvalSummary(
        config=config,
        recall_at_k=recall,
        mrr=mrr,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        num_answerable=n,
        num_unanswerable=len(results) - n,
        results=results,
    )
