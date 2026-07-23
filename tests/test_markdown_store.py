"""Markdown retrieval mode: corpus sync, section store, LLM navigation.

Runs offline: the LLM is faked at the pipeline level and no embedding
model is ever loaded (markdown mode does not use embeddings at all).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vaillant_rag.indexing import sync_index
from vaillant_rag.llm import LLMError
from vaillant_rag.markdown_store import (
    MarkdownStore,
    format_outline,
    parse_selection,
    split_sections,
)
from vaillant_rag.pipeline import RagPipeline


@pytest.fixture
def md_settings(settings):
    """The shared settings fixture, switched to markdown mode."""
    return replace(settings, retrieval_mode="markdown")


def _write_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "pigeons.md").write_text(
        "# Pigeons\n\nIntro about pigeons.\n\n"
        "## Cher Ami\n\nCher Ami saved nearly 200 soldiers in 1918.\n\n"
        "## Vaillant\n\nVaillant flew the last message out of Fort Vaux in 1916.\n",
        encoding="utf-8",
    )
    (docs_dir / "python.txt").write_text(
        "Python is a programming language created by Guido van Rossum.",
        encoding="utf-8",
    )


# --- section splitting -------------------------------------------------


def test_split_sections_by_headings():
    sections = split_sections("doc.md", "intro\n\n# A\ntext a\n\n## B\ntext b\n", max_chars=1000)
    assert [s.heading for s in sections] == ["(start)", "A", "B"]
    assert sections[1].text == "text a"
    assert [s.section_id for s in sections] == ["doc.md#0", "doc.md#1", "doc.md#2"]


def test_split_sections_caps_length():
    sections = split_sections("doc.md", "# Long\n" + "x" * 250, max_chars=100)
    assert len(sections) == 3
    assert sections[0].heading == "Long (part 1)"
    assert all(len(s.text) <= 100 for s in sections)


def test_split_sections_empty_text():
    assert split_sections("doc.md", "   \n\n", max_chars=100) == []


# --- selection helpers -------------------------------------------------


def test_parse_selection_filters_and_dedupes():
    assert parse_selection("Sections 2, 1 and 2 then 9", n_sections=4) == [2, 1]
    assert parse_selection("none of them", n_sections=4) == []


def test_format_outline_numbers_and_previews():
    sections = split_sections("doc.md", "# Title\nsome body text\n", max_chars=1000)
    outline = format_outline(sections)
    assert outline.startswith("1. [doc.md] Title — some body text")


# --- corpus sync -------------------------------------------------------


def test_sync_builds_and_updates_markdown_corpus(md_settings):
    _write_docs(Path(md_settings.docs_dir))

    report = sync_index(md_settings)
    assert sorted(report.added_or_updated) == ["pigeons.md", "python.txt"]
    assert not report.failed

    store = MarkdownStore.load(md_settings.index_dir, md_settings.markdown_section_max_chars)
    assert not store.is_empty
    assert set(store.doc_hashes) == {"pigeons.md", "python.txt"}
    headings = [s.heading for s in store.sections]
    assert "Cher Ami" in headings and "Vaillant" in headings

    # Incremental: nothing changed, nothing re-rendered.
    report2 = sync_index(md_settings)
    assert sorted(report2.skipped_unchanged) == ["pigeons.md", "python.txt"]
    assert not report2.added_or_updated

    # Deleted source is dropped from the corpus.
    (Path(md_settings.docs_dir) / "python.txt").unlink()
    report3 = sync_index(md_settings)
    assert report3.removed == ["python.txt"]
    store = MarkdownStore.load(md_settings.index_dir, md_settings.markdown_section_max_chars)
    assert set(store.doc_hashes) == {"pigeons.md"}
    assert all(s.doc_name == "pigeons.md" for s in store.sections)


def test_bm25_select_ranks_keyword_match_first(md_settings):
    _write_docs(Path(md_settings.docs_dir))
    sync_index(md_settings)
    store = MarkdownStore.load(md_settings.index_dir, md_settings.markdown_section_max_chars)
    top = store.bm25_select("Which pigeon saved soldiers in 1918?", 1)
    assert top[0].heading == "Cher Ami"


# --- pipeline (LLM-navigated retrieval) --------------------------------


def _make_pipeline(md_settings, chat_replies: list[str]) -> tuple[RagPipeline, list]:
    """Pipeline with a scripted ``llm.chat``; records the calls."""
    pipeline = RagPipeline(md_settings)
    calls: list[tuple[str, str]] = []
    replies = iter(chat_replies)

    def fake_chat(system_prompt: str, user_content: str) -> str:
        calls.append((system_prompt, user_content))
        reply = next(replies)
        if reply == "<error>":
            raise LLMError("endpoint down")
        return reply

    pipeline.llm.chat = fake_chat  # type: ignore[method-assign]
    return pipeline, calls


def test_pipeline_llm_picks_sections(md_settings):
    _write_docs(Path(md_settings.docs_dir))
    sync_index(md_settings)

    # Find Cher Ami's outline number to make the test robust to ordering.
    store = MarkdownStore.load(md_settings.index_dir, md_settings.markdown_section_max_chars)
    outline_pick = next(
        number
        for number, section in enumerate(store.sections, start=1)
        if section.heading == "Cher Ami"
    )

    # First chat call selects sections; second generates the answer.
    pipeline, calls = _make_pipeline(
        md_settings, [str(outline_pick), "Cher Ami saved soldiers in 1918."]
    )
    result = pipeline.ask("Which pigeon saved soldiers?")

    assert result.answer == "Cher Ami saved soldiers in 1918."
    assert len(result.contexts) == 1
    assert "Cher Ami" in result.contexts[0].text
    # Selection call saw the outline; answer call saw the section text.
    assert "1." in calls[0][1] and "Which pigeon saved soldiers?" in calls[0][1]
    assert "Cher Ami saved nearly 200 soldiers" in calls[1][1]


def test_pipeline_falls_back_to_bm25_on_useless_selection(md_settings):
    _write_docs(Path(md_settings.docs_dir))
    sync_index(md_settings)
    pipeline, _calls = _make_pipeline(md_settings, ["no idea", "Cher Ami saved soldiers in 1918."])
    result = pipeline.ask("Which pigeon saved soldiers in 1918?")
    assert result.contexts  # BM25 fallback still grounded the answer
    assert any("Cher Ami" in c.text for c in result.contexts)


def test_pipeline_falls_back_to_bm25_on_selection_error(md_settings):
    _write_docs(Path(md_settings.docs_dir))
    sync_index(md_settings)
    pipeline, _calls = _make_pipeline(md_settings, ["<error>", "Cher Ami saved soldiers in 1918."])
    result = pipeline.ask("Which pigeon saved soldiers in 1918?")
    assert result.answer == "Cher Ami saved soldiers in 1918."
    assert result.contexts


def test_pipeline_empty_corpus_message(md_settings):
    Path(md_settings.docs_dir).mkdir(parents=True, exist_ok=True)
    pipeline, _calls = _make_pipeline(md_settings, [])
    result = pipeline.ask("anything?")
    assert "empty" in result.answer
    assert result.contexts == []
