import json
from unittest.mock import MagicMock, patch

import pytest

from vaillant_rag.llm import ChatClient, LLMError, build_rag_prompt


def _client() -> ChatClient:
    return ChatClient(base_url="http://fake:1/v1", model="test-model")


def _sse_lines(*deltas: str) -> list[str]:
    lines = []
    for delta in deltas:
        chunk = {"choices": [{"delta": {"content": delta}}]}
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # SSE blank separator
    lines.append("data: [DONE]")
    return lines


def test_chat_stream_yields_fragments():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = _sse_lines("Hel", "lo", "!")
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    with patch("vaillant_rag.llm.requests.post", return_value=response) as mock_post:
        fragments = list(_client().chat_stream("system", "user"))
    assert fragments == ["Hel", "lo", "!"]
    assert mock_post.call_args.kwargs["json"]["stream"] is True


def test_chat_stream_stops_at_done_marker():
    response = MagicMock()
    response.raise_for_status.return_value = None
    lines = _sse_lines("keep")
    extra = {"choices": [{"delta": {"content": "after-done, must not appear"}}]}
    lines.append(f"data: {json.dumps(extra)}")
    response.iter_lines.return_value = lines
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    with patch("vaillant_rag.llm.requests.post", return_value=response):
        assert list(_client().chat_stream("s", "u")) == ["keep"]


def test_chat_stream_malformed_chunk_raises():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = ["data: {not json"]
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    with (
        patch("vaillant_rag.llm.requests.post", return_value=response),
        pytest.raises(LLMError, match="Malformed stream chunk"),
    ):
        list(_client().chat_stream("s", "u"))


def test_chat_connection_error_message():
    import requests as requests_lib

    with (
        patch("vaillant_rag.llm.requests.post", side_effect=requests_lib.ConnectionError("boom")),
        pytest.raises(LLMError, match="Cannot reach LLM endpoint"),
    ):
        _client().chat("s", "u")


def test_build_rag_prompt_numbers_contexts():
    prompt = build_rag_prompt("Q?", ["ctx one", "ctx two"])
    assert "Context 1:\nctx one" in prompt
    assert "Context 2:\nctx two" in prompt
    assert prompt.endswith("Question: Q?")
