"""FastAPI service tests: offline (fake embedder, mocked LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vaillant_rag.api import create_app  # noqa: E402
from vaillant_rag.indexing import sync_index  # noqa: E402


class _FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "Grounded answer."}}]}


class _FakeStreamResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False

    def iter_lines(self, decode_unicode: bool = True):
        for delta in ["Stream", "ed."]:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}"
        yield "data: [DONE]"


@pytest.fixture
def client(settings, fake_embedder) -> TestClient:
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True)
    (docs_dir / "facts.txt").write_text(
        "The pigeon Cher Ami saved nearly 200 soldiers in 1918.", encoding="utf-8"
    )
    with patch("vaillant_rag.indexing.Embedder", return_value=fake_embedder):
        sync_index(settings)
    with patch("vaillant_rag.pipeline.Embedder", return_value=fake_embedder):
        app = create_app(settings)
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["documents"] == 1


def test_ask_returns_answer_and_contexts(client):
    with patch("vaillant_rag.llm.requests.post", return_value=_FakeResponse()):
        response = client.post("/ask", json={"question": "Which pigeon saved soldiers?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer."
    assert body["contexts"]
    assert "Cher Ami" in body["contexts"][0]["text"]


def test_ask_validation_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_llm_failure_returns_502(client):
    import requests as requests_lib

    with patch("vaillant_rag.llm.requests.post", side_effect=requests_lib.ConnectionError("down")):
        response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    # Detail stays generic: upstream error bodies must not leak to clients.
    assert response.json()["detail"] == "LLM backend error; see server logs."


def test_ask_stream_sends_contexts_then_deltas(client):
    with patch("vaillant_rag.llm.requests.post", return_value=_FakeStreamResponse()):
        response = client.post("/ask/stream", json={"question": "Which pigeon?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    head = json.loads(events[0])
    assert "contexts" in head and head["contexts"]
    deltas = [json.loads(e)["delta"] for e in events[1:-1]]
    assert "".join(deltas) == "Streamed."
    assert events[-1] == "[DONE]"


def test_reindex_picks_up_new_document(client, settings, fake_embedder):
    (Path(settings.docs_dir) / "new.txt").write_text("Fresh document.", encoding="utf-8")
    with patch("vaillant_rag.indexing.Embedder", return_value=fake_embedder):
        response = client.post("/reindex", json={"full_rebuild": False})
    assert response.status_code == 200
    body = response.json()
    assert "new.txt" in body["added_or_updated"]
    assert client.get("/health").json()["documents"] == 2
