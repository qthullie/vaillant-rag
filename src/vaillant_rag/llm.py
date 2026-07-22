"""LLM client for any OpenAI-compatible chat completions endpoint.

Works with Ollama (``http://localhost:11434/v1``), vLLM, LM Studio,
llama.cpp server, OpenAI, Mistral, and any other provider exposing
``POST {base_url}/chat/completions``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import requests

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM endpoint returns an error or malformed response."""


class ChatClient:
    """Minimal chat-completions client.

    Args:
        base_url: API root, e.g. ``http://localhost:11434/v1``.
        model: Model name understood by the endpoint.
        api_key: Optional bearer token (required by hosted providers).
        temperature: Sampling temperature.
        max_tokens: Response token cap.
        timeout_seconds: HTTP request timeout.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_seconds: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def _request_parts(
        self, system_prompt: str, user_content: str, stream: bool
    ) -> tuple[str, dict, dict]:
        """Build (url, headers, payload) for a chat completions call."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        return f"{self.base_url}/chat/completions", headers, payload

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Send one system+user exchange and return the assistant text.

        Raises:
            LLMError: On connection failure, HTTP error, or malformed response.
        """
        url, headers, payload = self._request_parts(system_prompt, user_content, stream=False)
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError as exc:
            raise LLMError(
                f"Cannot reach LLM endpoint at {url}. Is the server running (e.g. `ollama serve`)?"
            ) from exc
        except requests.Timeout as exc:
            raise LLMError(f"LLM request timed out after {self.timeout_seconds}s") from exc
        except requests.HTTPError as exc:
            raise LLMError(f"LLM endpoint error: {exc} — {response.text[:500]}") from exc
        except ValueError as exc:  # JSON decoding
            raise LLMError(f"LLM endpoint returned non-JSON response: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from LLM endpoint: {data}") from exc

    def chat_stream(self, system_prompt: str, user_content: str) -> Iterator[str]:
        """Stream the assistant response as text deltas (SSE).

        Yields content fragments as the model produces them.

        Raises:
            LLMError: On connection failure, HTTP error, or malformed stream.
        """
        url, headers, payload = self._request_parts(system_prompt, user_content, stream=True)
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout_seconds, stream=True
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise LLMError(
                f"Cannot reach LLM endpoint at {url}. Is the server running (e.g. `ollama serve`)?"
            ) from exc
        except requests.Timeout as exc:
            raise LLMError(f"LLM request timed out after {self.timeout_seconds}s") from exc
        except requests.HTTPError as exc:
            raise LLMError(f"LLM endpoint error: {exc} — {response.text[:500]}") from exc

        with response:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMError(f"Malformed stream chunk from LLM endpoint: {data!r}") from exc
                content = delta.get("content")
                if content:
                    yield content


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    """Assemble the user message for a grounded question-answering call."""
    numbered = "\n\n".join(
        f"Context {i}:\n{context}" for i, context in enumerate(contexts, start=1)
    )
    return (
        "Answer the question using only the context below. "
        "If the context is insufficient, say so instead of guessing.\n\n"
        f"{numbered}\n\nQuestion: {question}"
    )
