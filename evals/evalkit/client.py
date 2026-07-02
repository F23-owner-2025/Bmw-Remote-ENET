"""Chat clients.

ChatClient speaks the OpenAI chat-completions protocol — the least common
denominator across vLLM, llama.cpp server, Ollama, and hosted APIs. Tool
schemas are NOT passed via the API `tools` field: they are embedded in the
system prompt in the exact Hermes format the model was trained on, and the
model answers with <tool_call> tags in plain content. That keeps behavior
identical across servers regardless of their native tool-calling support.

ScriptedClient is the deterministic stand-in used by the test suite.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

Message = Dict[str, str]


class ClientError(RuntimeError):
    pass


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 3072,
        timeout: float = 300.0,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: List[Message]) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.timeout)
                if resp.status_code >= 500:
                    raise ClientError(f"server error {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    # 4xx is a configuration problem, not transient — no retry.
                    raise SystemExit(
                        f"endpoint rejected request ({resp.status_code}): "
                        f"{resp.text[:500]}"
                    )
                data = resp.json()
                content = data["choices"][0]["message"].get("content")
                if content is None:
                    raise ClientError(f"no content in response: {str(data)[:300]}")
                return content
            except (ClientError, OSError) as err:  # OSError covers requests' IO errors
                last_err = err
                if attempt < self.max_retries:
                    wait = 2.0 * (2 ** attempt)
                    print(f"[client] {err} — retry {attempt + 1}/{self.max_retries} "
                          f"in {wait:.0f}s")
                    time.sleep(wait)
        raise ClientError(f"chat failed after {self.max_retries + 1} attempts: {last_err}")


class ScriptedClient:
    """Returns canned responses in order, or via a callback on the messages.

    Used by tests and the smoke run. Records every request for assertions.
    """

    def __init__(self, responses: Optional[List[str]] = None,
                 fn: Optional[Callable[[List[Message]], str]] = None):
        if (responses is None) == (fn is None):
            raise ValueError("provide exactly one of responses / fn")
        self._responses = list(responses) if responses else None
        self._fn = fn
        self.requests: List[List[Message]] = []
        self.model = "scripted"

    def chat(self, messages: List[Message]) -> str:
        self.requests.append([dict(m) for m in messages])
        if self._fn:
            return self._fn(messages)
        if not self._responses:
            raise ClientError("ScriptedClient ran out of responses")
        return self._responses.pop(0)
