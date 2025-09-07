import json
import io
import logging

from agentx.config import Config
from agentx.providers.openai import OpenAIProvider
from agentx.providers.base import ChatMessage, ChatRequest


class DummyResponse:
    def __init__(self, body: bytes, lines=None):
        self._body = body
        self._lines = lines

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines or [])


def test_openai_complete_parses_content(monkeypatch):
    cfg = Config.from_dict({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "endpoints": {"openai": "https://example.com/v1"},
        "keys": {"OPENAI_API_KEY": "test"},
    })
    provider = OpenAIProvider(config=cfg, logger=logging.getLogger("test"))

    body = json.dumps({
        "choices": [{"message": {"content": "hello"}}]
    }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: DummyResponse(body))

    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model=cfg.model, stream=False)
    resp = provider.complete(req)
    assert resp.content == "hello"


def test_openai_stream_parses_sse(monkeypatch):
    cfg = Config.from_dict({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "endpoints": {"openai": "https://example.com/v1"},
        "keys": {"OPENAI_API_KEY": "test"},
    })
    provider = OpenAIProvider(config=cfg, logger=logging.getLogger("test"))

    lines = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Hel\"}}] }\n",
        b"\n",
        b"data: {\"choices\": [{\"delta\": {\"content\": \"lo\"}}] }\n",
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: DummyResponse(b"", lines))

    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model=cfg.model, stream=True)
    chunks = [d.content for d in provider.stream(req) if d.content]
    assert "".join(chunks) == "Hello"

