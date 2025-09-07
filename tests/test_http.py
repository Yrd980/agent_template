import io
from urllib import error

import pytest

from agentx.http import post_json
from agentx.errors import ProviderError


class DummyResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body


def test_post_json_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            fp = io.BytesIO(b"rate limited")
            raise error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs=None, fp=fp)
        return DummyResponse(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    resp = post_json(
        base_url="https://example.com/v1",
        path="chat/completions",
        payload={"ok": True},
        headers={"Content-Type": "application/json"},
        timeout_s=1.0,
        max_attempts=2,
        backoff=(1, 1.0, False),
        accept_sse=False,
        retry_enabled=True,
        retry_status_codes={429},
        retry_include_5xx=False,
    )
    assert isinstance(resp, DummyResponse)
    assert calls["n"] == 2


def test_post_json_raises_after_exhaust(monkeypatch):
    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        fp = io.BytesIO(b"oops")
        raise error.HTTPError(req.full_url, 500, "Server Error", hdrs=None, fp=fp)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ProviderError) as ei:
        post_json(
            base_url="https://example.com/v1",
            path="chat/completions",
            payload={"ok": True},
            headers={"Content-Type": "application/json"},
            timeout_s=0.1,
            max_attempts=1,
            backoff=(1, 1.0, False),
            accept_sse=False,
            retry_enabled=False,
        )
    assert "Request failed" not in str(ei.value)
