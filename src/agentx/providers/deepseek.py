from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterable, Optional

from .base import ChatRequest, ChatResponse, Provider, StreamDelta
from .registry import register_provider
from ..errors import ProviderError
from ..http import post_json


class DeepSeekProvider(Provider):
    name = "deepseek"

    def __init__(self, *, config, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(config=config, logger=logger or logging.getLogger("provider.deepseek"))
        self.log = logger or logging.getLogger("provider.deepseek")

    def _base_url(self) -> str:
        # Use configured base as-is; default includes /v1 to avoid 404s
        return self.config.endpoints.get("deepseek", "https://api.deepseek.com/v1")

    def _api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY") or self.config.keys.get("DEEPSEEK_API_KEY", "")

    def _timeout(self, req: ChatRequest) -> float:
        t = (req.timeout_s if req.timeout_s is not None else self.config.timeouts.read_ms / 1000.0)
        return max(1.0, float(t))

    def _build_payload(self, req: ChatRequest, stream: bool) -> Dict:
        model = (req.model or self.config.model or "deepseek-chat")
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        payload: Dict = {"model": model, "messages": messages, "stream": stream}
        if req.extra:
            payload.update(req.extra)
        return payload

    def _request(self, path: str, payload: Dict, timeout: float, *, stream: bool = False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key()}",
        }
        if not self._api_key():
            raise ProviderError("Missing DEEPSEEK_API_KEY")
        # Resolve retry policy with per-provider overrides
        retries = self.config.retries
        prov = retries.providers.get("deepseek", {}) if hasattr(retries, "providers") else {}
        retry_enabled = bool(prov.get("enabled", retries.enabled))
        status_codes = set(prov.get("status_codes", retries.status_codes)) if hasattr(retries, "status_codes") else None
        include_5xx = bool(prov.get("include_5xx", retries.include_5xx if hasattr(retries, "include_5xx") else True))

        resp = post_json(
            base_url=self._base_url(),
            path=path,
            payload=payload,
            headers=headers,
            timeout_s=timeout,
            max_attempts=self.config.retries.max_attempts,
            backoff=(self.config.retries.backoff.base_ms, self.config.retries.backoff.factor, self.config.retries.backoff.jitter),
            accept_sse=stream,
            retry_enabled=retry_enabled,
            retry_status_codes=status_codes,
            retry_include_5xx=include_5xx,
        )
        return resp

    def complete(self, req: ChatRequest) -> ChatResponse:
        payload = self._build_payload(req, stream=False)
        resp = self._request("chat/completions", payload, timeout=self._timeout(req))
        text = resp.read().decode("utf-8")
        data = json.loads(text)
        content = ""
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except Exception:
            self.log.debug("Unexpected response shape: %s", text)
        return ChatResponse(content=content, model=payload.get("model"))

    def stream(self, req: ChatRequest) -> Iterable[StreamDelta]:
        payload = self._build_payload(req, stream=True)
        resp = self._request("chat/completions", payload, timeout=self._timeout(req), stream=True)
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    yield StreamDelta(content="", done=True)
                    break
                try:
                    obj = json.loads(data_str)
                except Exception:
                    continue
                try:
                    delta = obj["choices"][0]["delta"].get("content", "")
                except Exception:
                    delta = ""
                if delta:
                    yield StreamDelta(content=delta, done=False)
        finally:
            try:
                resp.close()
            except Exception:
                pass


register_provider("deepseek", DeepSeekProvider)
