from __future__ import annotations

import logging
from typing import List

from .config import Config
from .events import EventBus
from .providers import ChatMessage, ChatRequest, ChatResponse, Provider, ProviderFactory


class Agent:
    def __init__(self, cfg: Config, provider: Provider, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        self.provider = provider
        self.bus = bus or EventBus()
        self.history: List[ChatMessage] = []
        self.log = logging.getLogger("agent")
        self.in_flight: bool = False
        self._cancel_requested: bool = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def set_provider(self, provider: Provider) -> None:
        self.provider = provider

    def add_user_message(self, content: str) -> None:
        self.history.append(ChatMessage(role="user", content=content))

    def add_system_message(self, content: str) -> None:
        # Ensure system prompt is at the front so providers honor it
        self.history.insert(0, ChatMessage(role="system", content=content))

    def send(self, content: str, *, stream: bool | None = None) -> ChatResponse:
        self.add_user_message(content)
        req = ChatRequest(messages=self.history[:], model=self.cfg.model, stream=self.cfg.streaming if stream is None else stream)
        self._cancel_requested = False
        self.in_flight = True
        try:
            if req.stream:
                full: list[str] = []
                canceled = False
                gen = self.provider.stream(req)
                try:
                    for delta in gen:
                        if self.bus:
                            self.bus.publish("token", {"delta": delta.content, "done": delta.done})
                        if delta.content:
                            full.append(delta.content)
                        if self._cancel_requested:
                            canceled = True
                            try:
                                gen.close()
                            except Exception:
                                pass
                            break
                        if delta.done:
                            break
                finally:
                    pass
                content = "".join(full)
                resp = ChatResponse(content=content, model=self.cfg.model)
                if canceled:
                    # Do not record assistant message on cancel
                    return resp
            else:
                resp = self.provider.complete(req)
        finally:
            self.in_flight = False
        self.history.append(ChatMessage(role="assistant", content=resp.content))
        return resp

    @classmethod
    def from_config(cls, cfg: Config) -> "Agent":
        logger = logging.getLogger("provider")
        provider = ProviderFactory.create(cfg.provider, config=cfg, logger=logger)
        return cls(cfg, provider)
