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

    def set_provider(self, provider: Provider) -> None:
        self.provider = provider

    def add_user_message(self, content: str) -> None:
        self.history.append(ChatMessage(role="user", content=content))

    def add_system_message(self, content: str) -> None:
        self.history.append(ChatMessage(role="system", content=content))

    def send(self, content: str, *, stream: bool | None = None) -> ChatResponse:
        self.add_user_message(content)
        req = ChatRequest(messages=self.history[:], model=self.cfg.model, stream=self.cfg.streaming if stream is None else stream)
        if req.stream:
            full = []
            for delta in self.provider.stream(req):
                if self.bus:
                    self.bus.publish("token", {"delta": delta.content, "done": delta.done})
                if delta.content:
                    full.append(delta.content)
                if delta.done:
                    break
            content = "".join(full)
            resp = ChatResponse(content=content, model=self.cfg.model)
        else:
            resp = self.provider.complete(req)
        self.history.append(ChatMessage(role="assistant", content=resp.content))
        return resp

    @classmethod
    def from_config(cls, cfg: Config) -> "Agent":
        logger = logging.getLogger("provider")
        provider = ProviderFactory.create(cfg.provider, config=cfg, logger=logger)
        return cls(cfg, provider)

