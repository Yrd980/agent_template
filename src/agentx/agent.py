from __future__ import annotations

import logging
from typing import List

from .config import Config
from .events import EventBus, EV_SESSION_SAVE
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
        self.session_id: str | None = None
        self.session_name: str | None = None

    def cancel(self) -> None:
        self._cancel_requested = True

    def set_provider(self, provider: Provider) -> None:
        self.provider = provider

    def add_user_message(self, content: str) -> None:
        self.history.append(ChatMessage(role="user", content=content))
        self._enforce_history_limit()

    def add_system_message(self, content: str) -> None:
        # Ensure system prompt is at the front so providers honor it
        self.history.insert(0, ChatMessage(role="system", content=content))
        self._enforce_history_limit(system_keep=True)

    def _enforce_history_limit(self, system_keep: bool = True) -> None:
        limit = int(self.cfg.tui.history_limit or 0)
        if limit and len(self.history) > limit:
            if system_keep:
                # Keep first system if present
                sys_msgs = [m for m in self.history if m.role == "system"]
                first_system = sys_msgs[0] if sys_msgs else None
                # keep last (limit-1) non-system + optional system at start
                tail_needed = limit - (1 if first_system else 0)
                tail = [m for m in self.history if m.role != "system"][-max(tail_needed, 0):]
                self.history = ([first_system] if first_system else []) + tail
            else:
                self.history = self.history[-limit:]

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
                    self._autosave_if_enabled()
                    return resp
            else:
                resp = self.provider.complete(req)
        finally:
            self.in_flight = False
        self.history.append(ChatMessage(role="assistant", content=resp.content))
        self._enforce_history_limit()
        self._autosave_if_enabled()
        return resp

    def _autosave_if_enabled(self) -> None:
        if getattr(self.cfg, "session", None) and self.cfg.session.autosave and self.bus:
            try:
                self.bus.publish(EV_SESSION_SAVE, {})
            except Exception:
                pass

    @classmethod
    def from_config(cls, cfg: Config) -> "Agent":
        logger = logging.getLogger("provider")
        provider = ProviderFactory.create(cfg.provider, config=cfg, logger=logger)
        return cls(cfg, provider)
