from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


Role = str  # "system" | "user" | "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class ChatRequest:
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = True
    timeout_s: Optional[float] = None
    extra: Dict[str, Any] = None


@dataclass
class StreamDelta:
    content: str
    done: bool = False


@dataclass
class ChatResponse:
    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class Provider(abc.ABC):
    name: str = "base"

    def __init__(self, *, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger

    @abc.abstractmethod
    def complete(self, req: ChatRequest) -> ChatResponse:
        """Non-streaming completion."""

    @abc.abstractmethod
    def stream(self, req: ChatRequest) -> Iterable[StreamDelta]:
        """Streaming completion yielding deltas until done."""

