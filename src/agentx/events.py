from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List


Callback = Callable[[Dict[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[Callback]] = defaultdict(list)

    def subscribe(self, event: str, cb: Callback) -> None:
        self._subs[event].append(cb)

    def publish(self, event: str, payload: Dict[str, Any]) -> None:
        for cb in list(self._subs.get(event, [])):
            try:
                cb(payload)
            except Exception:
                # Keep bus resilient: ignore handler errors for now
                pass

