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


# Common event names (lightweight constants)
EV_TOKEN = "token"

# Config events
EV_CONFIG_RELOAD = "config.reload"         # { overrides?: dict }
EV_CONFIG_UPDATE = "config.update"         # { data: dict }
EV_CONFIG_UPDATED = "config.updated"       # { config: dict }

# Provider/model events
EV_PROVIDER_SET = "provider.set"           # { name: str }
EV_PROVIDER_UPDATED = "provider.updated"   # { name: str }
EV_MODEL_SET = "model.set"                 # { name: str }
EV_MODEL_UPDATED = "model.updated"         # { name: str }

# History/session events
EV_HISTORY_CLEAR = "history.clear"         # {}
EV_HISTORY_CLEARED = "history.cleared"     # {}
