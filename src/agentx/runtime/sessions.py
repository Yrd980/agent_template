from __future__ import annotations

import logging
from typing import Any, Dict

from ..agent import Agent
from ..events import (
    EV_SESSION_CREATED,
    EV_SESSION_LIST,
    EV_SESSION_LISTED,
    EV_SESSION_LOAD,
    EV_SESSION_LOADED,
    EV_SESSION_NEW,
    EV_SESSION_RENAME,
    EV_SESSION_RENAMED,
    EV_SESSION_SAVE,
    EV_SESSION_SAVED,
)
from ..session import SessionStore


log = logging.getLogger("runtime.sessions")


def attach_session_listeners(agent: Agent, store: SessionStore) -> None:
    bus = agent.bus

    def on_new(payload: Dict[str, Any]) -> None:
        name = (payload or {}).get("name") or "New Session"
        agent.history.clear()
        meta = store.save(agent.history, sid=None, name=name)
        agent.session_id = meta.id
        agent.session_name = meta.name
        bus.publish(EV_SESSION_CREATED, {"id": meta.id, "name": meta.name})

    def on_save(payload: Dict[str, Any]) -> None:
        name = (payload or {}).get("name") or agent.session_name
        sid = agent.session_id
        meta = store.save(agent.history, sid=sid, name=name)
        agent.session_id = meta.id
        agent.session_name = meta.name
        bus.publish(EV_SESSION_SAVED, {"id": meta.id, "name": meta.name})

    def on_load(payload: Dict[str, Any]) -> None:
        sid = (payload or {}).get("id")
        name = (payload or {}).get("name")
        try:
            meta, msgs = store.load(sid=sid, name=name)
        except Exception as e:
            log.error("load failed: %s", e)
            return
        agent.history = msgs
        agent.session_id = meta.id
        agent.session_name = meta.name
        bus.publish(EV_SESSION_LOADED, {"id": meta.id, "name": meta.name, "size": meta.size})

    def on_list(_payload: Dict[str, Any]) -> None:
        metas = store.list()
        items = [{"id": m.id, "name": m.name, "updated_at": m.updated_at, "size": m.size} for m in metas]
        bus.publish(EV_SESSION_LISTED, {"sessions": items})

    def on_rename(payload: Dict[str, Any]) -> None:
        new_name = (payload or {}).get("name")
        if not new_name:
            return
        sid = agent.session_id
        if not sid or agent.session_name is None:
            return
        try:
            meta = store.rename(sid, agent.session_name, new_name)
        except Exception as e:
            log.error("rename failed: %s", e)
            return
        agent.session_name = meta.name
        bus.publish(EV_SESSION_RENAMED, {"id": meta.id, "name": meta.name})

    bus.subscribe(EV_SESSION_NEW, on_new)
    bus.subscribe(EV_SESSION_SAVE, on_save)
    bus.subscribe(EV_SESSION_LOAD, on_load)
    bus.subscribe(EV_SESSION_LIST, on_list)
    bus.subscribe(EV_SESSION_RENAME, on_rename)

