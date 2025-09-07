from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from .providers.base import ChatMessage


@dataclass
class SessionMeta:
    id: str
    name: str
    created_at: float
    updated_at: float
    size: int


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "session"


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, sid: str, name: str) -> Path:
        return self.root / f"{sid}-{_slugify(name)}.json"

    def list(self) -> List[SessionMeta]:
        out: List[SessionMeta] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                out.append(SessionMeta(
                    id=data.get("id"),
                    name=data.get("name"),
                    created_at=float(data.get("created_at", 0.0)),
                    updated_at=float(data.get("updated_at", 0.0)),
                    size=len(data.get("messages", [])),
                ))
            except Exception:
                continue
        return out

    def _find(self, *, sid: Optional[str] = None, name: Optional[str] = None) -> Optional[Path]:
        if sid:
            matches = list(self.root.glob(f"{sid}-*.json"))
            if matches:
                return matches[0]
        if name:
            slug = _slugify(name)
            # pick latest updated among same slug
            candidates = sorted(self.root.glob(f"*-{slug}.json"))
            if candidates:
                return candidates[-1]
        return None

    def load(self, *, sid: Optional[str] = None, name: Optional[str] = None) -> tuple[SessionMeta, List[ChatMessage]]:
        p = self._find(sid=sid, name=name)
        if not p:
            raise FileNotFoundError("session not found")
        data = json.loads(p.read_text())
        meta = SessionMeta(
            id=data["id"],
            name=data["name"],
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            size=len(data.get("messages", [])),
        )
        msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in data.get("messages", [])]
        return meta, msgs

    def save(self, messages: Iterable[ChatMessage], *, sid: Optional[str], name: Optional[str]) -> SessionMeta:
        now = time.time()
        if not sid:
            sid = uuid.uuid4().hex[:12]
        if not name:
            name = f"Session {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
        msgs_list = list(messages)
        meta = SessionMeta(id=sid, name=name, created_at=now, updated_at=now, size=len(msgs_list))
        data = {
            "id": meta.id,
            "name": meta.name,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
            "messages": [asdict(m) for m in msgs_list],
        }
        path = self._path_for(meta.id, meta.name)
        path.write_text(json.dumps(data, indent=2))
        return meta

    def rename(self, sid: str, old_name: str, new_name: str) -> SessionMeta:
        p = self._find(sid=sid)
        if not p:
            raise FileNotFoundError("session not found")
        data = json.loads(p.read_text())
        data["name"] = new_name
        data["updated_at"] = time.time()
        new_meta = SessionMeta(id=data["id"], name=new_name, created_at=float(data.get("created_at", 0)), updated_at=float(data.get("updated_at", 0)), size=len(data.get("messages", [])))
        new_path = self._path_for(new_meta.id, new_meta.name)
        new_path.write_text(json.dumps(data, indent=2))
        if new_path != p:
            try:
                p.unlink()
            except Exception:
                pass
        return new_meta

