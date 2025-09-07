from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..agent import Agent
from ..config import Config
from ..providers import ChatMessage
from ..events import (
    EV_CONFIG_RELOAD,
    EV_CONFIG_UPDATE,
    EV_CONFIG_UPDATED,
    EV_HISTORY_CLEAR,
    EV_HISTORY_CLEARED,
    EV_MODEL_SET,
    EV_MODEL_UPDATED,
    EV_PROVIDER_SET,
    EV_PROVIDER_UPDATED,
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


CommandHandler = Callable[["REPL", str], None]


@dataclass
class Command:
    name: str
    help: str
    handler: CommandHandler
    aliases: List[str] = field(default_factory=list)


class REPL:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.log = logging.getLogger("tui")
        self.commands: Dict[str, Command] = {}
        self._stream_started: bool = False
        self._register_commands()
        self._attach_bus_handlers()

    def _register_commands(self) -> None:
        self.register(Command("help", "Show help or /help <cmd>", lambda s, a: s._cmd_help(a), aliases=["h"]))
        self.register(Command("exit", "Exit", lambda s, a: setattr(s, "_stop", True), aliases=["quit", "q"]))
        self.register(Command("provider", "Get/Set provider", lambda s, a: s._cmd_provider(a), aliases=["p"]))
        self.register(Command("model", "Get/Set model", lambda s, a: s._cmd_model(a), aliases=["m"]))
        self.register(Command("history", "Show history length", lambda s, a: s._cmd_history(a)))
        self.register(Command("clear", "Clear history", lambda s, a: s._cmd_clear(a)))
        self.register(Command("reload", "Reload config (.env + json)", lambda s, a: s._cmd_reload(a)))
        self.register(Command("set", "Update config: /set key=value", lambda s, a: s._cmd_set(a)))
        self.register(Command("cancel", "Cancel current streaming reply (use Ctrl+C)", lambda s, a: s._cmd_cancel(a)))
        self.register(Command("status", "Show provider/model/streaming", lambda s, a: s._cmd_status(a)))
        self.register(Command("config", "Print current config (redacted)", lambda s, a: s._cmd_config(a)))
        self.register(Command("stream", "Get/Set streaming: /stream [on|off]", lambda s, a: s._cmd_stream(a)))
        self.register(Command("system", "Get/Set system prompt: /system [text|clear]", lambda s, a: s._cmd_system(a)))
        self.register(Command("autosave", "Get/Set autosave: /autosave [on|off]", lambda s, a: s._cmd_autosave(a)))
        self.register(Command("retry", "Get/Set retries: /retry [on|off]", lambda s, a: s._cmd_retry(a)))
        self.register(Command("save", "Save chat history to file: /save [path]", lambda s, a: s._cmd_save_history(a)))
        self.register(Command("load", "Load chat history from file: /load [path]", lambda s, a: s._cmd_load_history(a)))
        self.register(Command("sessions", "List saved sessions", lambda s, a: s._cmd_sessions(a)))
        self.register(Command("session", "Manage session: new/save/load/rename", lambda s, a: s._cmd_session(a)))

    def _attach_bus_handlers(self) -> None:
        bus = self.agent.bus

        def on_provider_updated(payload):
            name = payload.get("name")
            print(f"provider updated: {name}")

        def on_model_updated(payload):
            name = payload.get("name")
            print(f"model updated: {name}")

        def on_config_updated(payload):
            print("config reloaded/updated")

        def on_history_cleared(_payload):
            print("history cleared")

        def on_config_reload(_payload):
            print("reloading config…")

        def on_config_update(payload):
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            if isinstance(data, dict) and data:
                keys = ", ".join(sorted(data.keys()))
                print(f"updating config: {keys}")
            else:
                print("updating config…")

        def on_session_created(payload):
            print(f"session created: {payload.get('name')} ({payload.get('id')})")

        def on_session_saved(payload):
            print(f"session saved: {payload.get('name')} ({payload.get('id')})")

        def on_session_loaded(payload):
            print(f"session loaded: {payload.get('name')} (messages: {payload.get('size')})")

        def on_session_listed(payload):
            items = payload.get("sessions", [])
            if not items:
                print("no sessions")
                return
            print("sessions:")
            for it in items:
                print(f"  {it['id']}  {it['name']}  ({it['size']})")

        def on_session_renamed(payload):
            print(f"session renamed: {payload.get('name')} ({payload.get('id')})")

        bus.subscribe(EV_PROVIDER_UPDATED, on_provider_updated)
        bus.subscribe(EV_MODEL_UPDATED, on_model_updated)
        bus.subscribe(EV_CONFIG_UPDATED, on_config_updated)
        bus.subscribe(EV_HISTORY_CLEARED, on_history_cleared)
        bus.subscribe(EV_CONFIG_RELOAD, on_config_reload)
        bus.subscribe(EV_CONFIG_UPDATE, on_config_update)
        bus.subscribe(EV_SESSION_CREATED, on_session_created)
        bus.subscribe(EV_SESSION_SAVED, on_session_saved)
        bus.subscribe(EV_SESSION_LOADED, on_session_loaded)
        bus.subscribe(EV_SESSION_LISTED, on_session_listed)
        bus.subscribe(EV_SESSION_RENAMED, on_session_renamed)

        # Live token streaming with prefix
        def on_token(payload):
            delta = payload.get("delta", "")
            done = bool(payload.get("done", False))
            if delta:
                if not self._stream_started:
                    print("assistant: ", end="", flush=True)
                    self._stream_started = True
                print(delta, end="", flush=True)
            if done:
                print()
                self._stream_started = False

        bus.subscribe("token", on_token)

    def register(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd
        for a in cmd.aliases:
            self.commands[a] = cmd

    def _cmd_help(self, arg: str = "") -> None:
        arg = arg.strip()
        if arg and arg in self.commands:
            cmd = self.commands[arg]
            alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            print(f"/{cmd.name}{alias_str} - {cmd.help}")
            return
        print("Commands:")
        seen = set()
        for key in sorted(self.commands):
            cmd = self.commands[key]
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            print(f"  /{cmd.name:<9} - {cmd.help}")

    def _cmd_provider(self, args: str) -> None:
        args = args.strip()
        if not args:
            print(f"provider: {self.agent.cfg.provider}")
            return
        name = shlex.split(args)[0]
        self.agent.bus.publish(EV_PROVIDER_SET, {"name": name})

    def _cmd_model(self, args: str) -> None:
        args = args.strip()
        if not args:
            print(f"model: {self.agent.cfg.model}")
        else:
            name = shlex.split(args)[0]
            self.agent.bus.publish(EV_MODEL_SET, {"name": name})

    def _cmd_history(self, args: str) -> None:  # noqa: ARG002
        print(f"history messages: {len(self.agent.history)}")

    def _cmd_clear(self, args: str) -> None:  # noqa: ARG002
        self.agent.bus.publish(EV_HISTORY_CLEAR, {})

    def _cmd_reload(self, args: str) -> None:  # noqa: ARG002
        # Reload config from files (+ env)
        self.agent.bus.publish(EV_CONFIG_RELOAD, {"overrides": {}})

    def _cmd_set(self, args: str) -> None:
        # Update config fields on the fly, e.g., /set timeouts.read_ms=45000
        args = args.strip()
        if not args:
            print("usage: /set key=value [key=value ...]")
            return
        updates: Dict[str, object] = {}
        parts = shlex.split(args)
        for p in parts:
            if "=" not in p:
                print(f"skip invalid: {p}")
                continue
            k, v = p.split("=", 1)
            # Try to cast numbers and booleans
            val: object
            if v.lower() in {"true", "false"}:
                val = v.lower() == "true"
            else:
                try:
                    val = int(v)
                except ValueError:
                    try:
                        val = float(v)
                    except ValueError:
                        val = v
            updates[k] = val
        if not updates:
            return
        # Persist to config.json (project-level preferred), then reload
        changed_path = self._update_config_file(updates)
        if changed_path:
            print(f"wrote {changed_path}")
        # Publish update for immediate in-memory effect, then reload to resync
        self.agent.bus.publish(EV_CONFIG_UPDATE, {"data": updates})
        self.agent.bus.publish(EV_CONFIG_RELOAD, {"overrides": {}})

    def _update_config_file(self, updates: Dict[str, object]) -> Optional[str]:
        import json
        from pathlib import Path

        paths = Config.default_paths()
        target = paths["project_json"] if paths["project_json"].exists() else paths["user_json"]
        if not target.exists():
            # Create project config.json by default
            target = paths["project_json"]
            target.parent.mkdir(parents=True, exist_ok=True)
            base: Dict[str, object] = {}
        else:
            try:
                base = json.loads(target.read_text())
            except Exception:
                base = {}

        def set_path(d: Dict[str, object], path: str, value: object) -> None:
            parts = path.split(".")
            cur = d
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                if is_last:
                    cur[part] = value
                else:
                    nxt = cur.get(part)
                    if not isinstance(nxt, dict):
                        nxt = {}
                        cur[part] = nxt
                    cur = nxt  # type: ignore[assignment]

        for k, v in updates.items():
            set_path(base, k, v)

        target.write_text(json.dumps(base, indent=2))
        return str(target)

    def _cmd_cancel(self, args: str) -> None:  # noqa: ARG002
        if getattr(self.agent, "in_flight", False):
            self.agent.cancel()
            print("cancel requested")
        else:
            print("No active reply to cancel.")

    def _cmd_status(self, args: str) -> None:  # noqa: ARG002
        autosave = getattr(self.agent.cfg, "session", None) and self.agent.cfg.session.autosave
        retries = getattr(self.agent.cfg, "retries", None) and self.agent.cfg.retries.enabled
        print(f"provider: {self.agent.cfg.provider}; model: {self.agent.cfg.model}; streaming: {self.agent.cfg.streaming}; autosave: {bool(autosave)}; retries: {bool(retries)}")

    def _cmd_config(self, args: str) -> None:  # noqa: ARG002
        d = self.agent.cfg.to_dict(redact_sensitive=True)
        import json
        print(json.dumps(d, indent=2))

    def _cmd_stream(self, args: str) -> None:
        a = args.strip().lower()
        if not a:
            print(f"streaming: {self.agent.cfg.streaming}")
            return
        if a in {"on", "true", "1"}:
            self.agent.cfg.streaming = True
        elif a in {"off", "false", "0"}:
            self.agent.cfg.streaming = False
        else:
            print("usage: /stream [on|off]")
            return
        print(f"streaming set to {self.agent.cfg.streaming}")

    def _cmd_autosave(self, args: str) -> None:
        a = args.strip().lower()
        if not a:
            val = getattr(self.agent.cfg, "session", None) and self.agent.cfg.session.autosave
            print(f"autosave: {bool(val)}")
            return
        if a in {"on", "true", "1"}:
            self.agent.cfg.session.autosave = True
        elif a in {"off", "false", "0"}:
            self.agent.cfg.session.autosave = False
        else:
            print("usage: /autosave [on|off]")
            return
        print(f"autosave set to {self.agent.cfg.session.autosave}")

    def _cmd_retry(self, args: str) -> None:
        a = args.strip().lower()
        if not a:
            val = getattr(self.agent.cfg, "retries", None) and self.agent.cfg.retries.enabled
            print(f"retries: {bool(val)}")
            return
        if a in {"on", "true", "1"}:
            self.agent.cfg.retries.enabled = True
            self._update_config_file({"retries.enabled": True})
        elif a in {"off", "false", "0"}:
            self.agent.cfg.retries.enabled = False
            self._update_config_file({"retries.enabled": False})
        else:
            print("usage: /retry [on|off]")
            return
        print(f"retries set to {self.agent.cfg.retries.enabled}")

    def _cmd_system(self, args: str) -> None:
        s = args.strip()
        if not s:
            last = next((m.content for m in reversed(self.agent.history) if m.role == "system"), None)
            print(f"system: {last!r}")
            return
        if s.lower() == "clear":
            self.agent.history = [m for m in self.agent.history if m.role != "system"]
            print("system prompts cleared")
            return
        # Replace existing system prompts with a single one
        self.agent.history = [m for m in self.agent.history if m.role != "system"]
        self.agent.add_system_message(s)
        print("system prompt set")

    def _cmd_save_history(self, args: str) -> None:
        import json
        from pathlib import Path
        path = args.strip() or "history.json"
        p = Path(path)
        msgs = [
            {"role": m.role, "content": m.content}
            for m in self.agent.history
        ]
        p.write_text(json.dumps({"messages": msgs}, indent=2))
        print(f"saved history to {p}")

    def _cmd_load_history(self, args: str) -> None:
        import json
        from pathlib import Path
        path = args.strip() or "history.json"
        p = Path(path)
        if not p.exists():
            print(f"file not found: {p}")
            return
        try:
            data = json.loads(p.read_text())
            msgs = data.get("messages")
            if isinstance(msgs, list):
                sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
                non_sys = [m for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
                ordered = (sys_msgs[:1] + non_sys) if sys_msgs else non_sys
                new_history: List[ChatMessage] = []
                for m in ordered:
                    role = str(m.get("role", "user"))
                    content = str(m.get("content", ""))
                    if role not in ("system", "user", "assistant"):
                        continue
                    new_history.append(ChatMessage(role=role, content=content))
                self.agent.history = new_history
                print(f"loaded {len(new_history)} messages from {p}")
        except Exception as e:
            print(f"error loading {p}: {e}")

    def _cmd_sessions(self, args: str) -> None:  # noqa: ARG002
        self.agent.bus.publish(EV_SESSION_LIST, {})

    def _cmd_session(self, args: str) -> None:
        parts = shlex.split(args)
        if not parts:
            print("usage: /session [new|save|load|rename] [args]")
            return
        op = parts[0]
        rest = parts[1:]
        if op == "new":
            name = " ".join(rest) if rest else None
            self.agent.bus.publish(EV_SESSION_NEW, {"name": name} if name else {})
        elif op == "save":
            name = " ".join(rest) if rest else None
            self.agent.bus.publish(EV_SESSION_SAVE, {"name": name} if name else {})
        elif op == "load":
            if not rest:
                print("usage: /session load <id|name>")
                return
            target = " ".join(rest)
            # heuristic: if looks like id prefix (hex), pass as id; else as name
            if all(c in "0123456789abcdef" for c in target.lower()) and len(target) >= 6:
                self.agent.bus.publish(EV_SESSION_LOAD, {"id": target})
            else:
                self.agent.bus.publish(EV_SESSION_LOAD, {"name": target})
        elif op == "rename":
            if not rest:
                print("usage: /session rename <new name>")
                return
            name = " ".join(rest)
            self.agent.bus.publish(EV_SESSION_RENAME, {"name": name})
        else:
            print("usage: /session [new|save|load|rename] [args]")

    def run(self) -> None:
        self._stop = False  # type: ignore[attr-defined]
        print("AgentX REPL. Type /help for commands. Ctrl+C to exit.")
        while not getattr(self, "_stop"):
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                parts = line[1:].split(None, 1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                if cmd in self.commands:
                    try:
                        self.commands[cmd].handler(self, args)
                    except Exception as e:
                        print(f"error: {e}")
                else:
                    print(f"unknown command: {cmd}")
                continue

            # Normal message
            try:
                if self.agent.cfg.streaming:
                    # Streaming handled via token events
                    self.agent.send(line, stream=True)
                else:
                    resp = self.agent.send(line, stream=False)
                    print(f"assistant: {resp.content}")
            except KeyboardInterrupt:
                # Convert Ctrl+C into an agent cancel and user feedback
                self.agent.cancel()
                print("\n[canceled]")
            except Exception as e:
                print(f"error: {e}")
