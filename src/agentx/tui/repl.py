from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..agent import Agent
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

        bus.subscribe(EV_PROVIDER_UPDATED, on_provider_updated)
        bus.subscribe(EV_MODEL_UPDATED, on_model_updated)
        bus.subscribe(EV_CONFIG_UPDATED, on_config_updated)
        bus.subscribe(EV_HISTORY_CLEARED, on_history_cleared)
        bus.subscribe(EV_CONFIG_RELOAD, on_config_reload)
        bus.subscribe(EV_CONFIG_UPDATE, on_config_update)

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
        if updates:
            self.agent.bus.publish(EV_CONFIG_UPDATE, {"data": updates})

    def _cmd_cancel(self, args: str) -> None:  # noqa: ARG002
        if getattr(self.agent, "in_flight", False):
            self.agent.cancel()
            print("cancel requested")
        else:
            print("No active reply to cancel.")

    def _cmd_status(self, args: str) -> None:  # noqa: ARG002
        print(f"provider: {self.agent.cfg.provider}; model: {self.agent.cfg.model}; streaming: {self.agent.cfg.streaming}")

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
