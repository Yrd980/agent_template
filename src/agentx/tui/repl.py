from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..agent import Agent
from ..events import (
    EV_CONFIG_RELOAD,
    EV_CONFIG_UPDATE,
    EV_HISTORY_CLEAR,
    EV_HISTORY_CLEARED,
    EV_MODEL_SET,
    EV_MODEL_UPDATED,
    EV_PROVIDER_SET,
    EV_PROVIDER_UPDATED,
)
from ..providers import ProviderFactory


CommandHandler = Callable[["REPL", str], None]


@dataclass
class Command:
    name: str
    help: str
    handler: CommandHandler


class REPL:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.log = logging.getLogger("tui")
        self.commands: Dict[str, Command] = {}
        self._register_commands()
        self._attach_bus_handlers()

    def _register_commands(self) -> None:
        self.register(Command("help", "Show help", lambda s, a: s._cmd_help()))
        self.register(Command("exit", "Exit", lambda s, a: setattr(s, "_stop", True)))
        self.register(Command("provider", "Get/Set provider", self._cmd_provider))
        self.register(Command("model", "Get/Set model", self._cmd_model))
        self.register(Command("history", "Show history length", self._cmd_history))
        self.register(Command("clear", "Clear history", self._cmd_clear))
        self.register(Command("reload", "Reload config (.env + json)", self._cmd_reload))
        self.register(Command("set", "Update config: /set key=value", self._cmd_set))

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

        bus.subscribe(EV_PROVIDER_UPDATED, on_provider_updated)
        bus.subscribe(EV_MODEL_UPDATED, on_model_updated)
        bus.subscribe(EV_CONFIG_UPDATED, on_config_updated)
        bus.subscribe(EV_HISTORY_CLEARED, on_history_cleared)

    def register(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd

    def _cmd_help(self) -> None:
        print("Commands:")
        for name in sorted(self.commands):
            print(f"  /{name:<9} - {self.commands[name].help}")

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
            resp = self.agent.send(line, stream=True)
            print(resp.content)
