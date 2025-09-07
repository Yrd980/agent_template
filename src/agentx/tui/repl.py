from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..agent import Agent
from ..config import Config
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

    def _register_commands(self) -> None:
        self.register(Command("help", "Show help", lambda s, a: s._cmd_help()))
        self.register(Command("exit", "Exit", lambda s, a: setattr(s, "_stop", True)))
        self.register(Command("provider", "Get/Set provider", self._cmd_provider))
        self.register(Command("model", "Get/Set model", self._cmd_model))
        self.register(Command("history", "Show history length", self._cmd_history))
        self.register(Command("clear", "Clear history", self._cmd_clear))

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
        try:
            provider = ProviderFactory.create(name, config=self.agent.cfg, logger=logging.getLogger("provider"))
            self.agent.set_provider(provider)
            self.agent.cfg.provider = name
            print(f"provider set to {name}")
        except Exception as e:
            print(f"error: {e}")

    def _cmd_model(self, args: str) -> None:
        args = args.strip()
        if not args:
            print(f"model: {self.agent.cfg.model}")
        else:
            self.agent.cfg.model = shlex.split(args)[0]
            print(f"model set to {self.agent.cfg.model}")

    def _cmd_history(self, args: str) -> None:  # noqa: ARG002
        print(f"history messages: {len(self.agent.history)}")

    def _cmd_clear(self, args: str) -> None:  # noqa: ARG002
        self.agent.history.clear()
        print("history cleared")

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

