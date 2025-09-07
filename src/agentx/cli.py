from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .agent import Agent
from .config import Config
from .logging import setup_logging
from .tui.repl import REPL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentx", description="AgentX CLI")
    p.add_argument("--provider", help="Provider name (openai, ollama, deepseek, qwen, llama)")
    p.add_argument("--model", help="Model name override")
    p.add_argument("--log", help="Log level", default=None)
    p.add_argument("--log-file", help="Log file path", default=None)
    p.add_argument("--config", help="Path to config.json", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Optionally load from provided config path by temporarily changing cwd
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            # Read directly
            import json
            data = json.loads(cfg_path.read_text())
            cfg = Config.from_dict(data)
        else:
            raise SystemExit(f"config file not found: {args.config}")
    else:
        overrides = {}
        if args.provider:
            overrides["provider"] = args.provider
        if args.model:
            overrides["model"] = args.model
        cfg = Config.load(overrides=overrides)

    setup_logging(level=args.log or cfg.logging.level, component_filter=cfg.logging.filter, file=args.log_file or cfg.logging.file)

    logging.getLogger("agentx").info("Starting AgentX", extra={})

    agent = Agent.from_config(cfg)
    repl = REPL(agent)
    repl.run()


if __name__ == "__main__":
    main()

