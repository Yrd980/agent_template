from __future__ import annotations

import logging
import os
import re
from logging import Handler, LogRecord
from logging.handlers import RotatingFileHandler
from typing import Optional


SENSITIVE_KEYS = [
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "LLAMA_API_KEY",
    "ANTHROPIC_API_KEY",
]


class RedactFilter(logging.Filter):
    SECRET_REGEX = re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*([^\s,]+)")

    def filter(self, record: LogRecord) -> bool:
        msg = str(record.getMessage())
        # Drop noisy messages
        drop_terms = ("heartbeat", "keepalive", "stream_keepalive")
        if any(t in msg.lower() for t in drop_terms):
            return False

        # Apply redaction
        redacted = msg
        redacted = self.SECRET_REGEX.sub(r"\1: ***", redacted)
        for k in SENSITIVE_KEYS:
            v = os.getenv(k)
            if v:
                redacted = redacted.replace(v, "***")
        record.msg = redacted
        return True


def _make_handler(to_file: Optional[str]) -> Handler:
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if to_file:
        handler = RotatingFileHandler(to_file, maxBytes=1_000_000, backupCount=3)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.addFilter(RedactFilter())
    return handler


def setup_logging(level: str = "INFO", component_filter: Optional[str] = None, file: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = _make_handler(file)
    root.addHandler(handler)

    if component_filter:
        class ComponentFilter(logging.Filter):
            def filter(self, record: LogRecord) -> bool:
                return record.name.startswith(component_filter)

        root.addFilter(ComponentFilter())

