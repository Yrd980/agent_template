from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PROVIDERS = {"openai", "ollama", "deepseek", "qwen", "llama"}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Do not overwrite already-set env vars
        os.environ.setdefault(key, val)


@dataclass
class BackoffConfig:
    base_ms: int = 200
    factor: float = 2.0
    jitter: bool = True


@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff: BackoffConfig = field(default_factory=BackoffConfig)


@dataclass
class TimeoutConfig:
    connect_ms: int = 10_000
    read_ms: int = 60_000


@dataclass
class TuiConfig:
    history_limit: int = 200
    show_timestamps: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    filter: Optional[str] = None
    file: Optional[str] = None


@dataclass
class SessionConfig:
    autosave: bool = False


@dataclass
class Config:
    provider: str = "openai"
    model: Optional[str] = None
    streaming: bool = True
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    endpoints: Dict[str, str] = field(default_factory=dict)
    keys: Dict[str, str] = field(default_factory=dict)
    tui: TuiConfig = field(default_factory=TuiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    @staticmethod
    def default_paths() -> Dict[str, Path]:
        cwd = Path.cwd()
        home = Path.home()
        return {
            "dotenv": cwd / ".env",
            "project_json": cwd / "config.json",
            "user_json": home / ".agent" / "config.json",
        }

    @classmethod
    def load(cls, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        paths = cls.default_paths()

        # 1) Load .env first (non-destructive to existing env)
        _load_dotenv(paths["dotenv"])

        # 2) Load JSON config (project then user-level)
        data: Dict[str, Any] = {}
        if paths["project_json"].exists():
            try:
                data = json.loads(paths["project_json"].read_text())
            except Exception:
                raise ValueError("Failed to parse project config.json")
        elif paths["user_json"].exists():
            try:
                data = json.loads(paths["user_json"].read_text())
            except Exception:
                raise ValueError("Failed to parse user ~/.agent/config.json")

        # 3) Env overrides for top-level common fields
        env_overrides: Dict[str, Any] = {}
        if os.getenv("AGENT_PROVIDER"):
            env_overrides["provider"] = os.getenv("AGENT_PROVIDER")
        if os.getenv("AGENT_MODEL"):
            env_overrides["model"] = os.getenv("AGENT_MODEL")
        if os.getenv("AGENT_STREAMING"):
            v = os.getenv("AGENT_STREAMING", "true").lower()
            env_overrides["streaming"] = v in {"1", "true", "yes", "on"}

        # Provider keys via env
        keys = data.get("keys", {}).copy()
        for k in [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "QWEN_API_KEY",
            "LLAMA_API_KEY",
            "ANTHROPIC_API_KEY",  # future-friendly
        ]:
            if os.getenv(k):
                keys[k] = os.getenv(k, "")
        if keys:
            data["keys"] = keys

        # Merge order: file -> env overrides -> explicit overrides (CLI)
        merged = {**data, **env_overrides, **(overrides or {})}
        cfg = cls.from_dict(merged)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
            return d.get(key, default)

        timeouts = _get(data, "timeouts", {})
        retries = _get(data, "retries", {})
        backoff = _get(retries, "backoff", {})
        tui = _get(data, "tui", {})
        logging_cfg = _get(data, "logging", {})
        session_cfg = _get(data, "session", {})

        return cls(
            provider=_get(data, "provider", "openai"),
            model=_get(data, "model", None),
            streaming=bool(_get(data, "streaming", True)),
            timeouts=TimeoutConfig(
                connect_ms=int(_get(timeouts, "connect_ms", 10_000)),
                read_ms=int(_get(timeouts, "read_ms", 60_000)),
            ),
            retries=RetryConfig(
                max_attempts=int(_get(retries, "max_attempts", 3)),
                backoff=BackoffConfig(
                    base_ms=int(_get(backoff, "base_ms", 200)),
                    factor=float(_get(backoff, "factor", 2.0)),
                    jitter=bool(_get(backoff, "jitter", True)),
                ),
            ),
            endpoints=dict(_get(data, "endpoints", {})),
            keys=dict(_get(data, "keys", {})),
            tui=TuiConfig(
                history_limit=int(_get(tui, "history_limit", 200)),
                show_timestamps=bool(_get(tui, "show_timestamps", False)),
            ),
            logging=LoggingConfig(
                level=str(_get(logging_cfg, "level", "INFO")).upper(),
                filter=_get(logging_cfg, "filter", None),
                file=_get(logging_cfg, "file", None),
            ),
            session=SessionConfig(
                autosave=bool(_get(session_cfg, "autosave", False)),
            ),
        )

    def validate(self) -> None:
        if self.provider not in DEFAULT_PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Allowed: {sorted(DEFAULT_PROVIDERS)}")
        if self.retries.max_attempts < 0:
            raise ValueError("retries.max_attempts must be >= 0")
        if self.timeouts.connect_ms <= 0 or self.timeouts.read_ms <= 0:
            raise ValueError("timeouts must be positive")

    def to_dict(self, redact_sensitive: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if redact_sensitive and d.get("keys"):
            redacted = {}
            for k, v in d["keys"].items():
                if not v:
                    redacted[k] = v
                else:
                    redacted[k] = v[:4] + "…" if len(v) > 8 else "***"
            d["keys"] = redacted
        return d
