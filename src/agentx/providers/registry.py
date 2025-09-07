from __future__ import annotations

from typing import Dict, Type
import importlib

from .base import Provider


registry: Dict[str, Type[Provider]] = {}


def register_provider(name: str, cls: Type[Provider]) -> None:
    registry[name] = cls


class ProviderFactory:
    @staticmethod
    def create(name: str, *, config, logger) -> Provider:
        if name not in registry:
            # Try dynamic import of provider module (agentx.providers.<name>)
            try:
                importlib.import_module(f"agentx.providers.{name}")
            except Exception:
                pass
        if name not in registry:
            raise ValueError(f"Provider '{name}' not registered")
        return registry[name](config=config, logger=logger)
