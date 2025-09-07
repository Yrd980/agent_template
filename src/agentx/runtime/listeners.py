from __future__ import annotations

import logging
from typing import Any, Dict

from ..agent import Agent
from ..config import Config
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
from ..providers import ProviderFactory


log = logging.getLogger("runtime")


def _safe_update_cfg(cfg: Config, data: Dict[str, Any]) -> Config:
    # Merge current config dict with overrides, then re-validate
    base = cfg.to_dict(redact_sensitive=False)
    base.update(data)
    new_cfg = Config.from_dict(base)
    new_cfg.validate()
    return new_cfg


def attach_default_listeners(agent: Agent) -> None:
    bus = agent.bus

    def on_provider_set(payload: Dict[str, Any]) -> None:
        name = str(payload.get("name", "")).strip()
        if not name:
            return
        try:
            provider = ProviderFactory.create(name, config=agent.cfg, logger=logging.getLogger("provider"))
            agent.set_provider(provider)
            agent.cfg.provider = name
            bus.publish(EV_PROVIDER_UPDATED, {"name": name})
        except Exception as e:
            log.error("provider.set failed: %s", e)

    def on_model_set(payload: Dict[str, Any]) -> None:
        name = str(payload.get("name", "")).strip()
        if not name:
            return
        agent.cfg.model = name
        bus.publish(EV_MODEL_UPDATED, {"name": name})

    def on_config_reload(payload: Dict[str, Any]) -> None:
        overrides = payload.get("overrides", {}) if isinstance(payload, dict) else {}
        try:
            new_cfg = Config.load(overrides=overrides)
        except Exception as e:
            log.error("config.reload failed: %s", e)
            return
        agent.cfg = new_cfg
        # Recreate provider according to potentially changed provider name
        try:
            provider = ProviderFactory.create(new_cfg.provider, config=new_cfg, logger=logging.getLogger("provider"))
            agent.set_provider(provider)
            bus.publish(EV_PROVIDER_UPDATED, {"name": new_cfg.provider})
        except Exception as e:
            log.error("provider recreation failed after reload: %s", e)
        bus.publish(EV_CONFIG_UPDATED, {"config": new_cfg.to_dict()})

    def on_config_update(payload: Dict[str, Any]) -> None:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        if not isinstance(data, dict):
            return
        try:
            new_cfg = _safe_update_cfg(agent.cfg, data)
        except Exception as e:
            log.error("config.update failed: %s", e)
            return
        agent.cfg = new_cfg
        # If provider changed in update, re-create it
        try:
            provider = ProviderFactory.create(new_cfg.provider, config=new_cfg, logger=logging.getLogger("provider"))
            agent.set_provider(provider)
            bus.publish(EV_PROVIDER_UPDATED, {"name": new_cfg.provider})
        except Exception as e:
            log.error("provider recreation failed after update: %s", e)
        bus.publish(EV_CONFIG_UPDATED, {"config": new_cfg.to_dict()})

    def on_history_clear(_: Dict[str, Any]) -> None:
        agent.history.clear()
        bus.publish(EV_HISTORY_CLEARED, {})

    bus.subscribe(EV_PROVIDER_SET, on_provider_set)
    bus.subscribe(EV_MODEL_SET, on_model_set)
    bus.subscribe(EV_CONFIG_RELOAD, on_config_reload)
    bus.subscribe(EV_CONFIG_UPDATE, on_config_update)
    bus.subscribe(EV_HISTORY_CLEAR, on_history_clear)

