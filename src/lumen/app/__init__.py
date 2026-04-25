"""Application-layer entrypoints for Lumen."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AppController",
    "AppSettings",
    "ArchivedRunRecord",
    "CommandParser",
    "ConfigLoader",
    "SessionState",
]


def __getattr__(name: str):
    module_map = {
        "AppController": "lumen.app.controller",
        "AppSettings": "lumen.app.settings",
        "ArchivedRunRecord": "lumen.app.models",
        "CommandParser": "lumen.app.command_parser",
        "ConfigLoader": "lumen.app.config_loader",
        "SessionState": "lumen.app.models",
    }
    module_name = module_map.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
