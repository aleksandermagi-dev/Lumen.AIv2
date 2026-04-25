from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumen.db.persistence_manager import PersistenceManager

__all__ = ["PersistenceManager"]


def __getattr__(name: str):
    if name == "PersistenceManager":
        from lumen.db.persistence_manager import PersistenceManager

        return PersistenceManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
