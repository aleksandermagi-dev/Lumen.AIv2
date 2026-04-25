from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from typing import Iterator

from lumen.app.settings import AppSettings


class DatabaseManager:
    """Central SQLite connection boundary for Lumen persistence."""

    SQLITE_BUSY_TIMEOUT_MS = 15000

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.db_path = settings.persistence_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.SQLITE_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.SQLITE_BUSY_TIMEOUT_MS}")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
