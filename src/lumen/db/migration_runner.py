from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
from typing import Any

from lumen.db.database_manager import DatabaseManager


MigrationFn = Callable[[Any], None]


class MigrationRunner:
    """Applies explicit ordered migrations to the unified persistence DB."""

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def ensure_metadata_tables(self) -> None:
        with self.database_manager.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS import_runs (
                    name TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL,
                    details_json TEXT
                );
                """
            )

    def has_migration(self, version: str) -> bool:
        self.ensure_metadata_tables()
        with self.database_manager.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?",
                (version,),
            ).fetchone()
        return row is not None

    def apply(self, version: str, name: str, migrate: MigrationFn) -> None:
        self.ensure_metadata_tables()
        if self.has_migration(version):
            return
        with self.database_manager.transaction() as conn:
            migrate(conn)
            conn.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (version, name, datetime.now(UTC).isoformat()),
            )

    def record_import_run(self, name: str, details: dict[str, object] | None = None) -> None:
        self.ensure_metadata_tables()
        payload = json.dumps(details or {}, ensure_ascii=True, sort_keys=True)
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO import_runs(name, completed_at, details_json)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    completed_at = excluded.completed_at,
                    details_json = excluded.details_json
                """,
                (name, datetime.now(UTC).isoformat(), payload),
            )

    def list_migrations(self) -> list[dict[str, object]]:
        self.ensure_metadata_tables()
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT version, name, applied_at
                FROM schema_migrations
                ORDER BY version ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_import_runs(self) -> list[dict[str, object]]:
        self.ensure_metadata_tables()
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT name, completed_at, details_json
                FROM import_runs
                ORDER BY name ASC
                """
            ).fetchall()
        payloads: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            try:
                item["details_json"] = json.loads(str(item.get("details_json") or "{}"))
            except json.JSONDecodeError:
                pass
            payloads.append(item)
        return payloads
