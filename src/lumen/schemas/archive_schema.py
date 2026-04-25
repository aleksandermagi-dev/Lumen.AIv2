from __future__ import annotations

from typing import Any

from lumen.schemas.migration import SchemaMigration


ARCHIVE_RECORD_SCHEMA_VERSION = "1"


def _migrate_legacy_archive_record(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated["schema_type"] = "archive_record"
    migrated["schema_version"] = ARCHIVE_RECORD_SCHEMA_VERSION
    return migrated


ARCHIVE_RECORD_MIGRATION = SchemaMigration(
    schema_type="archive_record",
    current_version=ARCHIVE_RECORD_SCHEMA_VERSION,
    migrations={
        "0": _migrate_legacy_archive_record,
    },
)


class ArchiveRecordSchema:
    """Validation helpers for versioned archive record payloads."""

    @staticmethod
    def normalize(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = ARCHIVE_RECORD_MIGRATION.migrate(dict(payload))
        normalized.setdefault("schema_type", "archive_record")
        normalized.setdefault("schema_version", ARCHIVE_RECORD_SCHEMA_VERSION)
        return normalized

    @staticmethod
    def validate(payload: dict[str, Any]) -> None:
        schema_type = payload.get("schema_type", "archive_record")
        schema_version = str(payload.get("schema_version", ARCHIVE_RECORD_SCHEMA_VERSION))
        if schema_type != "archive_record":
            raise ValueError(f"Unsupported archive schema_type '{schema_type}'")
        if schema_version != ARCHIVE_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported archive schema_version '{schema_version}'. "
                f"Expected '{ARCHIVE_RECORD_SCHEMA_VERSION}'."
            )
        for field in ("session_id", "tool_id", "capability", "status", "summary", "created_at"):
            if field not in payload:
                raise ValueError(f"Archive record is missing required field '{field}'")
