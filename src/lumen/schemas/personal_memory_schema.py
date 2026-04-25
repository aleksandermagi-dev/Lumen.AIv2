from __future__ import annotations

from lumen.schemas.migration import SchemaMigration


class PersonalMemorySchema:
    """Versioned schema helper for explicitly saved personal memory entries."""

    SCHEMA_TYPE = "personal_memory"
    SCHEMA_VERSION = "1"

    REQUIRED_FIELDS = {
        "schema_type",
        "schema_version",
        "session_id",
        "created_at",
        "title",
        "content",
        "source_interaction_prompt",
        "source_interaction_mode",
    }

    @staticmethod
    def _migrate_legacy(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_type"] = PersonalMemorySchema.SCHEMA_TYPE
        migrated["schema_version"] = PersonalMemorySchema.SCHEMA_VERSION
        return migrated

    MIGRATION = SchemaMigration(
        schema_type=SCHEMA_TYPE,
        current_version=SCHEMA_VERSION,
        migrations={"0": _migrate_legacy.__func__},
    )

    @classmethod
    def normalize(cls, payload: dict[str, object]) -> dict[str, object]:
        normalized = cls.MIGRATION.migrate(dict(payload))
        normalized.setdefault("schema_type", cls.SCHEMA_TYPE)
        normalized.setdefault("schema_version", cls.SCHEMA_VERSION)
        normalized.setdefault("normalized_topic", None)
        normalized.setdefault("memory_classification", {})
        normalized.setdefault("source_interaction_path", None)
        normalized.setdefault("client_surface", "main")
        normalized.setdefault("memory_origin", "user")
        normalized.setdefault("source_interaction_summary", None)
        return normalized

    @classmethod
    def validate(cls, payload: dict[str, object]) -> None:
        missing = sorted(field for field in cls.REQUIRED_FIELDS if field not in payload)
        if missing:
            raise ValueError(f"Personal memory entry is missing required fields: {', '.join(missing)}")
        if payload.get("schema_type") != cls.SCHEMA_TYPE:
            raise ValueError(f"Unexpected personal memory schema_type: {payload.get('schema_type')}")
        if str(payload.get("schema_version")) != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported personal memory schema_version: {payload.get('schema_version')}")
