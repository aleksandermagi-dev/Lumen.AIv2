from __future__ import annotations

from lumen.schemas.migration import SchemaMigration


class ResearchArtifactSchema:
    """Versioned schema helper for promoted research artifacts."""

    SCHEMA_TYPE = "research_artifact"
    SCHEMA_VERSION = "1"

    REQUIRED_FIELDS = {
        "schema_type",
        "schema_version",
        "session_id",
        "created_at",
        "artifact_type",
        "title",
        "content",
        "source_note_path",
    }

    @staticmethod
    def _migrate_legacy(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_type"] = ResearchArtifactSchema.SCHEMA_TYPE
        migrated["schema_version"] = ResearchArtifactSchema.SCHEMA_VERSION
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
        normalized.setdefault("source_interaction_prompt", None)
        normalized.setdefault("source_interaction_path", None)
        normalized.setdefault("promotion_reason", None)
        return normalized

    @classmethod
    def validate(cls, payload: dict[str, object]) -> None:
        missing = sorted(field for field in cls.REQUIRED_FIELDS if field not in payload)
        if missing:
            raise ValueError(f"Research artifact is missing required fields: {', '.join(missing)}")
        if payload.get("schema_type") != cls.SCHEMA_TYPE:
            raise ValueError(f"Unexpected research artifact schema_type: {payload.get('schema_type')}")
        if str(payload.get("schema_version")) != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported research artifact schema_version: {payload.get('schema_version')}")
