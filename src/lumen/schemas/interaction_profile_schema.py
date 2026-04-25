from __future__ import annotations

from lumen.schemas.migration import SchemaMigration


class InteractionProfileSchema:
    """Versioned schema helper for persisted session-level interaction profiles."""

    SCHEMA_TYPE = "interaction_profile"
    SCHEMA_VERSION = "1"
    REQUIRED_FIELDS = {
        "schema_type",
        "schema_version",
        "interaction_style",
        "reasoning_depth",
        "selection_source",
        "allow_suggestions",
    }

    @staticmethod
    def _migrate_legacy(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_type"] = InteractionProfileSchema.SCHEMA_TYPE
        migrated["schema_version"] = InteractionProfileSchema.SCHEMA_VERSION
        return migrated

    MIGRATION = SchemaMigration(
        schema_type=SCHEMA_TYPE,
        current_version=SCHEMA_VERSION,
        migrations={
            "0": _migrate_legacy.__func__,
        },
    )

    @classmethod
    def normalize(cls, payload: dict[str, object]) -> dict[str, object]:
        normalized = cls.MIGRATION.migrate(dict(payload))
        normalized.setdefault("schema_type", cls.SCHEMA_TYPE)
        normalized.setdefault("schema_version", cls.SCHEMA_VERSION)
        normalized["interaction_style"] = cls._normalize_interaction_style(normalized.get("interaction_style"))
        normalized.setdefault("reasoning_depth", "normal")
        normalized.setdefault("selection_source", "user")
        normalized.setdefault("confidence", None)
        normalized.setdefault("allow_suggestions", True)
        return normalized

    @staticmethod
    def _normalize_interaction_style(style: object) -> str:
        normalized = str(style or "collab").strip().lower()
        if normalized == "conversational":
            return "collab"
        if normalized in {"default", "collab", "direct"}:
            return normalized
        return "collab"

    @classmethod
    def validate(cls, payload: dict[str, object]) -> None:
        missing = sorted(field for field in cls.REQUIRED_FIELDS if field not in payload)
        if missing:
            raise ValueError(f"Interaction profile is missing required fields: {', '.join(missing)}")
        if payload.get("schema_type") != cls.SCHEMA_TYPE:
            raise ValueError(f"Unexpected interaction profile schema_type: {payload.get('schema_type')}")
        if str(payload.get("schema_version")) != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported interaction profile schema_version: {payload.get('schema_version')}"
            )
