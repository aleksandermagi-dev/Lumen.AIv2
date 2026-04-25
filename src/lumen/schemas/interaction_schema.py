from __future__ import annotations

from lumen.schemas.migration import SchemaMigration


class InteractionRecordSchema:
    """Versioned schema helper for persisted assistant interactions."""

    SCHEMA_TYPE = "interaction_record"
    SCHEMA_VERSION = "5"

    REQUIRED_FIELDS = {
        "schema_type",
        "schema_version",
        "session_id",
        "prompt",
        "mode",
        "kind",
        "summary",
        "created_at",
        "response",
    }

    @staticmethod
    def _migrate_legacy(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_type"] = InteractionRecordSchema.SCHEMA_TYPE
        migrated["schema_version"] = "1"
        return migrated

    @staticmethod
    def _migrate_v1_to_v2(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_version"] = InteractionRecordSchema.SCHEMA_VERSION
        migrated.setdefault(
            "memory_classification",
            {
                "candidate_type": "ephemeral_conversation_context",
                "classification_confidence": 0.0,
                "save_eligible": False,
                "requires_explicit_user_consent": False,
                "explicit_save_requested": False,
                "reason": "Legacy interaction records default to not saving when classification is unavailable.",
            },
        )
        return migrated

    @staticmethod
    def _migrate_v2_to_v3(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_version"] = InteractionRecordSchema.SCHEMA_VERSION
        migrated.setdefault(
            "memory_write_decision",
            {
                "action": "skip",
                "save_research_note": False,
                "save_personal_memory": False,
                "blocked_by_surface_policy": False,
                "reason": "Legacy interaction records do not include an explicit memory write decision.",
            },
        )
        return migrated

    @staticmethod
    def _migrate_v3_to_v4(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_version"] = "4"
        migrated.setdefault("trainability_trace", {})
        return migrated

    @staticmethod
    def _migrate_v4_to_v5(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_version"] = InteractionRecordSchema.SCHEMA_VERSION
        migrated.setdefault("supervised_support_trace", {})
        return migrated

    MIGRATION = SchemaMigration(
        schema_type=SCHEMA_TYPE,
        current_version=SCHEMA_VERSION,
        allow_newer_versions=True,
        migrations={
            "0": _migrate_legacy.__func__,
            "1": _migrate_v1_to_v2.__func__,
            "2": _migrate_v2_to_v3.__func__,
            "3": _migrate_v3_to_v4.__func__,
            "4": _migrate_v4_to_v5.__func__,
        },
    )

    @classmethod
    def normalize(cls, payload: dict[str, object]) -> dict[str, object]:
        normalized = cls.MIGRATION.migrate(dict(payload))
        normalized.setdefault("schema_type", cls.SCHEMA_TYPE)
        normalized.setdefault("schema_version", cls.SCHEMA_VERSION)
        normalized.setdefault("route", {})
        normalized.setdefault("context", {})
        normalized.setdefault("response", {})
        normalized.setdefault("resolved_prompt", None)
        normalized.setdefault("resolution_strategy", None)
        normalized.setdefault("resolution_reason", None)
        normalized.setdefault("confidence_posture", None)
        normalized.setdefault("route_status", None)
        normalized.setdefault("support_status", None)
        normalized.setdefault("tension_status", None)
        normalized.setdefault("tool_route_origin", None)
        normalized.setdefault("local_context_assessment", None)
        normalized.setdefault("coherence_topic", None)
        normalized.setdefault(
            "interaction_profile",
            {
                "interaction_style": "conversational",
                "reasoning_depth": "normal",
                "selection_source": "user",
                "confidence": None,
                "allow_suggestions": True,
            },
        )
        normalized.setdefault("profile_advice", None)
        normalized.setdefault(
            "memory_classification",
            {
                "candidate_type": "ephemeral_conversation_context",
                "classification_confidence": 0.0,
                "save_eligible": False,
                "requires_explicit_user_consent": False,
                "explicit_save_requested": False,
                "reason": "The interaction remains unsaved by default until memory classification says otherwise.",
            },
        )
        normalized.setdefault(
            "memory_write_decision",
            {
                "action": "skip",
                "save_research_note": False,
                "save_personal_memory": False,
                "blocked_by_surface_policy": False,
                "reason": "The interaction remains unsaved by default until memory write policy says otherwise.",
            },
        )
        normalized.setdefault("personal_memory", None)
        normalized.setdefault("research_note", None)
        normalized.setdefault("client_surface", "main")
        normalized.setdefault("pipeline_observability", {})
        normalized.setdefault("pipeline_trace", {})
        normalized.setdefault("trainability_trace", {})
        normalized.setdefault("supervised_support_trace", {})
        normalized.setdefault("detected_language", None)
        normalized.setdefault("normalized_topic", None)
        normalized.setdefault("dominant_intent", None)
        normalized.setdefault("extracted_entities", [])
        return normalized

    @classmethod
    def validate(cls, payload: dict[str, object]) -> None:
        missing = sorted(field for field in cls.REQUIRED_FIELDS if field not in payload)
        if missing:
            raise ValueError(f"Interaction record is missing required fields: {', '.join(missing)}")
        if payload.get("schema_type") != cls.SCHEMA_TYPE:
            raise ValueError(f"Unexpected interaction schema_type: {payload.get('schema_type')}")
        if not cls._supports_version(payload.get("schema_version")):
            raise ValueError(f"Unsupported interaction schema_version: {payload.get('schema_version')}")

    @classmethod
    def _supports_version(cls, value: object) -> bool:
        version = str(value or "").strip()
        if version == cls.SCHEMA_VERSION:
            return True
        if not version.isdigit() or not cls.SCHEMA_VERSION.isdigit():
            return False
        return int(version) >= int(cls.SCHEMA_VERSION)
