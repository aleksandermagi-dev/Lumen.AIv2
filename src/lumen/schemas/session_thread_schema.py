from __future__ import annotations

from lumen.schemas.migration import SchemaMigration


class SessionThreadSchema:
    """Versioned schema helper for persisted active thread state."""

    SCHEMA_TYPE = "session_thread_state"
    SCHEMA_VERSION = "1"
    REQUIRED_FIELDS = {
        "schema_type",
        "schema_version",
        "session_id",
        "mode",
        "kind",
        "prompt",
        "objective",
        "thread_summary",
        "summary",
        "updated_at",
    }

    @staticmethod
    def _migrate_legacy(payload: dict[str, object]) -> dict[str, object]:
        migrated = dict(payload)
        migrated["schema_type"] = SessionThreadSchema.SCHEMA_TYPE
        migrated["schema_version"] = SessionThreadSchema.SCHEMA_VERSION
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
        normalized.setdefault("original_prompt", None)
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
        normalized.setdefault("pipeline_observability", {})
        normalized.setdefault("pipeline_trace", {})
        normalized.setdefault("detected_language", None)
        normalized.setdefault("normalized_topic", None)
        normalized.setdefault("dominant_intent", None)
        normalized.setdefault("intent_domain", None)
        normalized.setdefault("intent_domain_confidence", None)
        normalized.setdefault("response_depth", None)
        normalized.setdefault("conversation_phase", None)
        normalized.setdefault("next_step_state", {})
        normalized.setdefault("tool_suggestion_state", {})
        normalized.setdefault("trainability_trace", {})
        normalized.setdefault("supervised_support_trace", {})
        normalized.setdefault("extracted_entities", [])
        normalized.setdefault("tool_context", {})
        normalized.setdefault("continuation_offer", {})
        normalized.setdefault("reasoning_state", {})
        return normalized

    @classmethod
    def validate(cls, payload: dict[str, object]) -> None:
        missing = sorted(field for field in cls.REQUIRED_FIELDS if field not in payload)
        if missing:
            raise ValueError(f"Session thread state is missing required fields: {', '.join(missing)}")
        if payload.get("schema_type") != cls.SCHEMA_TYPE:
            raise ValueError(f"Unexpected session thread schema_type: {payload.get('schema_type')}")
        if str(payload.get("schema_version")) != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported session thread schema_version: {payload.get('schema_version')}")
