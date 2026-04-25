from __future__ import annotations

from typing import Any


BUNDLE_MANIFEST_SCHEMA_VERSION = "1"


class BundleManifestSchema:
    """Validation helpers for versioned bundle manifest payloads."""

    @staticmethod
    def normalize(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.setdefault("schema_version", BUNDLE_MANIFEST_SCHEMA_VERSION)
        return normalized

    @staticmethod
    def validate(payload: dict[str, Any]) -> None:
        schema_version = str(payload.get("schema_version", BUNDLE_MANIFEST_SCHEMA_VERSION))
        if schema_version != BUNDLE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported bundle manifest schema_version '{schema_version}'. "
                f"Expected '{BUNDLE_MANIFEST_SCHEMA_VERSION}'."
            )
        for field in ("id", "name", "entrypoint"):
            if field not in payload or not payload[field]:
                raise ValueError(f"Bundle manifest is missing required field '{field}'")
        for capability in payload.get("capabilities", []):
            for list_field in ("trigger_keywords", "structural_patterns", "intent_hints", "command_aliases"):
                value = capability.get(list_field)
                if value is None:
                    continue
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError(
                        f"Capability field '{list_field}' must be a list of strings"
                    )
            priority = capability.get("routing_priority")
            if priority is not None and not isinstance(priority, int):
                raise ValueError("Capability field 'routing_priority' must be an integer")
            tool_intent_required = capability.get("tool_intent_required")
            if tool_intent_required is not None and not isinstance(tool_intent_required, bool):
                raise ValueError("Capability field 'tool_intent_required' must be a boolean")
            safety_level = capability.get("safety_level")
            if safety_level is not None and safety_level not in {"allowed", "constrained", "blocked"}:
                raise ValueError(
                    "Capability field 'safety_level' must be one of: allowed, constrained, blocked"
                )
            safety_notes = capability.get("safety_notes")
            if safety_notes is not None and not isinstance(safety_notes, str):
                raise ValueError("Capability field 'safety_notes' must be a string")
