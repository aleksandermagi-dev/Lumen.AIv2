from __future__ import annotations

from lumen.reporting.output_formatter import OutputFormatter
from lumen.routing.tool_registry import ToolRegistry


class BundleService:
    """Handles bundle-level inspection/reporting concerns."""

    def __init__(self, registry: ToolRegistry, formatter: OutputFormatter):
        self.registry = registry
        self.formatter = formatter

    def inspect_bundle(self, bundle_id: str) -> dict[str, object]:
        manifests = self.registry.get_manifests()
        if bundle_id not in manifests:
            known = ", ".join(sorted(manifests)) or "<none>"
            raise ValueError(f"Unknown bundle '{bundle_id}'. Available bundles: {known}")

        manifest = manifests[bundle_id]
        return self.formatter.bundle_inspection_payload(
            bundle_id=manifest.id,
            name=manifest.name,
            version=manifest.version,
            schema_version=manifest.schema_version,
            description=manifest.description,
            manifest_path=str(manifest.manifest_path) if manifest.manifest_path else None,
            entrypoint=manifest.entrypoint,
            capabilities=[
                {
                    "id": capability.id,
                    "adapter": capability.adapter,
                    "description": capability.description,
                    "app_capability_key": capability.app_capability_key,
                    "app_description": capability.app_description,
                    "command_aliases": capability.command_aliases,
                    "input_schema": capability.input_schema,
                    "output_schema": capability.output_schema,
                    "trigger_keywords": capability.trigger_keywords,
                    "structural_patterns": capability.structural_patterns,
                    "intent_hints": capability.intent_hints,
                    "routing_priority": capability.routing_priority,
                    "tool_intent_required": capability.tool_intent_required,
                    "safety_level": capability.safety_level,
                    "safety_notes": capability.safety_notes,
                }
                for capability in manifest.capabilities
            ],
        )
