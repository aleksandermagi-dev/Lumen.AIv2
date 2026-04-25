from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

from lumen.schemas.manifest_schema import BUNDLE_MANIFEST_SCHEMA_VERSION, BundleManifestSchema


@dataclass(slots=True)
class Artifact:
    name: str
    path: Path
    media_type: str | None = None
    description: str | None = None


@dataclass(slots=True)
class ToolRequest:
    tool_id: str
    capability: str
    input_path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"
    run_root: Path | None = None


@dataclass(slots=True)
class ToolResult:
    status: str
    tool_id: str
    capability: str
    summary: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None
    error: str | None = None
    archive_path: Path | None = None


@dataclass(slots=True)
class CapabilityManifest:
    id: str
    adapter: str
    description: str = ""
    app_capability_key: str | None = None
    app_description: str | None = None
    command_aliases: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    trigger_keywords: list[str] = field(default_factory=list)
    structural_patterns: list[str] = field(default_factory=list)
    intent_hints: list[str] = field(default_factory=list)
    routing_priority: int = 0
    tool_intent_required: bool = True
    safety_level: str = "allowed"
    safety_notes: str = ""


@dataclass(slots=True)
class BundleManifest:
    id: str
    name: str
    version: str
    entrypoint: str
    schema_version: str = BUNDLE_MANIFEST_SCHEMA_VERSION
    description: str = ""
    capabilities: list[CapabilityManifest] = field(default_factory=list)
    manifest_path: Path | None = None

    @property
    def bundle_root(self) -> Path:
        if self.manifest_path is None:
            raise ValueError("manifest_path is not set")
        return self.manifest_path.parent

    def capability_map(self) -> dict[str, CapabilityManifest]:
        return {capability.id: capability for capability in self.capabilities}

    def validate(self) -> None:
        capability_ids: set[str] = set()
        app_capability_keys: set[str] = set()
        command_aliases: set[str] = set()

        for capability in self.capabilities:
            if capability.id in capability_ids:
                raise ValueError(
                    f"Duplicate capability id '{capability.id}' in bundle '{self.id}'"
                )
            capability_ids.add(capability.id)

            if capability.app_capability_key:
                if capability.app_capability_key in app_capability_keys:
                    raise ValueError(
                        f"Duplicate app capability key '{capability.app_capability_key}' "
                        f"in bundle '{self.id}'"
                    )
                app_capability_keys.add(capability.app_capability_key)

            for alias in capability.command_aliases:
                normalized = alias.strip().lower()
                if normalized in command_aliases:
                    raise ValueError(
                        f"Duplicate command alias '{alias}' in bundle '{self.id}'"
                    )
                command_aliases.add(normalized)

    @classmethod
    def from_file(cls, manifest_path: Path) -> "BundleManifest":
        payload = BundleManifestSchema.normalize(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        BundleManifestSchema.validate(payload)
        capabilities = [
            CapabilityManifest(
                id=item["id"],
                adapter=item["adapter"],
                description=item.get("description", ""),
                app_capability_key=item.get("app_capability_key"),
                app_description=item.get("app_description"),
                command_aliases=item.get("command_aliases", []),
                input_schema=item.get("input_schema", {}),
                output_schema=item.get("output_schema", {}),
                trigger_keywords=item.get("trigger_keywords", []),
                structural_patterns=item.get("structural_patterns", []),
                intent_hints=item.get("intent_hints", []),
                routing_priority=int(item.get("routing_priority", 0)),
                tool_intent_required=bool(item.get("tool_intent_required", True)),
                safety_level=str(item.get("safety_level", "allowed")),
                safety_notes=str(item.get("safety_notes", "")),
            )
            for item in payload.get("capabilities", [])
        ]
        manifest = cls(
            schema_version=str(payload.get("schema_version", BUNDLE_MANIFEST_SCHEMA_VERSION)),
            id=payload["id"],
            name=payload["name"],
            version=payload.get("version", "0.1.0"),
            entrypoint=payload["entrypoint"],
            description=payload.get("description", ""),
            capabilities=capabilities,
            manifest_path=manifest_path,
        )
        manifest.validate()
        return manifest
