from __future__ import annotations

from typing import Any

from lumen.tools.registry_types import ToolResult


OUTPUT_SCHEMA_VERSION = "1"


class OutputSchema:
    """Versioned helpers for tool execution and bundle inspection payloads."""

    @staticmethod
    def build_tool_result_payload(result: ToolResult) -> dict[str, Any]:
        return {
            "schema_type": "tool_result",
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": result.status,
            "tool_id": result.tool_id,
            "capability": result.capability,
            "summary": result.summary,
            "run_dir": str(result.run_dir) if result.run_dir else None,
            "archive_path": str(result.archive_path) if result.archive_path else None,
            "error": result.error,
            "structured_data": result.structured_data,
            "artifacts": [
                {
                    "name": artifact.name,
                    "path": str(artifact.path),
                    "media_type": artifact.media_type,
                    "description": artifact.description,
                }
                for artifact in result.artifacts
            ],
            "logs": result.logs,
            "provenance": result.provenance,
        }

    @staticmethod
    def build_bundle_inspection_payload(
        *,
        bundle_id: str,
        name: str,
        version: str,
        schema_version: str,
        description: str,
        manifest_path: str | None,
        entrypoint: str,
        capabilities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_type": "bundle_inspection",
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "name": name,
            "version": version,
            "bundle_schema_version": schema_version,
            "description": description,
            "manifest_path": manifest_path,
            "entrypoint": entrypoint,
            "capabilities": capabilities,
        }
