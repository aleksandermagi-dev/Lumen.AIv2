from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.tools.invent_tools import load_invent_params
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_artifact_manifest, write_json_artifact


def merged_params(*, request: ToolRequest) -> dict[str, Any]:
    return load_invent_params(input_path=request.input_path, params=request.params)


def build_invent_result(
    *,
    repo_root: Path,
    request: ToolRequest,
    payload: dict[str, Any],
    summary: str,
    json_name: str,
) -> ToolResult:
    run_dir = build_run_dir(repo_root=repo_root, request=request)
    outputs_dir = ensure_outputs_dir(run_dir)
    artifacts: list[Artifact] = []
    json_artifact = write_json_artifact(outputs_dir, json_name, payload)
    artifacts.append(Artifact(name=json_name, path=json_artifact, media_type="application/json"))
    manifest_artifact = write_artifact_manifest(
        outputs_dir,
        {
            "status": payload.get("status", "ok"),
            "artifacts": [
                {"name": artifact.name, "path": str(artifact.path), "media_type": artifact.media_type}
                for artifact in artifacts
            ],
            "runtime_diagnostics": payload.get("runtime_diagnostics", {}),
        },
    )
    artifacts.append(Artifact(name="artifact_manifest.json", path=manifest_artifact, media_type="application/json"))
    return ToolResult(
        status=str(payload.get("status") or "ok"),
        tool_id=request.tool_id,
        capability=request.capability,
        summary=summary,
        structured_data=payload,
        artifacts=artifacts,
        logs=[summary],
        provenance={"session_id": request.session_id},
        run_dir=run_dir,
        error=None if str(payload.get("status") or "ok") == "ok" else str(payload.get("failure_reason") or ""),
    )
