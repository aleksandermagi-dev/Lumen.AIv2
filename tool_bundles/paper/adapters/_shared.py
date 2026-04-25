from __future__ import annotations

from pathlib import Path
from typing import Any

from lumen.tools.paper_tools import compare_papers, extract_methods, load_paper_text, search_papers, summarize_paper
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_artifact_manifest, write_json_artifact


def build_paper_result(
    *,
    repo_root: Path,
    request: ToolRequest,
    payload: dict[str, Any],
    summary: str,
    json_name: str,
) -> ToolResult:
    run_dir = build_run_dir(repo_root=repo_root, request=request)
    outputs_dir = ensure_outputs_dir(run_dir)
    json_artifact = write_json_artifact(outputs_dir, json_name, payload)
    manifest_artifact = write_artifact_manifest(
        outputs_dir,
        {
            "status": payload.get("status", "ok"),
            "artifacts": [{"name": json_name, "path": str(json_artifact), "media_type": "application/json"}],
            "runtime_diagnostics": payload.get("runtime_diagnostics", {}),
        },
    )
    return ToolResult(
        status=str(payload.get("status") or "ok"),
        tool_id=request.tool_id,
        capability=request.capability,
        summary=summary,
        structured_data=payload,
        artifacts=[
            Artifact(name=json_name, path=json_artifact, media_type="application/json"),
            Artifact(name="artifact_manifest.json", path=manifest_artifact, media_type="application/json"),
        ],
        logs=[summary],
        provenance={"session_id": request.session_id},
        run_dir=run_dir,
        error=None if str(payload.get("status") or "ok") == "ok" else str(payload.get("failure_reason") or ""),
    )

