from __future__ import annotations

from pathlib import Path

from lumen.tools.design_spec_builder import DesignSpecBuilder
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class SystemSpecAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        brief = str(request.params.get("brief") or "").strip()
        if not brief:
            raise ValueError("system_spec requires a non-empty 'brief'")
        interaction_style = str(request.params.get("interaction_style") or "default").strip() or "default"
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = DesignSpecBuilder.build(
            brief=brief,
            interaction_style=interaction_style,
        )
        artifact_path = write_json_artifact(outputs_dir, "system_spec.json", payload)
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=str(payload.get("summary") or "Generated system spec."),
            structured_data=payload,
            artifacts=[
                Artifact(
                    name="system_spec.json",
                    path=artifact_path,
                    media_type="application/json",
                )
            ],
            logs=[f"Generated bounded system spec for '{brief}'."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
