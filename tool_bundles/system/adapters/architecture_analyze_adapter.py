from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact, write_text_artifact
from lumen.tools.system_analysis import analyze_repo_slice, render_architecture_map


class ArchitectureAnalyzeAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        target_path = request.params.get("target_path")
        depth = request.params.get("depth")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        analysis = analyze_repo_slice(
            self.repo_root,
            target_path=str(target_path) if target_path else None,
            depth=int(depth) if depth is not None else None,
        )
        json_artifact = write_json_artifact(outputs_dir, "architecture_report.json", analysis)
        text_artifact = write_text_artifact(outputs_dir, "architecture_map.txt", render_architecture_map(analysis))
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Architecture analysis completed for {analysis['target_path']}",
            structured_data=analysis,
            artifacts=[
                Artifact(name="architecture_report.json", path=json_artifact, media_type="application/json"),
                Artifact(name="architecture_map.txt", path=text_artifact, media_type="text/plain"),
            ],
            logs=[f"Analyzed {analysis['file_count']} files."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
