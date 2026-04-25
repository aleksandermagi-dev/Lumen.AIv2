from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact
from lumen.tools.system_analysis import analyze_repo_slice, suggest_refactors


class RefactorSuggestAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        target_path = str(request.params.get("target_path") or "").strip()
        goal = str(request.params.get("goal") or "extract_helpers").strip()
        if not target_path:
            raise ValueError("suggest.refactor requires 'target_path'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        analysis = analyze_repo_slice(self.repo_root, target_path=target_path, depth=2)
        payload = suggest_refactors(analysis, goal=goal)
        artifact = write_json_artifact(outputs_dir, "refactor_suggestions.json", payload)
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Generated refactor suggestions for {target_path}",
            structured_data=payload,
            artifacts=[Artifact(name="refactor_suggestions.json", path=artifact, media_type="application/json")],
            logs=[f"Generated {len(payload['recommendations'])} recommendations for goal '{goal}'."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
