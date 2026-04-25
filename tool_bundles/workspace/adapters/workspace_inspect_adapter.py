from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult


class WorkspaceInspectAdapter:
    """Inspect the local workspace without external dependencies."""

    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        run_dir = self._build_run_dir(request)
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        summary = self._workspace_summary()
        artifact_path = outputs_dir / "workspace_summary.json"
        artifact_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary="Workspace structure inspection completed",
            structured_data=summary,
            artifacts=[
                Artifact(
                    name="workspace_summary.json",
                    path=artifact_path,
                    media_type="application/json",
                    description="Summarized top-level workspace structure",
                )
            ],
            logs=["Collected top-level workspace structure."],
            provenance={
                "repo_root": str(self.repo_root),
                "session_id": request.session_id,
            },
            run_dir=run_dir,
        )

    def _build_run_dir(self, request: ToolRequest) -> Path:
        root = request.run_root or (self.repo_root / "data" / "tool_runs")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / request.session_id / request.tool_id / request.capability / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _workspace_summary(self) -> dict[str, object]:
        top_level = []
        for path in sorted(self.repo_root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if path.name in {".git", "__pycache__", ".pytest_cache", ".venv"}:
                continue
            if path.is_dir():
                child_count = sum(1 for _ in path.iterdir())
                top_level.append(
                    {
                        "name": path.name,
                        "type": "directory",
                        "child_count": child_count,
                    }
                )
            else:
                top_level.append(
                    {
                        "name": path.name,
                        "type": "file",
                    }
                )

        directories = [item["name"] for item in top_level if item["type"] == "directory"]
        files = [item["name"] for item in top_level if item["type"] == "file"]
        return {
            "repo_root": str(self.repo_root),
            "top_level_count": len(top_level),
            "directory_count": len(directories),
            "file_count": len(files),
            "top_level": top_level,
            "highlights": {
                "has_src": "src" in directories,
                "has_tests": "tests" in directories,
                "has_tool_bundles": "tool_bundles" in directories,
                "has_data": "data" in directories,
            },
        }
