from __future__ import annotations

from pathlib import Path

from lumen.tools.knowledge_graph_ops import KnowledgeOps
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class FindPathsAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root
        self.ops = KnowledgeOps.from_repo_root(repo_root)

    def execute(self, request: ToolRequest) -> ToolResult:
        source = str(request.params.get("source") or "").strip()
        target = str(request.params.get("target") or "").strip()
        max_hops = int(request.params.get("max_hops") or 3)
        if not source or not target:
            raise ValueError("knowledge.find_paths requires 'source' and 'target'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = self.ops.find_paths(source, target, max_hops=max_hops)
        artifact = write_json_artifact(outputs_dir, "knowledge_paths.json", payload)
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Found {payload['path_count']} knowledge paths",
            structured_data=payload,
            artifacts=[Artifact(name="knowledge_paths.json", path=artifact, media_type="application/json")],
            logs=[f"Resolved path search from '{source}' to '{target}'."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
