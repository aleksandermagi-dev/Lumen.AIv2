from __future__ import annotations

from pathlib import Path

from lumen.tools.knowledge_graph_ops import KnowledgeOps
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class ContradictionsAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root
        self.ops = KnowledgeOps.from_repo_root(repo_root)

    def execute(self, request: ToolRequest) -> ToolResult:
        claims = request.params.get("claims") or []
        strictness = str(request.params.get("strictness") or "medium").strip()
        if not isinstance(claims, list):
            raise ValueError("knowledge.contradictions requires 'claims' as a list")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = self.ops.contradictions([str(claim) for claim in claims], strictness)
        artifact = write_json_artifact(outputs_dir, "knowledge_contradictions.json", payload)
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Checked {len(claims)} claims for contradictions",
            structured_data=payload,
            artifacts=[Artifact(name="knowledge_contradictions.json", path=artifact, media_type="application/json")],
            logs=[f"Checked contradictions at strictness '{strictness}'."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
