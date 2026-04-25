from __future__ import annotations

from pathlib import Path

from lumen.tools.knowledge_graph_ops import KnowledgeOps
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class LinkAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root
        self.ops = KnowledgeOps.from_repo_root(repo_root)

    def execute(self, request: ToolRequest) -> ToolResult:
        items = request.params.get("items") or []
        if not isinstance(items, list):
            raise ValueError("knowledge.link requires 'items' as a list")
        relation_hint = request.params.get("relation_hint")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = self.ops.link([str(item) for item in items], str(relation_hint) if relation_hint else None)
        artifact = write_json_artifact(outputs_dir, "knowledge_links.json", payload)
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Generated {len(payload['links'])} knowledge links",
            structured_data=payload,
            artifacts=[Artifact(name="knowledge_links.json", path=artifact, media_type="application/json")],
            logs=[f"Linked {len(items)} items."],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
        )
