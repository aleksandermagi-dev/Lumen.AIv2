from __future__ import annotations

from pathlib import Path

from tool_bundles.paper.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_paper_result, search_papers


class SearchAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = search_papers(
            str(request.params.get("query") or "").strip(),
            input_path=request.input_path,
            params=request.params,
        )
        return build_paper_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Found {len(payload['results'])} paper candidates"
                if payload.get("status") == "ok"
                else "Paper search could not run in this runtime."
            ),
            json_name="paper_search.json",
        )

