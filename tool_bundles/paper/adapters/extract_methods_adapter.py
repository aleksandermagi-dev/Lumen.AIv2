from __future__ import annotations

from pathlib import Path

from tool_bundles.paper.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_paper_result, extract_methods, load_paper_text


class ExtractMethodsAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = extract_methods(load_paper_text(input_path=request.input_path, params=request.params))
        return build_paper_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                "Extracted methods from the supplied paper text"
                if payload.get("status") == "ok"
                else "Method extraction could not run without readable paper text."
            ),
            json_name="paper_methods.json",
        )
