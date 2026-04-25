from __future__ import annotations

from pathlib import Path

from tool_bundles.paper.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_paper_result, compare_papers, load_paper_text


class CompareAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        papers = request.params.get("papers") if isinstance(request.params.get("papers"), list) else None
        if not papers:
            text = load_paper_text(input_path=request.input_path, params=request.params)
            if "\n---\n" in text:
                papers = [item.strip() for item in text.split("\n---\n") if item.strip()]
        payload = compare_papers([str(item) for item in papers] if papers else [])
        return build_paper_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Compared {payload['paper_count']} supplied papers"
                if payload.get("status") == "ok"
                else "Paper comparison could not run without at least two readable paper texts."
            ),
            json_name="paper_compare.json",
        )

