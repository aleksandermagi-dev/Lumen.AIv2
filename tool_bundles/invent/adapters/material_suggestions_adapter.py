from __future__ import annotations

from pathlib import Path

from lumen.tools.invent_tools import infer_brief, material_suggestions_payload
from tool_bundles.invent.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_invent_result, merged_params


class MaterialSuggestionsAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        params = merged_params(request=request)
        payload = material_suggestions_payload(params)
        return build_invent_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Suggested material classes for {infer_brief(params)}",
            json_name="invent_material_suggestions.json",
        )
