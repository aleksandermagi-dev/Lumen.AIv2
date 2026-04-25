from __future__ import annotations

from pathlib import Path

from lumen.tools.invent_tools import failure_modes_payload, infer_brief
from tool_bundles.invent.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_invent_result, merged_params


class FailureModesAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        params = merged_params(request=request)
        payload = failure_modes_payload(params)
        return build_invent_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Analyzed likely failure modes for {infer_brief(params)}",
            json_name="invent_failure_modes.json",
        )
