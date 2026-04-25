from __future__ import annotations

from pathlib import Path

from lumen.tools.experiment_tools import experiment_variables_payload, infer_topic
from tool_bundles.experiment.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_experiment_result, merged_params


class VariablesAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        params = merged_params(request=request)
        payload = experiment_variables_payload(params)
        return build_experiment_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Identified experiment variables for {infer_topic(params)}",
            json_name="experiment_variables.json",
        )
