from __future__ import annotations

from pathlib import Path

from tool_bundles.data.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_data_result, load_dataset_or_error, summarize_records


class DescribeAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        records, error = load_dataset_or_error(request=request)
        if error is not None:
            return build_data_result(
                repo_root=self.repo_root,
                request=request,
                payload=error,
                summary="Couldn't describe the dataset because no usable structured input was available.",
                json_name="data_description.json",
            )
        payload = {"status": "ok", **summarize_records(records), "runtime_diagnostics": {"runtime_ready": True, "input_ready": True}}
        return build_data_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Described dataset with {payload['row_count']} rows and {payload['column_count']} columns",
            json_name="data_description.json",
        )

