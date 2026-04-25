from __future__ import annotations

from pathlib import Path

from tool_bundles.data.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_data_result, correlation_payload, load_dataset_or_error


class CorrelateAdapter:
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
                summary="Couldn't correlate the dataset because no usable structured input was available.",
                json_name="data_correlation.json",
            )
        columns = request.params.get("columns")
        payload = {"status": "ok", **correlation_payload(records, columns=columns if isinstance(columns, list) else None), "runtime_diagnostics": {"runtime_ready": True, "input_ready": True}}
        return build_data_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Computed {len(payload['pairs'])} correlation pairs",
            json_name="data_correlation.json",
        )

