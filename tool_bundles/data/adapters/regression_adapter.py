from __future__ import annotations

from pathlib import Path

from tool_bundles.data.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_data_result, load_dataset_or_error, regression_payload


class RegressionAdapter:
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
                summary="Couldn't run regression because no usable structured input was available.",
                json_name="data_regression.json",
            )
        payload = regression_payload(
            records,
            x_column=str(request.params.get("x_column") or "").strip() or None,
            y_column=str(request.params.get("y_column") or "").strip() or None,
        )
        return build_data_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Ran linear regression on {payload['x_column']} vs {payload['y_column']}"
                if payload.get("status") == "ok"
                else "Couldn't run regression on the supplied data."
            ),
            json_name="data_regression.json",
        )

