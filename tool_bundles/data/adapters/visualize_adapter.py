from __future__ import annotations

from pathlib import Path

from tool_bundles.data.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_chart_svg, build_data_result, extract_points_from_records, load_dataset_or_error


class VisualizeAdapter:
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
                summary="Couldn't visualize the dataset because no usable structured input was available.",
                json_name="data_visualization.json",
            )
        points_payload = extract_points_from_records(
            records,
            x_column=str(request.params.get("x_column") or "").strip() or None,
            y_column=str(request.params.get("y_column") or "").strip() or None,
        )
        svg = None
        if points_payload.get("status") == "ok":
            svg = build_chart_svg(
                title="Data visualization",
                points=points_payload["points"],
                x_label=points_payload["x_column"],
                y_label=points_payload["y_column"],
            )
        return build_data_result(
            repo_root=self.repo_root,
            request=request,
            payload=points_payload,
            summary=(
                f"Generated a visualization for {points_payload['x_column']} vs {points_payload['y_column']}"
                if points_payload.get("status") == "ok"
                else "Couldn't visualize the supplied data."
            ),
            json_name="data_visualization.json",
            svg_name="data_visualization.svg" if svg else None,
            svg_content=svg,
        )
