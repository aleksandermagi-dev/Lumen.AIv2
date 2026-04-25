from __future__ import annotations

from pathlib import Path

from tool_bundles.viz.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_chart_svg, build_viz_result, extract_points_from_records, load_records


class ParameterSpaceAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        records = load_records(input_path=request.input_path, params=request.params)
        points = request.params.get("points")
        if not records and isinstance(points, list):
            records = [item for item in points if isinstance(item, dict)]
        points_payload = extract_points_from_records(
            records,
            x_column=str(request.params.get("x_column") or "").strip() or None,
            y_column=str(request.params.get("y_column") or "").strip() or None,
        )
        return build_viz_result(
            repo_root=self.repo_root,
            request=request,
            payload=points_payload,
            summary=(
                f"Rendered parameter-space plot for {points_payload['x_column']} vs {points_payload['y_column']}"
                if points_payload.get("status") == "ok"
                else "Couldn't render the parameter-space plot."
            ),
            json_name="parameter_space.json",
            svg_name="parameter_space.svg",
            svg_content=build_chart_svg(
                title="Parameter space",
                points=points_payload.get("points", []),
                x_label=str(points_payload.get("x_column") or "x"),
                y_label=str(points_payload.get("y_column") or "y"),
            ),
        )
