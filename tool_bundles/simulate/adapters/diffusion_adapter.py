from __future__ import annotations

from pathlib import Path

from lumen.tools.simulation_tools import simulate_diffusion, simulation_svg
from tool_bundles.simulate.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_simulation_result, merged_params


class DiffusionAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = simulate_diffusion(merged_params(request=request))
        return build_simulation_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Simulated diffusion over {int(payload['parameters']['steps'])} steps; "
                f"center value {payload['center_value']}"
            ),
            json_name="diffusion_simulation.json",
            svg_name="diffusion_simulation.svg",
            svg_content=simulation_svg(
                title="Diffusion simulation",
                points=payload["points"],
                x_label="Position",
                y_label="Value",
            ),
        )
