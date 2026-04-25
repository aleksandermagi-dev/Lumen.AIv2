from __future__ import annotations

from pathlib import Path

from lumen.tools.simulation_tools import simulate_orbit, simulation_svg
from tool_bundles.simulate.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_simulation_result, merged_params


class OrbitAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = simulate_orbit(merged_params(request=request))
        return build_simulation_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Simulated orbit with semi-major axis {payload['parameters']['semi_major_axis']} "
                f"and eccentricity {payload['parameters']['eccentricity']}"
            ),
            json_name="orbit_simulation.json",
            svg_name="orbit_simulation.svg",
            svg_content=simulation_svg(
                title="Orbit simulation",
                points=payload["points"],
                x_label="x",
                y_label="y",
            ),
        )
