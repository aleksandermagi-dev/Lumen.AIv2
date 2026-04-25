from __future__ import annotations

from pathlib import Path

from lumen.tools.simulation_tools import simulate_population, simulation_svg
from tool_bundles.simulate.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_simulation_result, merged_params


class PopulationAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = simulate_population(merged_params(request=request))
        return build_simulation_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Simulated population growth over {int(payload['parameters']['steps'])} steps; "
                f"final population {payload['final_population']}"
            ),
            json_name="population_simulation.json",
            svg_name="population_simulation.svg",
            svg_content=simulation_svg(
                title="Population simulation",
                points=payload["points"],
                x_label="Step",
                y_label="Population",
            ),
        )
