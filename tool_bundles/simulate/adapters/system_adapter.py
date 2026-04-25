from __future__ import annotations

from pathlib import Path

from lumen.tools.simulation_tools import simulate_system, simulation_svg
from tool_bundles.simulate.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_simulation_result, merged_params


class SystemAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = simulate_system(merged_params(request=request))
        return build_simulation_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=f"Simulated system over {int(payload['parameters']['steps'])} steps; final value {payload['final_value']}",
            json_name="system_simulation.json",
            svg_name="system_simulation.svg",
            svg_content=simulation_svg(
                title="System simulation",
                points=payload["points"],
                x_label="Step",
                y_label="Value",
            ),
        )
