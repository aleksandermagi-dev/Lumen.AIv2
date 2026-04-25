from __future__ import annotations

from pathlib import Path

from lumen.tools.domain_tools import energy_model_payload, energy_model_svg
from tool_bundles.physics.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_domain_result, merged_params


class EnergyModelAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = energy_model_payload(merged_params(request=request))
        return build_domain_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary="Built a bounded classical energy model.",
            json_name="physics_energy_model.json",
            svg_name="physics_energy_model.svg",
            svg_content=energy_model_svg(payload),
        )
