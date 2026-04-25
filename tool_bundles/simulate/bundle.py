from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.simulate.adapters.diffusion_adapter import DiffusionAdapter
from tool_bundles.simulate.adapters.orbit_adapter import OrbitAdapter
from tool_bundles.simulate.adapters.population_adapter import PopulationAdapter
from tool_bundles.simulate.adapters.system_adapter import SystemAdapter


class SimulationToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "system": SystemAdapter(manifest=manifest, repo_root=repo_root),
            "orbit": OrbitAdapter(manifest=manifest, repo_root=repo_root),
            "population": PopulationAdapter(manifest=manifest, repo_root=repo_root),
            "diffusion": DiffusionAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return SimulationToolBundle(manifest=manifest, repo_root=repo_root)
