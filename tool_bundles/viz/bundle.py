from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.viz.adapters.graph_adapter import GraphAdapter
from tool_bundles.viz.adapters.network_adapter import NetworkAdapter
from tool_bundles.viz.adapters.parameter_space_adapter import ParameterSpaceAdapter
from tool_bundles.viz.adapters.timeline_adapter import TimelineAdapter


class VizToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "graph": GraphAdapter(manifest=manifest, repo_root=repo_root),
            "network": NetworkAdapter(manifest=manifest, repo_root=repo_root),
            "timeline": TimelineAdapter(manifest=manifest, repo_root=repo_root),
            "parameter_space": ParameterSpaceAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return VizToolBundle(manifest=manifest, repo_root=repo_root)
