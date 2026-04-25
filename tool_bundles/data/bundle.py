from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.data.adapters.cluster_adapter import ClusterAdapter
from tool_bundles.data.adapters.correlate_adapter import CorrelateAdapter
from tool_bundles.data.adapters.describe_adapter import DescribeAdapter
from tool_bundles.data.adapters.regression_adapter import RegressionAdapter
from tool_bundles.data.adapters.visualize_adapter import VisualizeAdapter


class DataToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "describe": DescribeAdapter(manifest=manifest, repo_root=repo_root),
            "correlate": CorrelateAdapter(manifest=manifest, repo_root=repo_root),
            "regression": RegressionAdapter(manifest=manifest, repo_root=repo_root),
            "cluster": ClusterAdapter(manifest=manifest, repo_root=repo_root),
            "visualize": VisualizeAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return DataToolBundle(manifest=manifest, repo_root=repo_root)
