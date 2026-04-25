from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.knowledge.adapters.cluster_adapter import ClusterAdapter
from tool_bundles.knowledge.adapters.contradictions_adapter import ContradictionsAdapter
from tool_bundles.knowledge.adapters.find_paths_adapter import FindPathsAdapter
from tool_bundles.knowledge.adapters.link_adapter import LinkAdapter


class KnowledgeToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "link": LinkAdapter(manifest=manifest, repo_root=repo_root),
            "find_paths": FindPathsAdapter(manifest=manifest, repo_root=repo_root),
            "cluster": ClusterAdapter(manifest=manifest, repo_root=repo_root),
            "contradictions": ContradictionsAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return KnowledgeToolBundle(manifest=manifest, repo_root=repo_root)
