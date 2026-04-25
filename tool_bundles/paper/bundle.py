from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.paper.adapters.compare_adapter import CompareAdapter
from tool_bundles.paper.adapters.extract_methods_adapter import ExtractMethodsAdapter
from tool_bundles.paper.adapters.search_adapter import SearchAdapter
from tool_bundles.paper.adapters.summary_adapter import SummaryAdapter


class PaperToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "search": SearchAdapter(manifest=manifest, repo_root=repo_root),
            "summary": SummaryAdapter(manifest=manifest, repo_root=repo_root),
            "compare": CompareAdapter(manifest=manifest, repo_root=repo_root),
            "extract.methods": ExtractMethodsAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return PaperToolBundle(manifest=manifest, repo_root=repo_root)
