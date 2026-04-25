from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.system.adapters.architecture_analyze_adapter import ArchitectureAnalyzeAdapter
from tool_bundles.system.adapters.docs_generate_adapter import DocsGenerateAdapter
from tool_bundles.system.adapters.refactor_suggest_adapter import RefactorSuggestAdapter


class SystemToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "analyze.architecture": ArchitectureAnalyzeAdapter(manifest=manifest, repo_root=repo_root),
            "suggest.refactor": RefactorSuggestAdapter(manifest=manifest, repo_root=repo_root),
            "generate.docs": DocsGenerateAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return SystemToolBundle(manifest=manifest, repo_root=repo_root)
