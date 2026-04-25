from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.content.adapters.generate_batch_adapter import GenerateBatchAdapter
from tool_bundles.content.adapters.generate_ideas_adapter import GenerateIdeasAdapter
from tool_bundles.content.adapters.format_platform_adapter import FormatPlatformAdapter


class ContentToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "generate_ideas": GenerateIdeasAdapter(manifest=manifest, repo_root=repo_root),
            "generate_batch": GenerateBatchAdapter(manifest=manifest, repo_root=repo_root),
            "format_platform": FormatPlatformAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return ContentToolBundle(manifest=manifest, repo_root=repo_root)
