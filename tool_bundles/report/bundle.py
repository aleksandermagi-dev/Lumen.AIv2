from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.report.adapters.session_confidence_adapter import SessionConfidenceAdapter


class ReportToolBundle(ToolBundle):
    """Thin facade for local reporting capabilities."""

    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self.repo_root = repo_root
        self._capabilities = {
            "session.confidence": SessionConfidenceAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(
                f"Capability '{request.capability}' is not available in bundle '{self.id}'"
            )
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return ReportToolBundle(manifest=manifest, repo_root=repo_root)
