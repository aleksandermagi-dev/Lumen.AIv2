from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.invent.adapters.constraint_check_adapter import ConstraintCheckAdapter
from tool_bundles.invent.adapters.failure_modes_adapter import FailureModesAdapter
from tool_bundles.invent.adapters.generate_concepts_adapter import GenerateConceptsAdapter
from tool_bundles.invent.adapters.material_suggestions_adapter import MaterialSuggestionsAdapter


class InventToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "generate_concepts": GenerateConceptsAdapter(manifest=manifest, repo_root=repo_root),
            "constraint_check": ConstraintCheckAdapter(manifest=manifest, repo_root=repo_root),
            "material_suggestions": MaterialSuggestionsAdapter(manifest=manifest, repo_root=repo_root),
            "failure_modes": FailureModesAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return InventToolBundle(manifest=manifest, repo_root=repo_root)
