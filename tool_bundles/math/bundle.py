from __future__ import annotations

from pathlib import Path

from lumen.tools.base import ToolBundle
from lumen.tools.registry_types import BundleManifest, ToolRequest, ToolResult

from tool_bundles.math.adapters.equation_solve_adapter import EquationSolveAdapter
from tool_bundles.math.adapters.matrix_operations_adapter import MatrixOperationsAdapter
from tool_bundles.math.adapters.numerical_integrate_adapter import NumericalIntegrateAdapter
from tool_bundles.math.adapters.optimize_function_adapter import OptimizeFunctionAdapter
from tool_bundles.math.adapters.symbolic_simplify_adapter import SymbolicSimplifyAdapter


class MathToolBundle(ToolBundle):
    def __init__(self, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest)
        self._capabilities = {
            "solve_equation": EquationSolveAdapter(manifest=manifest, repo_root=repo_root),
            "symbolic_simplify": SymbolicSimplifyAdapter(manifest=manifest, repo_root=repo_root),
            "matrix_operations": MatrixOperationsAdapter(manifest=manifest, repo_root=repo_root),
            "numerical_integrate": NumericalIntegrateAdapter(manifest=manifest, repo_root=repo_root),
            "optimize_function": OptimizeFunctionAdapter(manifest=manifest, repo_root=repo_root),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        if request.capability not in self._capabilities:
            raise KeyError(f"Capability '{request.capability}' is not available in bundle '{self.id}'")
        return self._capabilities[request.capability].execute(request)


def create_bundle(manifest: BundleManifest, repo_root: Path) -> ToolBundle:
    return MathToolBundle(manifest=manifest, repo_root=repo_root)
