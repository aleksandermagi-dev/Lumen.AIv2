from __future__ import annotations

from pathlib import Path

from lumen.tools.math_core import MathExpressionError, sampled_optimize
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class OptimizeFunctionAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        expression = str(request.params.get("expression") or "").strip()
        variable = str(request.params.get("variable") or "").strip()
        bounds = request.params.get("bounds") or {}
        lower = float(bounds.get("min"))
        upper = float(bounds.get("max"))
        objective = str(request.params.get("objective") or "minimize").strip().lower()
        samples = int(request.params.get("samples") or 25)
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            if lower >= upper:
                raise ValueError("Bounds must have min < max")
            best_input, best_value = sampled_optimize(
                expression,
                variable,
                lower,
                upper,
                objective=objective,
                samples=samples,
            )
            payload = {
                "status": "ok",
                "expression": expression,
                "variable": variable,
                "best_input": best_input,
                "best_value": best_value,
                "method": "sampled_interval_search",
                "assumptions": ["Bounded sampled search only; result is approximate."],
                "confidence": "medium",
            }
            summary = f"Computed approximate {objective} optimum"
            logs = [f"Optimized '{expression}' over [{lower}, {upper}] with {samples} samples."]
        except (MathExpressionError, TypeError, ValueError) as exc:
            payload = {
                "status": "error",
                "expression": expression,
                "variable": variable,
                "best_input": None,
                "best_value": None,
                "method": "sampled_interval_search",
                "assumptions": [str(exc)],
                "confidence": "low",
            }
            summary = "Could not optimize function"
            logs = [str(exc)]
        artifact_path = write_json_artifact(outputs_dir, "function_optimization.json", payload)
        return ToolResult(
            status=payload["status"],
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data=payload,
            artifacts=[Artifact(name="function_optimization.json", path=artifact_path, media_type="application/json")],
            logs=logs,
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
            error=None if payload["status"] == "ok" else payload["assumptions"][0],
        )
