from __future__ import annotations

from pathlib import Path

from lumen.tools.math_core import MathExpressionError, simpson_integrate, trapezoid_integrate
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class NumericalIntegrateAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        expression = str(request.params.get("expression") or "").strip()
        variable = str(request.params.get("variable") or "").strip()
        lower_bound = float(request.params.get("lower_bound"))
        upper_bound = float(request.params.get("upper_bound"))
        method = str(request.params.get("method") or "trapezoid").strip().lower()
        steps = int(request.params.get("steps") or 10)
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            if method == "simpson":
                estimate = simpson_integrate(expression, variable, lower_bound, upper_bound, steps)
            else:
                estimate = trapezoid_integrate(expression, variable, lower_bound, upper_bound, steps)
            payload = {
                "status": "ok",
                "expression": expression,
                "variable": variable,
                "estimate": estimate,
                "method": method,
                "error_notes": [],
                "assumptions": ["Bounded numerical integration only."],
                "confidence": "medium",
            }
            summary = f"Computed {method} integral estimate"
            logs = [f"Integrated '{expression}' from {lower_bound} to {upper_bound} using {method}."]
        except (MathExpressionError, TypeError, ValueError) as exc:
            payload = {
                "status": "error",
                "expression": expression,
                "variable": variable,
                "estimate": None,
                "method": method,
                "error_notes": [str(exc)],
                "assumptions": [str(exc)],
                "confidence": "low",
            }
            summary = "Could not compute integral estimate"
            logs = [str(exc)]
        artifact_path = write_json_artifact(outputs_dir, "numerical_integral.json", payload)
        return ToolResult(
            status=payload["status"],
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data=payload,
            artifacts=[Artifact(name="numerical_integral.json", path=artifact_path, media_type="application/json")],
            logs=logs,
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
            error=None if payload["status"] == "ok" else payload["assumptions"][0],
        )
