from __future__ import annotations

from pathlib import Path

from lumen.tools.math_core import MathExpressionError, solve_equation
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class EquationSolveAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        equation = str(request.params.get("equation") or "").strip()
        variable = str(request.params.get("variable") or "").strip()
        if not equation or not variable:
            raise ValueError("solve_equation requires 'equation' and 'variable'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            result = solve_equation(equation, variable)
            payload = {
                "status": "ok",
                "equation": equation,
                "variable": variable,
                "solution": result.solution,
                "steps": result.steps,
                "assumptions": result.assumptions,
                "confidence": result.confidence,
            }
            summary = f"Solved equation for {variable}"
            logs = [f"Solved equation '{equation}' for '{variable}'."]
        except MathExpressionError as exc:
            reason = str(exc)
            unsupported = "Only equations up to quadratic degree are supported" in reason
            payload = {
                "status": "error",
                "equation": equation,
                "variable": variable,
                "solution": [],
                "steps": [],
                "assumptions": [reason],
                "confidence": "low",
                "failure_category": "unsupported_operation" if unsupported else "invalid_input",
                "failure_reason": reason,
                "runtime_diagnostics": {
                    "failure_stage": "validation",
                    "input_ready": True,
                    "runtime_ready": True,
                    "unsupported_operation": unsupported,
                },
            }
            summary = (
                "The local math solver currently supports linear and quadratic equations; "
                f"that equation is beyond this solver."
                if unsupported
                else f"Could not solve equation for {variable}"
            )
            logs = [reason]
        artifact_path = write_json_artifact(outputs_dir, "equation_solution.json", payload)
        return ToolResult(
            status=payload["status"],
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data=payload,
            artifacts=[Artifact(name="equation_solution.json", path=artifact_path, media_type="application/json")],
            logs=logs,
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
            error=None if payload["status"] == "ok" else payload["assumptions"][0],
        )
