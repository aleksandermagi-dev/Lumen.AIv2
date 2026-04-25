from __future__ import annotations

from pathlib import Path

from lumen.tools.math_core import MathExpressionError, polynomial_from_expression, polynomial_to_expression
from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class SymbolicSimplifyAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        expression = str(request.params.get("expression") or "").strip()
        if not expression:
            raise ValueError("symbolic_simplify requires 'expression'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            simplified = polynomial_to_expression(polynomial_from_expression(expression))
            payload = {
                "status": "ok",
                "expression": expression,
                "simplified_expression": simplified,
                "applied_rules": ["normalize_terms", "combine_like_terms", "expand_powers"],
                "assumptions": ["Polynomial-style simplification only."],
                "confidence": "high",
            }
            summary = "Expression simplified"
            logs = [f"Simplified expression '{expression}'."]
        except MathExpressionError as exc:
            payload = {
                "status": "error",
                "expression": expression,
                "simplified_expression": expression,
                "applied_rules": [],
                "assumptions": [str(exc)],
                "confidence": "low",
            }
            summary = "Could not simplify expression"
            logs = [str(exc)]
        artifact_path = write_json_artifact(outputs_dir, "symbolic_simplify.json", payload)
        return ToolResult(
            status=payload["status"],
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data=payload,
            artifacts=[Artifact(name="symbolic_simplify.json", path=artifact_path, media_type="application/json")],
            logs=logs,
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
            error=None if payload["status"] == "ok" else payload["assumptions"][0],
        )
