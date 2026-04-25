from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_json_artifact


class MatrixOperationsAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        operation = str(request.params.get("operation") or "").strip().lower()
        matrix_a = request.params.get("matrix_a")
        matrix_b = request.params.get("matrix_b")
        if not operation or not isinstance(matrix_a, list):
            raise ValueError("matrix_operations requires 'operation' and 'matrix_a'")
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        payload = self._run(operation, matrix_a, matrix_b)
        artifact_path = write_json_artifact(outputs_dir, "matrix_operation.json", payload)
        return ToolResult(
            status=payload["status"],
            tool_id=request.tool_id,
            capability=request.capability,
            summary=payload["summary"],
            structured_data=payload,
            artifacts=[Artifact(name="matrix_operation.json", path=artifact_path, media_type="application/json")],
            logs=payload["logs"],
            provenance={"repo_root": str(self.repo_root), "session_id": request.session_id},
            run_dir=run_dir,
            error=payload.get("error"),
        )

    def _run(self, operation: str, matrix_a, matrix_b) -> dict[str, object]:
        try:
            a = [[float(value) for value in row] for row in matrix_a]
            b = [[float(value) for value in row] for row in matrix_b] if matrix_b is not None else None
            if operation == "add":
                if b is None or len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
                    raise ValueError("Matrix addition requires matching dimensions")
                result = [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
            elif operation == "multiply":
                if b is None or len(a[0]) != len(b):
                    raise ValueError("Matrix multiplication requires aligned dimensions")
                result = [
                    [sum(a[row][k] * b[k][col] for k in range(len(b))) for col in range(len(b[0]))]
                    for row in range(len(a))
                ]
            elif operation == "transpose":
                result = [list(col) for col in zip(*a)]
            elif operation == "determinant":
                if len(a) != 2 or len(a[0]) != 2 or len(a[1]) != 2:
                    raise ValueError("Determinant is limited to 2x2 matrices in V1")
                result = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])
            elif operation == "inverse":
                if len(a) != 2 or len(a[0]) != 2 or len(a[1]) != 2:
                    raise ValueError("Inverse is limited to 2x2 matrices in V1")
                determinant = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])
                if determinant == 0:
                    raise ValueError("Matrix is singular and cannot be inverted")
                result = [
                    [a[1][1] / determinant, -a[0][1] / determinant],
                    [-a[1][0] / determinant, a[0][0] / determinant],
                ]
            else:
                raise ValueError(f"Unsupported matrix operation '{operation}'")
            return {
                "status": "ok",
                "summary": f"Matrix operation '{operation}' completed",
                "operation": operation,
                "result": result,
                "dimensions": {"matrix_a": [len(a), len(a[0])], "matrix_b": [len(b), len(b[0])] if b else None},
                "assumptions": ["Matrix operations are bounded to dense in-memory arrays."],
                "confidence": "high",
                "logs": [f"Ran matrix operation '{operation}'."],
            }
        except Exception as exc:
            return {
                "status": "error",
                "summary": f"Matrix operation '{operation}' failed",
                "operation": operation,
                "result": None,
                "dimensions": None,
                "assumptions": [str(exc)],
                "confidence": "low",
                "logs": [str(exc)],
                "error": str(exc),
            }
