from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir, write_artifact_manifest, write_json_artifact, write_svg_artifact
from lumen.tools.structured_data_tools import build_chart_svg, escape_xml, extract_points_from_records, load_records


def build_viz_result(
    *,
    repo_root: Path,
    request: ToolRequest,
    payload: dict[str, Any],
    summary: str,
    json_name: str,
    svg_name: str,
    svg_content: str,
) -> ToolResult:
    run_dir = build_run_dir(repo_root=repo_root, request=request)
    outputs_dir = ensure_outputs_dir(run_dir)
    json_artifact = write_json_artifact(outputs_dir, json_name, payload)
    svg_artifact = write_svg_artifact(outputs_dir, svg_name, svg_content)
    manifest_artifact = write_artifact_manifest(
        outputs_dir,
        {
            "status": payload.get("status", "ok"),
            "artifacts": [
                {"name": json_name, "path": str(json_artifact), "media_type": "application/json"},
                {"name": svg_name, "path": str(svg_artifact), "media_type": "image/svg+xml"},
            ],
            "runtime_diagnostics": payload.get("runtime_diagnostics", {}),
        },
    )
    return ToolResult(
        status=str(payload.get("status") or "ok"),
        tool_id=request.tool_id,
        capability=request.capability,
        summary=summary,
        structured_data=payload,
        artifacts=[
            Artifact(name=json_name, path=json_artifact, media_type="application/json"),
            Artifact(name=svg_name, path=svg_artifact, media_type="image/svg+xml"),
            Artifact(name="artifact_manifest.json", path=manifest_artifact, media_type="application/json"),
        ],
        logs=[summary],
        provenance={"session_id": request.session_id},
        run_dir=run_dir,
        error=None if str(payload.get("status") or "ok") == "ok" else str(payload.get("failure_reason") or ""),
    )


def load_json_input(input_path: Path | None) -> Any:
    if input_path is None or not input_path.exists() or input_path.suffix.lower() != ".json":
        return None
    return json.loads(input_path.read_text(encoding="utf-8"))


def simple_graph_svg(*, title: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    width = 640
    height = 360
    positions: dict[str, tuple[float, float]] = {}
    center_x = width / 2
    center_y = height / 2 + 10
    radius = 110
    count = max(1, len(nodes))
    for index, node in enumerate(nodes):
        angle = (6.28318 * index) / count
        node_id = str(node.get("id") or node.get("label") or index)
        positions[node_id] = (
            center_x + (radius * math.cos(angle)),
            center_y + (radius * math.sin(angle)),
        )
    lines: list[str] = []
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        lines.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#94a3b8" />')
    circles: list[str] = []
    for node in nodes:
        node_id = str(node.get("id") or node.get("label") or "")
        x, y = positions.get(node_id, (center_x, center_y))
        label = escape_xml(str(node.get("label") or node_id))
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="#1f77b4" />'
            f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#ffffff">{label}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">'
        f'<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="24" y="30" font-family="Arial" font-size="18">{escape_xml(title)}</text>'
        + "".join(lines)
        + "".join(circles)
        + "</svg>"
    )

