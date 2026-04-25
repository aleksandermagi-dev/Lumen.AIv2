from __future__ import annotations

from pathlib import Path

from tool_bundles.viz.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_viz_result, load_json_input, simple_graph_svg


class GraphAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = load_json_input(request.input_path) if request.input_path else None
        nodes = request.params.get("nodes") if isinstance(request.params.get("nodes"), list) else None
        edges = request.params.get("edges") if isinstance(request.params.get("edges"), list) else None
        if isinstance(payload, dict):
            nodes = nodes or payload.get("nodes")
            edges = edges or payload.get("edges")
        nodes = nodes if isinstance(nodes, list) else []
        edges = edges if isinstance(edges, list) else []
        structured = (
            {"status": "ok", "nodes": nodes, "edges": edges, "runtime_diagnostics": {"runtime_ready": True, "input_ready": True}}
            if nodes
            else {
                "status": "error",
                "failure_category": "input_failure",
                "failure_reason": "Need node/edge data or an attached JSON graph file.",
                "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
            }
        )
        return build_viz_result(
            repo_root=self.repo_root,
            request=request,
            payload=structured,
            summary=(
                f"Rendered graph with {len(nodes)} nodes and {len(edges)} edges"
                if nodes
                else "Couldn't render the graph because no node data was available."
            ),
            json_name="graph_view.json",
            svg_name="graph_view.svg",
            svg_content=simple_graph_svg(title="Graph view", nodes=nodes, edges=edges),
        )

