from __future__ import annotations

from pathlib import Path

from lumen.tools.structured_data_tools import escape_xml
from tool_bundles.viz.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_viz_result, load_json_input


class TimelineAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = load_json_input(request.input_path) if request.input_path else None
        events = request.params.get("events") if isinstance(request.params.get("events"), list) else None
        if isinstance(payload, dict):
            events = events or payload.get("events")
        events = events if isinstance(events, list) else []
        structured = (
            {"status": "ok", "events": events, "runtime_diagnostics": {"runtime_ready": True, "input_ready": True}}
            if events
            else {
                "status": "error",
                "failure_category": "input_failure",
                "failure_reason": "Need dated events or an attached JSON timeline file.",
                "runtime_diagnostics": {"runtime_ready": True, "input_ready": False},
            }
        )
        return build_viz_result(
            repo_root=self.repo_root,
            request=request,
            payload=structured,
            summary=(
                f"Rendered timeline with {len(events)} events"
                if events
                else "Couldn't render the timeline because no events were available."
            ),
            json_name="timeline_view.json",
            svg_name="timeline_view.svg",
            svg_content=self._timeline_svg(events),
        )

    @staticmethod
    def _timeline_svg(events: list[dict[str, object]]) -> str:
        width = 720
        height = 220
        if not events:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="#ffffff" />'
                f'<text x="24" y="30" font-family="Arial" font-size="18">Timeline view</text>'
                f'<text x="24" y="70" font-family="Arial" font-size="14">No timeline events available.</text>'
                f"</svg>"
            )
        spacing = max(80, int((width - 80) / max(1, len(events))))
        markers: list[str] = [f'<line x1="40" y1="110" x2="{width - 40}" y2="110" stroke="#475569" />']
        for index, event in enumerate(events):
            x = 60 + (index * spacing)
            date = escape_xml(str(event.get("date") or event.get("time") or f"T{index + 1}"))
            label = escape_xml(str(event.get("label") or event.get("event") or f"Event {index + 1}"))
            markers.append(f'<circle cx="{x}" cy="110" r="8" fill="#1f77b4" />')
            markers.append(f'<text x="{x}" y="90" text-anchor="middle" font-family="Arial" font-size="11">{date}</text>')
            markers.append(f'<text x="{x}" y="138" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="#ffffff" />'
            f'<text x="24" y="30" font-family="Arial" font-size="18">Timeline view</text>'
            + "".join(markers)
            + "</svg>"
        )

