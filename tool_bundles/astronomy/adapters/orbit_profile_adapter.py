from __future__ import annotations

from pathlib import Path

from lumen.tools.domain_tools import orbit_profile_payload, orbit_profile_svg
from tool_bundles.astronomy.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_domain_result, merged_params


class OrbitProfileAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        payload = orbit_profile_payload(merged_params(request=request))
        return build_domain_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary="Built an astronomy orbit profile.",
            json_name="astronomy_orbit_profile.json",
            svg_name="astronomy_orbit_profile.svg",
            svg_content=orbit_profile_svg(payload),
        )
