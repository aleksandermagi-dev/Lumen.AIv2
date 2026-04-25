from __future__ import annotations

from pathlib import Path

from tool_bundles.viz.adapters.graph_adapter import GraphAdapter
from tool_bundles.viz.adapters._shared import BundleManifest


class NetworkAdapter(GraphAdapter):
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        super().__init__(manifest=manifest, repo_root=repo_root)

