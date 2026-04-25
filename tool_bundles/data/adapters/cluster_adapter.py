from __future__ import annotations

from pathlib import Path

from tool_bundles.data.adapters._shared import BundleManifest, ToolRequest, ToolResult, build_data_result, cluster_payload, load_dataset_or_error


class ClusterAdapter:
    def __init__(self, *, manifest: BundleManifest, repo_root: Path):
        self.manifest = manifest
        self.repo_root = repo_root

    def execute(self, request: ToolRequest) -> ToolResult:
        records, error = load_dataset_or_error(request=request)
        if error is not None:
            return build_data_result(
                repo_root=self.repo_root,
                request=request,
                payload=error,
                summary="Couldn't cluster the dataset because no usable structured input was available.",
                json_name="data_clusters.json",
            )
        columns = request.params.get("columns")
        cluster_count = int(request.params.get("cluster_count") or 2)
        payload = cluster_payload(
            records,
            columns=columns if isinstance(columns, list) else None,
            cluster_count=max(2, min(cluster_count, 6)),
        )
        return build_data_result(
            repo_root=self.repo_root,
            request=request,
            payload=payload,
            summary=(
                f"Clustered the dataset into {payload['cluster_count']} groups"
                if payload.get("status") == "ok"
                else "Couldn't cluster the supplied data."
            ),
            json_name="data_clusters.json",
        )

