from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir

from tool_bundles.content.adapters._content_adapter_utils import (
    classify_generation_error,
    coerce_count,
    normalize_platform,
    normalize_style_profile,
)


class GenerateIdeasAdapter:
    def __init__(
        self,
        *,
        manifest: BundleManifest,
        repo_root: Path,
        transport: object | None = None,
    ) -> None:
        self.manifest = manifest
        self.repo_root = repo_root
        self._transport = transport

    def execute(self, request: ToolRequest) -> ToolResult:
        topic = str(request.params.get("topic") or "").strip() or None
        count = coerce_count(request.params.get("count"), default=5, maximum=10)
        platform = normalize_platform(request.params.get("platform"), default="all")
        style_profile = normalize_style_profile(request.params.get("style_profile"))
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            from lumen.content_generation.artifacts import ContentArtifactWriter
            from lumen.content_generation.service import ContentGenerationService

            artifacts_writer = ContentArtifactWriter()
            service = ContentGenerationService(repo_root=self.repo_root, transport=self._transport)
            ideas = service.generate_ideas(
                topic=topic,
                count=count,
                platform=platform,
                style_profile=style_profile,
            )
        except Exception as exc:
            failure = classify_generation_error(exc)
            return ToolResult(
                status="error",
                tool_id=request.tool_id,
                capability=request.capability,
                summary=(
                    "Content idea generation is unavailable until a hosted content provider is configured."
                    if failure["failure_category"] == "missing_provider_config"
                    else "Content idea generation could not run."
                ),
                structured_data={
                    "result_quality": failure["result_quality"],
                    "failure_category": failure["failure_category"],
                    "failure_reason": failure["failure_reason"],
                    "failure_detail": failure["failure_detail"],
                    "runtime_diagnostics": failure["runtime_diagnostics"],
                    "topic": topic,
                    "platform": platform,
                    "style_profile": style_profile,
                },
                logs=[str(exc)],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error=str(exc),
            )
        package = artifacts_writer.write_ideas_package(outputs_dir=outputs_dir, ideas=ideas)
        artifacts = [
            Artifact(name=Path(path).name, path=Path(path), media_type="text/plain" if path.endswith(".txt") else "application/json")
            for path in package.artifact_paths
        ]
        if package.manifest_path:
            artifacts.append(
                Artifact(
                    name=Path(package.manifest_path).name,
                    path=Path(package.manifest_path),
                    media_type="application/json",
                )
            )
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=f"Generated {len(ideas)} content ideas.",
            structured_data={
                "ideas": [item.to_mapping() for item in ideas],
                "count": len(ideas),
                "platform": platform,
                "style_profile": style_profile,
                "artifact_package": package.to_mapping(),
                "runtime_diagnostics": {
                    "provider_status": "ready_or_local_stub",
                    "runtime_ready": True,
                },
            },
            artifacts=artifacts,
            logs=[f"Generated {len(ideas)} ranked ideas."],
            provenance={"repo_root": str(self.repo_root), "style_profile": style_profile},
            run_dir=run_dir,
        )
