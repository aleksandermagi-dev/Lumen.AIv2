from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir

from tool_bundles.content.adapters._content_adapter_utils import (
    classify_generation_error,
    coerce_count,
    collect_artifacts_from_package,
    load_recent_items,
    normalize_style_profile,
)


class GenerateBatchAdapter:
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
        topic = str(request.params.get("topic") or "").strip()
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        if not topic:
            return ToolResult(
                status="error",
                tool_id=request.tool_id,
                capability=request.capability,
                summary="Content batch generation needs a topic before it can run.",
                structured_data={
                    "result_quality": "missing_inputs",
                    "missing_inputs": ["topic"],
                },
                logs=["Missing required input: topic"],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error="generate_batch requires a non-empty 'topic'",
            )
        count = coerce_count(request.params.get("count"), default=3, maximum=8)
        style_profile = normalize_style_profile(request.params.get("style_profile"))
        recent_items = load_recent_items(request.params.get("recent_items"))
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            from lumen.content_generation.artifacts import ContentArtifactWriter
            from lumen.content_generation.service import ContentGenerationService

            artifacts_writer = ContentArtifactWriter()
            service = ContentGenerationService(repo_root=self.repo_root, transport=self._transport)
            batch = service.generate_batch(
                topic=topic,
                count=count,
                style_profile=style_profile,
                recent_items=recent_items,
            )
        except Exception as exc:
            failure = classify_generation_error(exc)
            return ToolResult(
                status="error",
                tool_id=request.tool_id,
                capability=request.capability,
                summary=(
                    "Content batch generation is unavailable until a hosted content provider is configured."
                    if failure["failure_category"] == "missing_provider_config"
                    else "Content batch generation could not run."
                ),
                structured_data={
                    "result_quality": failure["result_quality"],
                    "failure_category": failure["failure_category"],
                    "failure_reason": failure["failure_reason"],
                    "failure_detail": failure["failure_detail"],
                    "runtime_diagnostics": failure["runtime_diagnostics"],
                    "topic": topic,
                    "style_profile": style_profile,
                },
                logs=[str(exc)],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error=str(exc),
            )
        package = artifacts_writer.write_batch_package(outputs_dir=outputs_dir, items=batch.items, variants=batch.variants)
        batch.package = package
        artifacts = [
            Artifact(
                name=path.name,
                path=path,
                media_type="application/json" if path.suffix == ".json" else "text/plain",
            )
            for path in collect_artifacts_from_package(package)
        ]
        structured = batch.to_mapping()
        structured.update(
            {
                "topic": topic,
                "style_profile": style_profile,
                "result_quality": "batch_generated" if batch.items else "batch_empty",
                "generated_count": len(batch.items),
                "runtime_diagnostics": {
                    "provider_status": "ready_or_local_stub",
                    "runtime_ready": True,
                },
            }
        )
        summary = f"Generated {len(batch.items)} master content drafts."
        if batch.discarded_items:
            summary += f" Discarded {len(batch.discarded_items)} weak or unsafe drafts."
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data=structured,
            artifacts=artifacts,
            logs=[summary],
            provenance={"repo_root": str(self.repo_root), "style_profile": style_profile},
            run_dir=run_dir,
        )
