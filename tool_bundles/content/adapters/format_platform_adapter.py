from __future__ import annotations

from pathlib import Path

from lumen.tools.registry_types import Artifact, BundleManifest, ToolRequest, ToolResult
from lumen.tools.run_utils import build_run_dir, ensure_outputs_dir

from tool_bundles.content.adapters._content_adapter_utils import (
    classify_generation_error,
    collect_artifacts_from_package,
    load_recent_items,
    normalize_platform,
    normalize_style_profile,
)


class FormatPlatformAdapter:
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
        run_dir = build_run_dir(repo_root=self.repo_root, request=request)
        draft_payload = request.params.get("draft")
        source_text = str(request.params.get("source_text") or "").strip()
        topic_hint = str(request.params.get("topic") or "").strip()
        platform = normalize_platform(request.params.get("platform"), default="")
        if platform not in {"tiktok", "youtube_shorts"}:
            return ToolResult(
                status="error",
                tool_id=request.tool_id,
                capability=request.capability,
                summary="Platform formatting needs a target platform like TikTok or YouTube Shorts.",
                structured_data={
                    "result_quality": "missing_inputs",
                    "missing_inputs": ["platform"],
                },
                logs=["Missing or unsupported platform for format_platform."],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error="format_platform requires platform='tiktok' or 'youtube_shorts'",
            )
        if not isinstance(draft_payload, dict) and not source_text:
            return ToolResult(
                status="error",
                tool_id=request.tool_id,
                capability=request.capability,
                summary="Platform formatting needs source text or a structured draft before it can run.",
                structured_data={
                    "result_quality": "missing_inputs",
                    "missing_inputs": ["draft_or_source_text"],
                    "platform": platform,
                },
                logs=["Missing required input: draft or source_text"],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error="format_platform requires a structured 'draft' payload or 'source_text'",
            )
        style_profile = normalize_style_profile(request.params.get("style_profile"))
        recent_items = load_recent_items(request.params.get("recent_items"))
        if isinstance(draft_payload, dict):
            from lumen.content_generation.models import GeneratedContentDraft
            draft = GeneratedContentDraft.from_mapping(draft_payload)
        else:
            draft = self._draft_from_source_text(
                source_text=source_text,
                topic_hint=topic_hint,
                style_profile=style_profile,
            )
        outputs_dir = ensure_outputs_dir(run_dir)
        try:
            from lumen.content_generation.artifacts import ContentArtifactWriter
            from lumen.content_generation.service import ContentGenerationService

            artifacts_writer = ContentArtifactWriter()
            service = ContentGenerationService(repo_root=self.repo_root, transport=self._transport)
            variant = service.format_variant(
                draft=draft,
                platform=platform,
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
                    "Platform formatting is unavailable until a hosted content provider is configured."
                    if failure["failure_category"] == "missing_provider_config"
                    else "Platform formatting could not run."
                ),
                structured_data={
                    "result_quality": failure["result_quality"],
                    "failure_category": failure["failure_category"],
                    "failure_reason": failure["failure_reason"],
                    "failure_detail": failure["failure_detail"],
                    "runtime_diagnostics": failure["runtime_diagnostics"],
                    "platform": platform,
                    "style_profile": style_profile,
                },
                logs=[str(exc)],
                provenance={"repo_root": str(self.repo_root)},
                run_dir=run_dir,
                error=str(exc),
            )
        package = artifacts_writer.write_batch_package(outputs_dir=outputs_dir, items=[draft], variants=[variant])
        artifacts = [
            Artifact(
                name=path.name,
                path=path,
                media_type="application/json" if path.suffix == ".json" else "text/plain",
            )
            for path in collect_artifacts_from_package(package)
        ]
        summary = f"Formatted draft '{draft.topic}' for {platform.replace('_', ' ')}."
        return ToolResult(
            status="ok",
            tool_id=request.tool_id,
            capability=request.capability,
            summary=summary,
            structured_data={
                "draft": draft.to_mapping(),
                "variant": variant.to_mapping(),
                "platform": platform,
                "style_profile": style_profile,
                "artifact_package": package.to_mapping(),
                "result_quality": "variant_generated",
                "runtime_diagnostics": {
                    "provider_status": "ready_or_local_stub",
                    "runtime_ready": True,
                },
            },
            artifacts=artifacts,
            logs=[summary],
            provenance={"repo_root": str(self.repo_root), "style_profile": style_profile},
            run_dir=run_dir,
        )

    @staticmethod
    def _draft_from_source_text(
        *,
        source_text: str,
        topic_hint: str,
        style_profile: str,
    ):
        from lumen.content_generation.models import GeneratedContentDraft

        cleaned = str(source_text or "").strip()
        lines = [line.strip(" -\t") for line in cleaned.splitlines() if line.strip()]
        if not lines:
            sentence_parts = [part.strip() for part in cleaned.replace("?", ".").replace("!", ".").split(".") if part.strip()]
            lines = sentence_parts or [cleaned]
        script_lines = lines[:6]
        hook = script_lines[0]
        topic = topic_hint.strip() or hook[:80].strip() or "Formatted content draft"
        caption = cleaned if len(cleaned) <= 180 else f"{cleaned[:177].rstrip()}..."
        return GeneratedContentDraft.from_mapping(
            {
                "id": f"draft_{abs(hash((topic, hook, caption))) % 100000000:08d}",
                "topic": topic,
                "hook": hook,
                "script_lines": script_lines,
                "caption": caption,
                "hashtags": [],
                "platform_notes": "Adapted from plain source text.",
                "retention_check": "Keep the original idea intact while tightening the pacing.",
                "suggested_post_time": "afternoon",
                "style_profile": style_profile,
                "platform": "master",
            }
        )
