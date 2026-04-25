from __future__ import annotations

from pathlib import Path
from typing import Any

def coerce_count(value: Any, *, default: int, minimum: int = 1, maximum: int = 12) -> int:
    if value in (None, ""):
        return default
    count = int(value)
    return max(minimum, min(maximum, count))


def normalize_style_profile(value: Any) -> str:
    profile = str(value or "observational").strip().lower()
    return profile or "observational"


def normalize_platform(value: Any, *, default: str = "all") -> str:
    platform = str(value or default).strip().lower()
    aliases = {
        "youtube": "youtube_shorts",
        "youtube shorts": "youtube_shorts",
        "shorts": "youtube_shorts",
        "tik tok": "tiktok",
    }
    return aliases.get(platform, platform or default)


def load_recent_items(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    from lumen.content_generation.models import GeneratedContentDraft

    items: list[GeneratedContentDraft] = []
    for item in payload:
        if isinstance(item, dict):
            items.append(GeneratedContentDraft.from_mapping(item))
    return items


def collect_artifacts_from_package(package) -> list[Path]:
    paths: list[Path] = []
    if package.summary_path:
        paths.append(Path(package.summary_path))
    if package.manifest_path:
        paths.append(Path(package.manifest_path))
    for path in package.artifact_paths:
        candidate = Path(path)
        if candidate not in paths:
            paths.append(candidate)
    return paths


def classify_generation_error(exc: Exception) -> dict[str, object]:
    message = str(exc).strip()
    lowered = message.lower()
    if isinstance(exc, ModuleNotFoundError) and "lumen.content_generation" in message:
        return {
            "result_quality": "capability_unavailable",
            "failure_category": "runtime_dependency_failure",
            "failure_reason": "The packaged runtime is missing the internal content-generation module.",
            "failure_detail": message,
            "runtime_diagnostics": {
                "provider_status": "packaged_runtime_incomplete",
                "runtime_ready": False,
                "failure_hint": "Rebuild the packaged desktop runtime with lumen.content_generation included.",
                "missing_module": "lumen.content_generation",
            },
        }
    if (
        "hosted content generation is not configured" in lowered
        or "no openai model is configured for hosted inference" in lowered
        or "openai_responses_model" in lowered
    ):
        return {
            "result_quality": "capability_unavailable",
            "failure_category": "missing_provider_config",
            "failure_reason": "Hosted content generation is not configured for this runtime.",
            "failure_detail": message,
            "runtime_diagnostics": {
                "provider_status": "missing_provider_config",
                "runtime_ready": False,
                "failure_hint": "Configure OPENAI_API_KEY and an OpenAI Responses model for hosted generation.",
            },
        }
    if "valid json" in lowered:
        return {
            "result_quality": "generation_failed",
            "failure_category": "invalid_model_output",
            "failure_reason": "The hosted model returned output that could not be parsed into the expected content schema.",
            "failure_detail": message,
            "runtime_diagnostics": {
                "provider_status": "configured_or_unknown",
                "runtime_ready": True,
                "failure_hint": "Retry with a model or prompt that reliably emits valid structured JSON.",
            },
        }
    return {
        "result_quality": "generation_failed",
        "failure_category": "tool_execution_failure",
        "failure_reason": f"Content generation failed during hosted execution: {exc.__class__.__name__}.",
        "failure_detail": message,
        "runtime_diagnostics": {
            "provider_status": "configured_or_unknown",
            "runtime_ready": True,
            "exception_type": exc.__class__.__name__,
        },
    }
