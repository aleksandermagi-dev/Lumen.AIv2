from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from lumen.app.settings import AppSettings
from lumen.content_generation.models import (
    ContentBatchResult,
    ContentSafetyAssessment,
    GeneratedContentDraft,
    GeneratedContentVariant,
    GeneratedIdea,
)
from lumen.content_generation.prompts import (
    build_batch_prompt,
    build_format_prompt,
    build_ideas_prompt,
    build_rewrite_prompt,
    build_system_prompt,
)
from lumen.content_generation.safety import ContentSafetyLayer
from lumen.providers.factory import build_model_provider
from lumen.providers.local_provider import LocalOnlyProvider
from lumen.providers.models import InferenceRequest


class GenerationTransport(Protocol):
    def __call__(
        self,
        *,
        instructions: str,
        input_text: str,
        metadata: dict[str, Any],
        model: str | None,
    ) -> str: ...


class ProviderGenerationTransport:
    def __init__(self, repo_root: Path):
        self.settings = AppSettings.from_repo_root(repo_root)
        self.provider = build_model_provider(self.settings)

    def __call__(
        self,
        *,
        instructions: str,
        input_text: str,
        metadata: dict[str, Any],
        model: str | None,
    ) -> str:
        if isinstance(self.provider, LocalOnlyProvider) or self.provider.provider_id == "local":
            raise RuntimeError(
                "Hosted content generation is not configured. "
                "Set OPENAI_API_KEY and an OpenAI Responses model or configure a hosted inference provider."
            )
        request = InferenceRequest(
            model=model or self.settings.openai_responses_model,
            instructions=instructions,
            input_text=input_text,
            metadata=metadata,
            temperature=0.55,
            max_output_tokens=1400,
        )
        return self.provider.infer(request).output_text


class ContentGenerationService:
    def __init__(
        self,
        *,
        repo_root: Path,
        transport: GenerationTransport | None = None,
        model: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.transport = transport or ProviderGenerationTransport(repo_root)
        self.model = model
        self.safety = ContentSafetyLayer()

    def generate_ideas(
        self,
        *,
        topic: str | None,
        count: int,
        platform: str,
        style_profile: str,
    ) -> list[GeneratedIdea]:
        payload = self._request_json(
            prompt=build_ideas_prompt(topic=topic, count=count, platform=platform, style_profile=style_profile),
            style_profile=style_profile,
            metadata={"generation_mode": "ideas", "platform": platform},
        )
        return [
            GeneratedIdea.from_mapping(item, rank=index)
            for index, item in enumerate(payload.get("ideas", []), start=1)
        ]

    def generate_batch(
        self,
        *,
        topic: str,
        count: int,
        style_profile: str,
        recent_items: list[GeneratedContentDraft] | None = None,
    ) -> ContentBatchResult:
        payload = self._request_json(
            prompt=build_batch_prompt(topic=topic, count=count, style_profile=style_profile),
            style_profile=style_profile,
            metadata={"generation_mode": "batch"},
        )
        items: list[GeneratedContentDraft] = []
        discarded: list[dict[str, Any]] = []
        for item in payload.get("posts", []):
            draft = GeneratedContentDraft.from_mapping(
                {
                    **item,
                    "style_profile": style_profile,
                    "platform": "master",
                }
            )
            stable_draft, safety = self._stabilize_draft(
                draft,
                style_profile=style_profile,
                recent_items=recent_items,
            )
            if safety.decision == "DISCARD":
                discarded.append({"topic": stable_draft.topic, "reasons": list(safety.reasons)})
                continue
            items.append(stable_draft.with_updates(safety=safety))
        return ContentBatchResult(items=items, discarded_items=discarded)

    def format_variant(
        self,
        *,
        draft: GeneratedContentDraft,
        platform: str,
        style_profile: str,
        recent_items: list[GeneratedContentDraft] | None = None,
    ) -> GeneratedContentVariant:
        payload = self._request_json(
            prompt=build_format_prompt(
                item_payload=draft.to_mapping(),
                platform=platform,
                style_profile=style_profile,
            ),
            style_profile=style_profile,
            metadata={"generation_mode": "format", "platform": platform},
        )
        variant = GeneratedContentVariant.from_mapping(
            {
                **payload,
                "source_draft_id": draft.id,
                "platform": platform,
                "style_profile": style_profile,
            }
        )
        safety = self.safety.evaluate_variant(variant, recent_items=recent_items)
        if safety.decision == "REWRITE":
            rewritten = self._rewrite_payload(
                payload=variant.to_mapping(),
                safety=safety,
                style_profile=style_profile,
            )
            variant = GeneratedContentVariant.from_mapping(
                {
                    **rewritten,
                    "source_draft_id": draft.id,
                    "platform": platform,
                    "style_profile": style_profile,
                }
            )
            safety = self.safety.evaluate_variant(variant, recent_items=recent_items, rewrite_attempts=1)
            if safety.decision == "REWRITE":
                safety = ContentSafetyAssessment.from_mapping(
                    {
                        **safety.to_mapping(),
                        "decision": "DISCARD",
                        "reasons": [*safety.reasons, "Rewrite did not clear the quality gate."],
                    }
                )
        return GeneratedContentVariant.from_mapping({**variant.to_mapping(), "safety": safety.to_mapping()})

    def _stabilize_draft(
        self,
        draft: GeneratedContentDraft,
        *,
        style_profile: str,
        recent_items: list[GeneratedContentDraft] | None,
    ) -> tuple[GeneratedContentDraft, ContentSafetyAssessment]:
        safety = self.safety.evaluate_draft(draft, recent_items=recent_items)
        if safety.decision == "PASS":
            return draft, safety
        if safety.decision == "DISCARD":
            return draft, safety
        rewritten = self._rewrite_payload(
            payload=draft.to_mapping(),
            safety=safety,
            style_profile=style_profile,
        )
        rewritten_draft = GeneratedContentDraft.from_mapping(
            {
                **rewritten,
                "id": draft.id,
                "style_profile": style_profile,
                "platform": draft.platform,
            }
        )
        rewritten_safety = self.safety.evaluate_draft(
            rewritten_draft,
            recent_items=recent_items,
            rewrite_attempts=1,
        )
        if rewritten_safety.decision == "REWRITE":
            rewritten_safety = ContentSafetyAssessment.from_mapping(
                {
                    **rewritten_safety.to_mapping(),
                    "decision": "DISCARD",
                    "reasons": [*rewritten_safety.reasons, "Rewrite did not clear the quality gate."],
                }
            )
        return rewritten_draft, rewritten_safety

    def _rewrite_payload(
        self,
        *,
        payload: dict[str, Any],
        safety: ContentSafetyAssessment,
        style_profile: str,
    ) -> dict[str, Any]:
        return self._request_json(
            prompt=build_rewrite_prompt(
                post_payload=payload,
                safety_reasons=safety.reasons,
                style_profile=style_profile,
            ),
            style_profile=style_profile,
            metadata={"generation_mode": "rewrite"},
        )

    def _request_json(
        self,
        *,
        prompt: str,
        style_profile: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        raw = self.transport(
            instructions=build_system_prompt(style_profile=style_profile),
            input_text=prompt,
            metadata=metadata,
            model=self.model,
        )
        try:
            return json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response was not valid JSON: {raw}") from exc


def _extract_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return cleaned
