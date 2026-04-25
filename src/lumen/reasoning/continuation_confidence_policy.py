from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder


@dataclass(slots=True)
class ContinuationConfidenceResult:
    normalized_prompt: str
    inherits_confidence: bool
    target_prompt: str
    source: str | None = None


class ContinuationConfidencePolicy:
    """Shared continuation-confidence helper for expansion turns; does not choose route."""

    CONTINUATION_CUES = {
        "tell me more",
        "tell me more about that",
        "go deeper",
        "continue",
        "keep going",
        "go on",
        "expand that",
        "expand on that",
        "what else",
    }

    @classmethod
    def evaluate(
        cls,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> ContinuationConfidenceResult:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if normalized not in cls.CONTINUATION_CUES or not recent_interactions:
            return ContinuationConfidenceResult(
                normalized_prompt=normalized,
                inherits_confidence=False,
                target_prompt=prompt,
            )
        latest = recent_interactions[0]
        latest_mode = str(latest.get("mode") or "").strip()
        latest_kind = str(latest.get("kind") or "").strip()
        if latest_mode != "research" or latest_kind not in {"research.summary", "research.general"}:
            return ContinuationConfidenceResult(
                normalized_prompt=normalized,
                inherits_confidence=False,
                target_prompt=prompt,
            )
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        previous_answer = str(
            response.get("user_facing_answer") or response.get("summary") or latest.get("summary") or ""
        ).strip()
        latest_prompt = str(
            response.get("resolved_prompt")
            or latest.get("resolved_prompt")
            or latest.get("prompt")
            or latest.get("normalized_topic")
            or response.get("topic")
            or ""
        ).strip()
        if previous_answer and latest_prompt and cls._is_substantive_answer(previous_answer):
            return ContinuationConfidenceResult(
                normalized_prompt=normalized,
                inherits_confidence=True,
                target_prompt=latest_prompt,
                source="recent_research_answer",
            )
        return ContinuationConfidenceResult(
            normalized_prompt=normalized,
            inherits_confidence=False,
            target_prompt=prompt,
        )

    @staticmethod
    def _is_substantive_answer(answer: str) -> bool:
        normalized = " ".join(str(answer or "").strip().lower().split())
        fallback_markers = (
            "i don't have enough grounded detail",
            "i dont have enough grounded detail",
            "i don't have enough detail",
            "i dont have enough detail",
            "i can tell the subject points to",
            "that's something i could explain, but",
        )
        return bool(normalized) and not any(marker in normalized for marker in fallback_markers)
