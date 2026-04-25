from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.models import PromptUnderstanding
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy, ExplanatorySupportSignals


@dataclass(slots=True)
class RouteSupportSignals:
    dominant_intent: str
    normalized_topic: str | None
    explanatory: ExplanatorySupportSignals

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant_intent": self.dominant_intent,
            "normalized_topic": self.normalized_topic,
            "explanatory_prompt": self.explanatory.explanatory_prompt,
            "broad_explanatory_prompt": self.explanatory.broad_explanatory_prompt,
            "topic_only_query": self.explanatory.topic_only_query,
            "explanatory_entities": self.explanatory.explanatory_entities,
            "blocked_knowledge_prompt": self.explanatory.blocked_knowledge_prompt,
        }


class RouteSupportSignalBuilder:
    """Builds downstream-safe support signals from upstream analysis without changing route authority."""

    @staticmethod
    def build(
        *,
        prompt: str | None = None,
        dominant_intent: str | None = None,
        normalized_topic: str | None = None,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
        prompt_understanding: PromptUnderstanding | None = None,
    ) -> RouteSupportSignals:
        if prompt_understanding is not None:
            prompt = prompt_understanding.canonical_text
            dominant_intent = prompt_understanding.intent.label
            normalized_topic = prompt_understanding.topic.value
            entities = tuple(entity.to_dict() for entity in prompt_understanding.entities)
        explanatory = ExplanatorySupportPolicy.evaluate(
            prompt=str(prompt or ""),
            entities=entities,
        )
        return RouteSupportSignals(
            dominant_intent=str(dominant_intent or "unknown"),
            normalized_topic=normalized_topic,
            explanatory=explanatory,
        )
