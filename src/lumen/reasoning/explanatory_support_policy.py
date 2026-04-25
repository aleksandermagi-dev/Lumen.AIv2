from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.explanatory_intent_policy import ExplanatoryIntentPolicy
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder


@dataclass(slots=True)
class ExplanatorySupportSignals:
    normalized_prompt: str
    topic_only_query: bool
    broad_explanatory_prompt: bool
    blocked_knowledge_prompt: bool
    explanatory_entities: bool

    @property
    def explanatory_prompt(self) -> bool:
        return self.topic_only_query or self.broad_explanatory_prompt

    def should_consult_local_knowledge(self, *, route_mode: str, route_kind: str) -> bool:
        if route_mode != "research":
            return False
        if route_kind == "research.comparison":
            return True
        if route_kind not in {"research.summary", "research.general"}:
            return False
        if self.blocked_knowledge_prompt:
            return False
        return self.topic_only_query or self.broad_explanatory_prompt or self.explanatory_entities


class ExplanatorySupportPolicy:
    """Shared explanatory support signals for routing/orchestration/finalization."""

    @staticmethod
    def evaluate(
        *,
        prompt: str,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
    ) -> ExplanatorySupportSignals:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        return ExplanatorySupportSignals(
            normalized_prompt=normalized,
            topic_only_query=ExplanatoryIntentPolicy.looks_like_topic_only_query(normalized),
            broad_explanatory_prompt=ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt(normalized),
            blocked_knowledge_prompt=ExplanatoryIntentPolicy.is_blocked_knowledge_explanatory_prompt(normalized),
            explanatory_entities=ExplanatoryIntentPolicy.has_explanatory_entities(entities),
        )
