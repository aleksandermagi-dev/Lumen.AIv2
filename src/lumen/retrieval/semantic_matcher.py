from __future__ import annotations

from dataclasses import dataclass, field

from lumen.nlu.models import PromptUnderstanding


@dataclass(slots=True)
class SemanticCandidate:
    prompt: str | None = None
    normalized_topic: str | None = None
    dominant_intent: str | None = None
    extracted_entities: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class SemanticMatchResult:
    score: int
    shared_prompt_tokens: tuple[str, ...] = field(default_factory=tuple)
    shared_topic_tokens: tuple[str, ...] = field(default_factory=tuple)
    shared_entities: tuple[str, ...] = field(default_factory=tuple)
    intent_match: bool = False


@dataclass(slots=True)
class SemanticSignature:
    prompt_tokens: tuple[str, ...] = field(default_factory=tuple)
    topic_tokens: tuple[str, ...] = field(default_factory=tuple)
    dominant_intent: str | None = None
    entities: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": list(self.prompt_tokens),
            "topic_tokens": list(self.topic_tokens),
            "dominant_intent": self.dominant_intent,
            "entities": list(self.entities),
        }


class SemanticMatcher:
    """Builds lightweight semantic similarity from prompt NLU signals."""

    STOPWORDS = {"the", "a", "an", "for", "of", "and", "to", "with", "that", "this", "it"}

    def score(
        self,
        query_understanding: PromptUnderstanding,
        candidate: SemanticCandidate,
    ) -> SemanticMatchResult:
        return self.score_signature(
            query_understanding,
            self.signature_from_candidate(candidate),
        )

    def score_signature(
        self,
        query_understanding: PromptUnderstanding,
        signature: SemanticSignature,
    ) -> SemanticMatchResult:
        query_prompt_tokens = self._meaningful_tokens(query_understanding.normalized_text)
        shared_prompt_tokens = tuple(sorted(query_prompt_tokens & set(signature.prompt_tokens)))

        query_topic_tokens = set(query_understanding.topic.tokens)
        shared_topic_tokens = tuple(sorted(query_topic_tokens & set(signature.topic_tokens)))

        query_entities = {
            entity.value.strip().lower()
            for entity in query_understanding.entities
            if entity.value
        }
        shared_entities = tuple(sorted(query_entities & set(signature.entities)))

        query_intent = str(query_understanding.intent.label or "").strip().lower()
        candidate_intent = str(signature.dominant_intent or "").strip().lower()
        intent_match = bool(query_intent and query_intent != "unknown" and query_intent == candidate_intent)

        if (
            len(shared_prompt_tokens) < 2
            and not shared_topic_tokens
            and not shared_entities
            and not intent_match
        ):
            return SemanticMatchResult(score=0)

        score = 0
        score += min(4, len(shared_prompt_tokens))
        score += min(4, len(shared_topic_tokens) * 2)
        score += min(4, len(shared_entities) * 2)
        if intent_match:
            score += 3

        return SemanticMatchResult(
            score=score,
            shared_prompt_tokens=shared_prompt_tokens,
            shared_topic_tokens=shared_topic_tokens,
            shared_entities=shared_entities,
            intent_match=intent_match,
        )

    def signature_from_candidate(self, candidate: SemanticCandidate) -> SemanticSignature:
        return SemanticSignature(
            prompt_tokens=tuple(sorted(self._meaningful_tokens(candidate.prompt))),
            topic_tokens=tuple(sorted(self._meaningful_tokens(candidate.normalized_topic))),
            dominant_intent=str(candidate.dominant_intent or "").strip().lower() or None,
            entities=tuple(
                sorted(
                    entity.strip().lower()
                    for entity in candidate.extracted_entities
                    if entity and entity.strip()
                )
            ),
        )

    def _meaningful_tokens(self, text: str | None) -> set[str]:
        if not text:
            return set()
        return {
            token
            for token in text.lower().split()
            if token not in self.STOPWORDS and len(token) > 2
        }
