from __future__ import annotations

from lumen.nlu.explanatory_intent_policy import ExplanatoryIntentPolicy
from lumen.nlu.models import ExtractedEntity, IntentParse, NormalizedTopic
from lumen.nlu.starter_prompt_layer import StarterPromptLayer


class IntentExtractor:
    """Heuristic intent and entity extraction for early NLU integration."""

    PLANNING_PREFIXES = (
        "create a plan",
        "create a roadmap",
        "build a roadmap",
        "create a migration plan",
        "design the architecture",
        "propose an architecture",
    )
    RESEARCH_PREFIXES = (
        "summarize",
        "summary of",
        "explain",
        "describe",
        "tell me about",
        "what about",
        "what is",
        "what are",
        "how about",
        "who is",
        "who was",
        "where is",
        "where was",
        "when was",
        "when did",
        "define",
        "compare",
        "contrast",
    )
    TOOL_HINTS = (
        "analyze ga",
        "inspect workspace",
        "report session confidence",
        "inspect session timeline",
    )
    ENTITY_HINTS = {
        "tool": ("anh", "workspace", "report", "memory"),
        "capability": ("ga", "workspace", "confidence", "timeline"),
        "domain": ("archive", "routing", "migration", "roadmap", "session"),
        "concept": ("black hole", "gravity", "evolution", "democracy", "relativity"),
    }
    EVENT_MARKERS = ("war", "battle", "revolution", "election", "treaty", "era", "crisis")
    PLACE_MARKERS = ("city", "river", "mount", "mountain", "ocean", "sea", "country", "state")
    OBJECT_MARKERS = ("planet", "star", "galaxy", "telescope", "atom", "bridge", "engine")
    FORMULA_MARKERS = ("formula", "law", "equation")
    PROCESS_MARKERS = ("process", "cycle", "reaction", "transfer")
    SYSTEM_MARKERS = ("system", "loop", "network", "circuit", "engine")
    EXPLANATORY_ENTITY_LABELS = ExplanatoryIntentPolicy.EXPLANATORY_ENTITY_LABELS

    def extract_intent(self, text: str, *, topic: NormalizedTopic | None = None) -> IntentParse:
        normalized = " ".join(text.strip().lower().split())
        starter_analysis = StarterPromptLayer.analyze(normalized)
        planning_prefix_match = any(normalized.startswith(prefix) for prefix in self.PLANNING_PREFIXES)
        research_prefix_match = any(normalized.startswith(prefix) for prefix in self.RESEARCH_PREFIXES)
        if any(hint in normalized for hint in self.TOOL_HINTS):
            return IntentParse(label="tool", confidence=0.9)
        if starter_analysis.category_scores.get("science", 0) >= 2:
            return IntentParse(label="research", confidence=0.8)
        if starter_analysis.category_scores.get("builder", 0) >= 2:
            return IntentParse(label="planning", confidence=0.78)
        if planning_prefix_match:
            return IntentParse(label="planning", confidence=0.72)
        if research_prefix_match:
            return IntentParse(label="research", confidence=0.7)
        if starter_analysis.category_scores.get("reasoning", 0) >= 2:
            return IntentParse(label="planning", confidence=0.74)
        if starter_analysis.category_scores.get("system", 0) >= 2:
            return IntentParse(label="research", confidence=0.72)
        if starter_analysis.category_scores.get("exploration", 0) >= 2:
            return IntentParse(label="research", confidence=0.68)
        if self._looks_like_topic_only_query(normalized):
            return IntentParse(label="research", confidence=0.64)
        topic_tokens = set((topic.tokens if topic is not None else ()))
        if {"migration", "roadmap", "architecture"} & topic_tokens:
            return IntentParse(label="planning", confidence=0.68)
        if {"summary", "compare", "archive"} & topic_tokens:
            return IntentParse(label="research", confidence=0.66)
        if self._looks_like_explanatory_subject_query(normalized):
            return IntentParse(label="research", confidence=0.62)
        return IntentParse(label="unknown", confidence=0.45)

    def extract_entities(
        self,
        text: str,
        *,
        topic: NormalizedTopic | None = None,
        original_text: str | None = None,
    ) -> tuple[ExtractedEntity, ...]:
        normalized = " ".join(text.strip().lower().split())
        topic_tokens = set((topic.tokens if topic is not None else ()))
        entities: list[ExtractedEntity] = []
        for label, hints in self.ENTITY_HINTS.items():
            for hint in hints:
                if hint in normalized or hint in topic_tokens:
                    confidence = 0.82 if hint in normalized else 0.72
                    entities.append(
                        ExtractedEntity(
                            label=label,
                            value=hint,
                            confidence=confidence,
                        )
                    )
        entities.extend(
            self._subject_entities(
                normalized=normalized,
                topic=topic,
                original_text=original_text or text,
            )
        )
        deduped: dict[tuple[str, str], ExtractedEntity] = {}
        for entity in entities:
            key = (entity.label, entity.value)
            existing = deduped.get(key)
            if existing is None or entity.confidence > existing.confidence:
                deduped[key] = entity
        return tuple(deduped.values())

    @staticmethod
    def _looks_like_topic_only_query(normalized: str) -> bool:
        return ExplanatoryIntentPolicy.looks_like_topic_only_query(normalized)

    @classmethod
    def _looks_like_explanatory_subject_query(cls, normalized: str) -> bool:
        return ExplanatoryIntentPolicy.looks_like_explanatory_subject_query(normalized)

    @classmethod
    def _subject_entities(
        cls,
        *,
        normalized: str,
        topic: NormalizedTopic | None,
        original_text: str,
    ) -> list[ExtractedEntity]:
        topic_value = str(getattr(topic, "value", "") or normalized).strip().lower()
        original_normalized = " ".join(str(original_text).strip().lower().split())
        if not topic_value:
            return []
        entities: list[ExtractedEntity] = []
        raw_words = [word.strip(".,!?") for word in original_text.split() if word.strip(".,!?")]
        title_case_words = [word for word in raw_words if word[:1].isupper()]
        if len(title_case_words) >= 2:
            entities.append(ExtractedEntity(label="person", value=" ".join(title_case_words[:3]).lower(), confidence=0.74))
        if any(marker in topic_value for marker in cls.EVENT_MARKERS):
            entities.append(ExtractedEntity(label="event", value=topic_value, confidence=0.7))
        elif any(marker in topic_value for marker in cls.PLACE_MARKERS):
            entities.append(ExtractedEntity(label="place", value=topic_value, confidence=0.68))
        elif any(marker in topic_value for marker in cls.FORMULA_MARKERS) or any(symbol in topic_value for symbol in ("=", "^", "sqrt")):
            entities.append(ExtractedEntity(label="formula", value=topic_value, confidence=0.7))
        elif any(marker in topic_value for marker in cls.SYSTEM_MARKERS):
            entities.append(ExtractedEntity(label="system", value=topic_value, confidence=0.68))
        elif any(marker in topic_value for marker in cls.PROCESS_MARKERS):
            entities.append(ExtractedEntity(label="process", value=topic_value, confidence=0.66))
        elif any(marker in topic_value for marker in cls.OBJECT_MARKERS):
            entities.append(ExtractedEntity(label="object", value=topic_value, confidence=0.66))
        elif cls._looks_like_explanatory_subject_query(original_normalized):
            entities.append(ExtractedEntity(label="concept", value=topic_value, confidence=0.62))
        return entities
