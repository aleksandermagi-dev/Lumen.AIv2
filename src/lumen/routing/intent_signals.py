from __future__ import annotations

from dataclasses import dataclass, field

from lumen.nlu.follow_up_inventory import looks_like_general_follow_up, looks_like_reference_follow_up
from lumen.nlu.intent_extractor import IntentExtractor
from lumen.nlu.models import PromptUnderstanding
from lumen.nlu.social_phrase_inventory import SOCIAL_KIND_BY_PHRASE, phrases_for
from lumen.nlu.starter_prompt_layer import StarterPromptLayer
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy
from lumen.routing.anchor_registry import AnchorResolution, resolve_anchor_context


@dataclass(slots=True)
class PromptIntentSignals:
    normalized_prompt: str
    reconstructed_prompt: str
    prompt_parts: tuple[str, ...]
    detected_language: str = "en"
    normalized_topic: str | None = None
    dominant_intent: str = "unknown"
    intent_confidence: float = 0.0
    extracted_entities: tuple[dict[str, object], ...] = ()
    tool_alias_match: str | None = None
    explicit_greeting: bool = False
    explicit_social_kind: str | None = None
    explicit_planning_kind: str | None = None
    explicit_summary: bool = False
    explicit_comparison: bool = False
    follow_up: bool = False
    planning_score: int = 0
    research_score: int = 0
    action_score: int = 0
    answer_score: int = 0
    starter_category_scores: dict[str, int] = field(default_factory=dict)
    starter_categories: tuple[str, ...] = ()
    migration_hints: list[str] = field(default_factory=list)
    comparison_hints: list[str] = field(default_factory=list)
    anchor_resolution: dict[str, object] = field(default_factory=dict)
    structure_subject: str | None = None
    structure_predicate: str | None = None
    structure_object: str | None = None
    structure_modifiers: tuple[str, ...] = ()
    structure_completeness: str = "complete"
    structure_ambiguity_flags: tuple[str, ...] = ()
    structure_fragmentation_markers: tuple[str, ...] = ()
    structure_reconstruction_confidence: float = 0.0


class IntentSignalExtractor:
    """Normalizes prompt-level intent signals for the domain router."""

    def __init__(self) -> None:
        self.prompt_nlu = PromptNLU()

    EXPLICIT_GREETING_PREFIXES = phrases_for("conversation.greeting")

    GRATITUDE_PHRASES = phrases_for("conversation.gratitude")

    AFFIRMATION_PHRASES = phrases_for("conversation.affirmation")

    FAREWELL_PHRASES = phrases_for("conversation.farewell")

    CHECK_IN_PHRASES = phrases_for("conversation.check_in")

    MIGRATION_HINTS = (
        "migration",
        "migrate",
        "roadmap",
        "phases",
        "refactor",
    )

    COMPARISON_HINTS = (
        "compare",
        "comparison",
        "versus",
        "vs",
        "tradeoff",
        "tradeoffs",
        "difference",
        "differences",
    )

    def extract(self, prompt: str | PromptUnderstanding) -> PromptIntentSignals:
        understanding = prompt if isinstance(prompt, PromptUnderstanding) else self.prompt_nlu.analyze(prompt)
        router_view = understanding.router_view()
        normalized = router_view.route_ready_text.replace(" versus ", " vs ")
        anchor_resolution = resolve_anchor_context(normalized)
        starter_analysis = StarterPromptLayer.analyze(normalized)
        reference_follow_up = looks_like_reference_follow_up(normalized)
        parts = tuple(normalized.split(maxsplit=1))
        planning_score, research_score, action_score, answer_score = self._scores_from_anchors(anchor_resolution)
        planning_score += starter_analysis.category_scores.get("reasoning", 0)
        planning_score += starter_analysis.category_scores.get("builder", 0) * 2
        planning_score += starter_analysis.category_scores.get("system", 0)
        research_score += starter_analysis.category_scores.get("science", 0) * 2
        research_score += starter_analysis.category_scores.get("system", 0)
        research_score += starter_analysis.category_scores.get("exploration", 0)
        action_score += starter_analysis.category_scores.get("builder", 0)
        action_score += starter_analysis.category_scores.get("reasoning", 0)
        answer_score += starter_analysis.category_scores.get("science", 0) * 2
        answer_score += starter_analysis.category_scores.get("reasoning", 0)
        if self._looks_like_topic_only_query(normalized):
            research_score = max(research_score, 2)
            answer_score = max(answer_score, 2)
        if reference_follow_up:
            research_score = min(research_score, 1)
            answer_score = min(answer_score, 1)
        entity_values = {
            str(entity.value).strip().lower()
            for entity in understanding.entities
            if entity.value
        }
        entity_labels = {
            str(entity.label).strip().lower()
            for entity in understanding.entities
            if entity.label
        }
        planning_score, research_score, action_score, answer_score = self._cap_keyword_authority(
            planning_score=planning_score,
            research_score=research_score,
            action_score=action_score,
            answer_score=answer_score,
            dominant_intent=understanding.intent.label,
            intent_confidence=understanding.intent.confidence,
            starter_category_scores=starter_analysis.category_scores,
            entity_labels=entity_labels,
        )
        if understanding.intent.label == "planning":
            planning_score += 1
        elif understanding.intent.label == "research":
            research_score += 1
        if understanding.structure.predicate in {"design", "build", "create", "plan"}:
            planning_score += 1
            action_score += 1
        if understanding.structure.predicate in {"solve", "calculate"}:
            action_score += 2
            answer_score += 1
            research_score = min(research_score, max(research_score, 1))
        if "math_expression" in understanding.structure.fragmentation_markers:
            action_score += 1
        if "follow_up_shorthand" in understanding.structure.fragmentation_markers:
            reference_follow_up = True
        migration_hints = [hint for hint in self.MIGRATION_HINTS if hint in normalized or hint in entity_values]
        comparison_hints = [hint for hint in self.COMPARISON_HINTS if hint in normalized or hint in entity_values]
        if migration_hints:
            planning_score += 1
        if comparison_hints:
            research_score += 1
        if "archive" in entity_values:
            research_score += 1
        if any(label in IntentExtractor.EXPLANATORY_ENTITY_LABELS for label in entity_labels):
            research_score = max(research_score, 2)
            answer_score = max(answer_score, 2)
        return PromptIntentSignals(
            normalized_prompt=normalized,
            reconstructed_prompt=router_view.canonical_text,
            prompt_parts=parts,
            detected_language=router_view.detected_language,
            normalized_topic=router_view.normalized_topic,
            dominant_intent=router_view.dominant_intent,
            intent_confidence=router_view.intent_confidence,
            extracted_entities=router_view.extracted_entities,
            explicit_greeting=self._is_explicit_greeting(normalized),
            explicit_social_kind=self._explicit_social_kind(normalized),
            explicit_planning_kind=self._explicit_planning_kind(normalized, anchor_resolution),
            explicit_summary=self._is_explicit_summary(normalized, anchor_resolution),
            explicit_comparison=self._is_explicit_comparison(anchor_resolution),
            follow_up=self._looks_like_follow_up(normalized, anchor_resolution),
            planning_score=planning_score,
            research_score=research_score,
            action_score=action_score,
            answer_score=answer_score,
            starter_category_scores=dict(starter_analysis.category_scores),
            starter_categories=starter_analysis.matched_categories,
            migration_hints=migration_hints,
            comparison_hints=comparison_hints,
            anchor_resolution=anchor_resolution.to_dict(),
            structure_subject=router_view.structure_subject,
            structure_predicate=router_view.structure_predicate,
            structure_object=router_view.structure_object,
            structure_modifiers=router_view.structure_modifiers,
            structure_completeness=router_view.structure_completeness,
            structure_ambiguity_flags=router_view.structure_ambiguity_flags,
            structure_fragmentation_markers=router_view.structure_fragmentation_markers,
            structure_reconstruction_confidence=router_view.structure_reconstruction_confidence,
        )

    @staticmethod
    def _score(normalized_prompt: str, weights: dict[str, int]) -> int:
        score = 0
        for hint, weight in weights.items():
            if hint in normalized_prompt:
                score += weight
        return score

    @staticmethod
    def _cap_keyword_authority(
        *,
        planning_score: int,
        research_score: int,
        action_score: int,
        answer_score: int,
        dominant_intent: str,
        intent_confidence: float,
        starter_category_scores: dict[str, int],
        entity_labels: set[str],
    ) -> tuple[int, int, int, int]:
        strong_starter = any(score >= 3 for score in starter_category_scores.values())
        strong_entities = any(label in IntentExtractor.EXPLANATORY_ENTITY_LABELS for label in entity_labels)
        strong_intent = dominant_intent in {"planning", "research"} and intent_confidence >= 0.78
        if not (strong_starter or strong_entities or strong_intent):
            return planning_score, research_score, action_score, answer_score

        if dominant_intent == "planning":
            return (
                min(planning_score, 4),
                min(research_score, 2),
                min(action_score, 4),
                min(answer_score, 2),
            )
        if dominant_intent == "research" or strong_entities:
            return (
                min(planning_score, 2),
                min(research_score, 4),
                min(action_score, 2),
                min(answer_score, 4),
            )
        return (
            min(planning_score, 4),
            min(research_score, 4),
            min(action_score, 3),
            min(answer_score, 3),
        )

    @staticmethod
    def _scores_from_anchors(anchor_resolution: AnchorResolution) -> tuple[int, int, int, int]:
        planning_score = 0
        research_score = 0
        action_score = 0
        answer_score = 0
        for domain in anchor_resolution.domains:
            if domain in {"engineering", "planning"}:
                planning_score += 2
            elif domain == "social":
                pass
            else:
                research_score += 2
                answer_score += 1
        for action in anchor_resolution.actions:
            if action in {"plan", "brainstorm", "debug"}:
                planning_score += 2
                action_score += 2
            if action in {"define", "explain", "compare", "summarize", "analyze"}:
                research_score += 2
                answer_score += 2 if action in {"define", "explain", "summarize", "compare"} else 1
            if action in {"solve", "calculate"}:
                action_score += 2
                answer_score += 1
                research_score += 1
        return planning_score, research_score, action_score, answer_score

    def _explicit_planning_kind(self, normalized_prompt: str, anchor_resolution: AnchorResolution) -> str | None:
        explicit_planning_prefixes = (
            "create a migration plan",
            "create migration plan",
            "create a roadmap",
            "build a roadmap",
            "how do i build",
            "how do we build",
            "design ",
            "plan ",
            "blueprint ",
            "roadmap ",
        )
        if not normalized_prompt.startswith(explicit_planning_prefixes):
            return None
        if any(hint in normalized_prompt for hint in self.MIGRATION_HINTS):
            return "planning.migration"
        return "planning.architecture"

    def _is_explicit_greeting(self, normalized_prompt: str) -> bool:
        return normalized_prompt in self.EXPLICIT_GREETING_PREFIXES

    def _explicit_social_kind(self, normalized_prompt: str) -> str | None:
        return SOCIAL_KIND_BY_PHRASE.get(normalized_prompt)

    def _is_explicit_summary(self, normalized_prompt: str, anchor_resolution: AnchorResolution) -> bool:
        if any(
            token in normalized_prompt
            for token in ("content ideas", "content batch", "content drafts", "brainstorm content ideas")
        ):
            return False
        if anchor_resolution.primary_action in {"define", "explain", "summarize"}:
            return not normalized_prompt.startswith("what does ")
        return normalized_prompt.startswith(("what is ", "what are ", "tell me about ", "explain "))

    def _is_explicit_comparison(self, anchor_resolution: AnchorResolution) -> bool:
        return anchor_resolution.primary_action == "compare"

    def _looks_like_follow_up(self, normalized_prompt: str, anchor_resolution: AnchorResolution) -> bool:
        return bool(anchor_resolution.follow_up_kind) or looks_like_general_follow_up(normalized_prompt)

    @staticmethod
    def _looks_like_topic_only_query(normalized_prompt: str) -> bool:
        return ExplanatorySupportPolicy.evaluate(prompt=normalized_prompt).topic_only_query
