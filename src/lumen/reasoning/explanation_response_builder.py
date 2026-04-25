from __future__ import annotations

from dataclasses import dataclass

from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.knowledge.models import KnowledgeEntry, KnowledgeLookupResult
from lumen.nlu.explanatory_intent_policy import ExplanatoryIntentPolicy
from lumen.nlu.focus_resolution import FocusResolutionSupport
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.explanatory_support_policy import ExplanatorySupportPolicy, ExplanatorySupportSignals


@dataclass(slots=True)
class ExplanationAnswerResult:
    answer: str
    source: str
    lookup_succeeded: bool = False
    confidence_inherited: bool = False

    @property
    def should_replace_surface(self) -> bool:
        return self.source in {"lookup", "partial", "provider"}


class ExplanationResponseBuilder:
    """Builds user-facing explanatory answers from local knowledge first."""

    _default_knowledge_service: KnowledgeService | None = None

    @classmethod
    def build_answer(
        cls,
        *,
        prompt: str,
        interaction_style: str,
        explanation_strategy: str | None = None,
        continuation: bool = False,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
        provider_text: str | None = None,
        knowledge_service: KnowledgeService | None = None,
    ) -> ExplanationAnswerResult:
        service = knowledge_service or cls._default_service()
        simple_requested = cls._wants_simple_explanation(prompt)
        deep_requested = cls._wants_deeper_explanation(prompt)
        strategy = cls._resolve_strategy(
            prompt=prompt,
            explanation_strategy=explanation_strategy,
            simple_requested=simple_requested,
            deep_requested=deep_requested,
        )
        lookup = service.lookup(prompt) if service is not None else None
        answer_source = "lookup"
        answer = cls._answer_from_lookup(
            lookup,
            knowledge_service=service,
            prompt=prompt,
            strategy=strategy,
            continuation=continuation,
            simple_requested=simple_requested,
            deep_requested=deep_requested,
        )
        lookup_succeeded = bool(answer)
        if not answer and service is not None:
            partial_lookup = service.partial_lookup(prompt)
            answer = cls._partial_match_answer(prompt=prompt, lookup=partial_lookup)
            if answer:
                answer_source = "partial"
        if not answer:
            answer = cls._clean_provider_text(provider_text)
            if answer:
                answer_source = "provider"
        if not answer:
            answer = cls._generic_entity_answer(prompt=prompt, entities=entities)
            if answer:
                answer_source = "generic"
        if not answer:
            answer = (
                "Sorry, I don't know that because I don't have enough local knowledge on that specific topic yet. "
                "If you narrow it, I can try the closest grounded answer."
            )
            answer_source = "fallback"
        return ExplanationAnswerResult(
            answer=cls._apply_style(answer=answer, interaction_style=interaction_style, answer_source=answer_source),
            source=answer_source,
            lookup_succeeded=lookup_succeeded or answer_source == "partial",
            confidence_inherited=continuation,
        )

    @classmethod
    def lookup_local_knowledge(
        cls,
        *,
        prompt: str,
        knowledge_service: KnowledgeService | None = None,
    ) -> KnowledgeLookupResult | None:
        service = knowledge_service or cls._default_service()
        return service.lookup(prompt)

    @classmethod
    def should_consult_local_knowledge(
        cls,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
        support_signals: ExplanatorySupportSignals | None = None,
    ) -> bool:
        # Route is already selected upstream; this gate only decides whether local knowledge may shape the answer.
        signals = support_signals or ExplanatorySupportPolicy.evaluate(prompt=prompt, entities=entities)
        return signals.should_consult_local_knowledge(route_mode=route_mode, route_kind=route_kind)

    @classmethod
    def _default_service(cls) -> KnowledgeService:
        if cls._default_knowledge_service is None:
            cls._default_knowledge_service = KnowledgeService.in_memory()
        return cls._default_knowledge_service

    @staticmethod
    def _looks_like_explanatory_prompt(normalized_prompt: str) -> bool:
        return ExplanatoryIntentPolicy.looks_like_broad_explanatory_prompt(normalized_prompt)

    @staticmethod
    def _is_blocked_explanatory_prompt(normalized_prompt: str) -> bool:
        return ExplanatoryIntentPolicy.is_blocked_knowledge_explanatory_prompt(normalized_prompt)

    @staticmethod
    def _has_explanatory_entities(
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None,
    ) -> bool:
        return ExplanatoryIntentPolicy.has_explanatory_entities(entities)

    @classmethod
    def _answer_from_lookup(
        cls,
        lookup: KnowledgeLookupResult | None,
        *,
        knowledge_service: KnowledgeService | None,
        prompt: str,
        strategy: str,
        continuation: bool = False,
        simple_requested: bool = False,
        deep_requested: bool = False,
    ) -> str | None:
        if lookup is None:
            return None
        if lookup.partial:
            return None
        if lookup.mode == "comparison" and lookup.primary is not None and lookup.secondary is not None:
            return cls._comparison_answer(
                lookup.primary,
                lookup.secondary,
                relation=lookup.comparison_relation,
                prompt=prompt,
                strategy=strategy,
            )
        if lookup.primary is None:
            return None
        if lookup.primary.formula is not None:
            return cls._formula_answer(
                lookup.primary,
                strategy=strategy,
                continuation=continuation or deep_requested,
                simple_requested=simple_requested,
            )
        related_connections = (
            knowledge_service.related_connections(lookup.primary.id)
            if knowledge_service is not None
            else []
        )
        return cls._entry_answer(
            lookup.primary,
            related_connections=related_connections,
            strategy=strategy,
            continuation=continuation or deep_requested,
            simple_requested=simple_requested,
        )

    @staticmethod
    def _comparison_answer(
        left: KnowledgeEntry,
        right: KnowledgeEntry,
        relation: str | None = None,
        *,
        prompt: str = "",
        strategy: str = "compare_contrast",
    ) -> str:
        normalized_prompt = " ".join(str(prompt or "").strip().lower().split())
        relation_requested = " in relation to " in normalized_prompt or " related to " in normalized_prompt
        shared_frame = f"{left.title} and {right.title} are related, but they are not the same thing."
        if relation == "compares_with":
            shared_frame = f"{left.title} and {right.title} are often compared because they share some broad territory, but the key distinction matters."
        if relation_requested:
            shared_frame = f"{left.title} and {right.title} connect through the same broader physical picture, even though they describe different parts of it."

        left_core = left.summary_short or left.summary_medium
        right_core = right.summary_short or right.summary_medium
        difference = ExplanationResponseBuilder._comparison_difference(left=left, right=right)
        use_case = ExplanationResponseBuilder._comparison_use_case(left=left, right=right)
        bridge = ExplanationResponseBuilder._comparison_bridge(
            left=left,
            right=right,
            relation_requested=relation_requested,
            strategy=strategy,
        )

        parts = [shared_frame]
        if left_core:
            parts.append(f"{left.title}: {left_core}")
        if right_core:
            parts.append(f"{right.title}: {right_core}")
        if bridge:
            parts.append(bridge)
        if difference:
            parts.append(difference)
        if use_case:
            parts.append(use_case)
        if strategy == "step_by_step":
            steps = [
                f"1. Start with {left.title}: {left_core or left.title}.",
                f"2. Then add {right.title}: {right_core or right.title}.",
            ]
            if bridge:
                steps.append(f"3. Connection: {bridge}")
            elif difference:
                steps.append(f"3. Main difference: {difference}")
            return "\n".join(steps)
        return " ".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _comparison_difference(*, left: KnowledgeEntry, right: KnowledgeEntry) -> str | None:
        if left.entry_type == right.entry_type == "object":
            left_points = tuple(point.lower() for point in left.key_points)
            right_points = tuple(point.lower() for point in right.key_points)
            left_has_surface = any(
                "surface" in point and "no " not in point and "unlike" not in point
                for point in left_points
            )
            right_has_surface = any(
                "surface" in point and "no " not in point and "unlike" not in point
                for point in right_points
            )
            left_lacks_surface = any("no solid surface" in point or "does not" in point for point in left_points)
            right_lacks_surface = any("no solid surface" in point or "does not" in point for point in right_points)
            if right_has_surface and left_lacks_surface:
                return f"The main difference is that {right.title} still has a physical surface, while {left.title} does not in the ordinary sense."
            if left_has_surface and right_lacks_surface:
                return f"The main difference is that {left.title} still has a physical surface, while {right.title} does not in the ordinary sense."
        if left.entry_type == right.entry_type == "concept":
            return f"The main difference is what each one measures: {left.title} and {right.title} describe different parts of the same broader system."
        if left.entry_type == right.entry_type == "system":
            return f"The main difference is role: {left.title} and {right.title} sit in the same space, but they solve different parts of the problem."
        return None

    @staticmethod
    def _comparison_use_case(*, left: KnowledgeEntry, right: KnowledgeEntry) -> str | None:
        left_related = {topic.lower() for topic in left.related_topics}
        right_related = {topic.lower() for topic in right.related_topics}
        if left.entry_type == right.entry_type == "object" and ("gravity" in left_related or "gravity" in right_related):
            return f"If you're comparing them in astronomy, the useful question is usually whether you're looking at an ultra-dense star with a surface or an object whose gravity creates an event horizon."
        if left.entry_type == right.entry_type == "concept":
            return f"If you want, the next useful step is usually to compare how {left.title} and {right.title} interact in a concrete example."
        return None

    @staticmethod
    def _comparison_bridge(
        *,
        left: KnowledgeEntry,
        right: KnowledgeEntry,
        relation_requested: bool,
        strategy: str,
    ) -> str | None:
        if not relation_requested and strategy != "compare_contrast":
            return None
        left_title = left.title.lower()
        right_title = right.title.lower()
        titles = {left_title, right_title}
        if "entropy" in titles and "black hole" in titles:
            return (
                "In relation to black holes, entropy is used to talk about how many physical states are consistent with the black hole as a thermodynamic object, rather than just treating the black hole as gravity alone."
            )
        if left.entry_type == "concept" and right.entry_type == "object":
            return f"{left.title} helps describe one aspect of how {right.title} behaves or is interpreted in a broader physical theory."
        if left.entry_type == "object" and right.entry_type == "concept":
            return f"{right.title} helps describe one aspect of how {left.title} behaves or is interpreted in a broader physical theory."
        if left.entry_type == right.entry_type == "concept":
            return f"They connect because each one describes a different part of the same larger framework."
        return None

    @staticmethod
    def _formula_answer(
        entry: KnowledgeEntry,
        *,
        strategy: str = "direct_definition",
        continuation: bool = False,
        simple_requested: bool = False,
    ) -> str:
        formula = entry.formula
        if formula is None:
            return entry.summary_medium or entry.summary_short
        variable_text = ", ".join(
            f"{name} = {meaning}" for name, meaning in formula.variable_meanings.items()
        )
        base_summary = entry.summary_medium or entry.summary_short
        if simple_requested and entry.summary_short:
            base_summary = entry.summary_short
        elif continuation and entry.summary_deep:
            base_summary = entry.summary_deep
        answer = f"{base_summary} The formula is {formula.formula_text}."
        if variable_text:
            answer += f" Here, {variable_text}."
        if formula.units:
            units_text = ", ".join(f"{name} in {unit}" for name, unit in formula.units.items())
            answer += f" Typical units are {units_text}."
        if formula.interpretation:
            answer += f" {formula.interpretation}"
        if formula.example_usage:
            answer += f" Example: {formula.example_usage}"
        if strategy == "step_by_step" and formula.example_usage:
            answer = "\n".join(
                [
                    f"1. Start with the idea: {base_summary}",
                    f"2. Use the formula {formula.formula_text}.",
                    f"3. Example: {formula.example_usage}",
                ]
            )
        return answer

    @staticmethod
    def _entry_answer(
        entry: KnowledgeEntry,
        *,
        related_connections: list[tuple[str, str]],
        strategy: str = "direct_definition",
        continuation: bool = False,
        simple_requested: bool = False,
    ) -> str:
        if simple_requested and entry.summary_short:
            base = entry.summary_short
        else:
            base = (
                entry.summary_deep
                if continuation and entry.summary_deep
                else entry.summary_medium or entry.summary_short
            )
        if not base:
            return entry.title
        detail = ""
        if entry.entry_type in {"person", "event"} and entry.key_points and not simple_requested:
            detail = entry.key_points[1] if continuation and len(entry.key_points) > 1 else entry.key_points[0]
        elif entry.entry_type in {"concept", "object", "system", "process", "place"} and not simple_requested:
            if entry.key_points:
                detail = entry.key_points[1] if continuation and len(entry.key_points) > 1 else entry.key_points[0]
            elif entry.examples:
                detail = entry.examples[0]
        parts = [base]
        if detail and detail.lower() not in base.lower():
            parts.append(detail)
        analogy = ExplanationResponseBuilder._simple_analogy_sentence(entry=entry) if (simple_requested or strategy == "analogy") else None
        if analogy and analogy.lower() not in " ".join(parts).lower():
            parts.append(analogy)
        if strategy == "concrete_example":
            example = ExplanationResponseBuilder._example_sentence(entry=entry)
            if example and example.lower() not in " ".join(parts).lower():
                parts.append(example)
        relation_sentence = ExplanationResponseBuilder._relationship_sentence(
            entry=entry,
            base_text=" ".join(parts),
            related_connections=related_connections,
        )
        if relation_sentence:
            parts.append(relation_sentence)
        if strategy == "step_by_step":
            steps = [f"1. Core idea: {base.rstrip('.')}."]
            if detail and detail.lower() not in base.lower():
                steps.append(f"2. Next detail: {detail.rstrip('.')}.")
            elif analogy:
                steps.append(f"2. Picture it this way: {analogy.rstrip('.')}.")
            if relation_sentence:
                steps.append(f"3. Connection: {relation_sentence.rstrip('.')}.")
            return "\n".join(steps)
        if strategy == "analogy" and analogy:
            ordered = [base]
            if analogy.lower() not in base.lower():
                ordered.append(analogy)
            if detail and detail.lower() not in " ".join(ordered).lower():
                ordered.append(detail)
            if relation_sentence and relation_sentence.lower() not in " ".join(ordered).lower():
                ordered.append(relation_sentence)
            return " ".join(part.strip() for part in ordered if part and part.strip())
        return " ".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _example_sentence(*, entry: KnowledgeEntry) -> str | None:
        if entry.examples:
            return f"For example, {entry.examples[0].rstrip('.')}."
        title = entry.title.lower()
        if "entropy" in title:
            return "For example, when heat spreads through a room, the energy becomes more spread out and less concentrated in one useful place."
        return None

    @staticmethod
    def _wants_simple_explanation(prompt: str) -> bool:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        return any(
            phrase in normalized
            for phrase in (
                "simply",
                "simple ",
                "break it down",
                "break down",
                "in simple terms",
            )
        )

    @staticmethod
    def _wants_deeper_explanation(prompt: str) -> bool:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        return any(
            phrase in normalized
            for phrase in (
                "deeply",
                "go deeper",
                "in depth",
                "in detail",
                "more deeply",
            )
        )

    @staticmethod
    def _simple_analogy_sentence(*, entry: KnowledgeEntry) -> str | None:
        title = entry.title.lower()
        if "entropy" in title:
            return "A simple way to picture it is energy spreading out, like heat moving through a room until it is harder to use in one concentrated place."
        if entry.entry_type == "system":
            return f"You can think of {entry.title} as a set of connected parts that only make sense when they work together."
        if entry.entry_type == "process":
            return f"You can think of {entry.title} as a chain of steps, where each stage sets up the next one."
        return None

    @staticmethod
    def _relationship_sentence(
        *,
        entry: KnowledgeEntry,
        base_text: str,
        related_connections: list[tuple[str, str]],
    ) -> str | None:
        normalized_base = base_text.lower()
        for relation_type, target_title in related_connections:
            if target_title.lower() in normalized_base:
                continue
            if relation_type == "part_of":
                return f"It fits into the broader context of {target_title}."
            if relation_type == "related_to":
                return f"It's closely connected to {target_title}."
            if relation_type == "causes":
                return f"It directly contributes to {target_title}."
            if relation_type == "opposite_of":
                return f"It is often contrasted with {target_title}."
        if entry.related_topics:
            first_related = entry.related_topics[0]
            if first_related.lower() not in normalized_base:
                return f"It's also useful to connect it to {first_related}."
        return None

    @staticmethod
    def _clean_provider_text(provider_text: str | None) -> str | None:
        cleaned = " ".join(str(provider_text or "").split()).strip()
        return cleaned or None

    @classmethod
    def _generic_entity_answer(
        cls,
        *,
        prompt: str,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None,
    ) -> str | None:
        entity_list = list(entities or [])
        labels = {str(entity.get("label") or "").strip().lower() for entity in entity_list if isinstance(entity, dict)}
        if "person" in labels:
            return (
                "Sorry, I can tell this points to a person, but I don't have enough local knowledge on that person yet. "
                "If you want, ask a narrower question about who they were, what they did, or why they matter."
            )
        if "place" in labels:
            return (
                "Sorry, I can tell this points to a place, but I don't have enough local knowledge on that place yet. "
                "If you want, ask for its location, role, or why it matters."
            )
        if "event" in labels:
            return (
                "Sorry, I can tell this points to an event or historical development, but I don't have enough local knowledge on that event yet. "
                "If you want, ask about what happened, when it happened, or why it mattered."
            )
        if "object" in labels:
            return (
                "Sorry, I can tell this points to a physical object or system, but I don't have enough local knowledge on that specific object yet. "
                "If you want, ask what it is, how it works, or how it compares to something nearby."
            )
        if "concept" in labels:
            return (
                "Sorry, I don't know that because I don't have enough local knowledge on that specific concept yet. "
                "If you want, ask a narrower version and I can answer that directly."
            )
        return None

    @classmethod
    def _partial_match_answer(
        cls,
        *,
        prompt: str,
        lookup: KnowledgeLookupResult | None,
    ) -> str | None:
        if lookup is None or lookup.primary is None:
            return None
        subject = cls._subject_key(cls._normalize(prompt))
        anchor = cls._normalize(lookup.matched_alias or lookup.primary.title)
        subtopic = cls._extract_subtopic(subject=subject, anchor=anchor)
        if not subtopic:
            return None
        title = lookup.primary.title
        if lookup.primary.entry_type == "person":
            return (
                f"I know who {title} is, but I don't yet have enough local knowledge on {subtopic} to answer that confidently. "
                f"If you want, ask a narrower question about {subtopic}."
            )
        article = "the" if lookup.primary.entry_type in {"formula", "process", "system"} else ""
        subject_label = f"{article} {title}".strip()
        return (
            f"I can identify {subject_label}, but I don't have enough local knowledge on {subtopic} to answer confidently yet. "
            f"If you want, ask a narrower question about {subtopic}."
        )

    @staticmethod
    def _normalize(prompt: str) -> str:
        return PromptSurfaceBuilder.build(prompt).lookup_ready_text

    @classmethod
    def _subject_key(cls, normalized_prompt: str) -> str:
        return FocusResolutionSupport.subject_focus(normalized_prompt).focus

    @staticmethod
    def _clean_subject(normalized_prompt: str) -> str:
        return FocusResolutionSupport.subject_focus(normalized_prompt).focus

    @staticmethod
    def _extract_subtopic(*, subject: str, anchor: str) -> str:
        normalized_subject = " ".join(subject.split())
        if anchor and anchor in normalized_subject:
            tail = normalized_subject.replace(anchor, "", 1).strip()
        else:
            tail = normalized_subject
        tail = tail.lstrip("'s ").lstrip("s ").strip()
        connectors = ("on ", "about ", "regarding ", "and ")
        for connector in connectors:
            if tail.startswith(connector):
                tail = tail[len(connector):].strip()
        return tail

    @staticmethod
    def _apply_style(*, answer: str, interaction_style: str, answer_source: str) -> str:
        style = str(interaction_style or "default").strip().lower()
        if style == "conversational":
            style = "collab"
        if style == "direct":
            return answer
        if style == "collab":
            if answer_source == "lookup":
                partner_tail = " If you want, we can go one layer deeper on it."
                if answer.endswith(partner_tail.strip()):
                    return answer
                return f"{answer}{partner_tail}"
            if answer.lower().startswith(("right now", "honestly")):
                return answer
            return answer
        if style == "default" and answer_source == "lookup" and len(answer) < 280:
            return f"{answer} If you want, I can unpack it a bit further."
        return answer

    @staticmethod
    def _resolve_strategy(
        *,
        prompt: str,
        explanation_strategy: str | None,
        simple_requested: bool,
        deep_requested: bool,
    ) -> str:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if explanation_strategy:
            return str(explanation_strategy).strip() or "direct_definition"
        if "analogy" in normalized:
            return "analogy"
        if "step by step" in normalized or "break it down" in normalized:
            return "step_by_step"
        if "compare " in normalized or " vs " in normalized or " versus " in normalized:
            return "compare_contrast"
        if " in relation to " in normalized or " related to " in normalized:
            return "compare_contrast"
        if simple_requested:
            return "concrete_example"
        if deep_requested:
            return "direct_definition"
        return "direct_definition"
