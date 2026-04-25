from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.nlu.models import PromptUnderstanding
from lumen.reasoning.self_overview_surface_support import SelfOverviewSurfaceSupport
from lumen.reasoning.response_models import RouteMetadata
from lumen.nlu.follow_up_inventory import looks_like_reference_follow_up
from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.route_clarification import RouteClarificationPolicy
from lumen.routing.intent_signals import IntentSignalExtractor, PromptIntentSignals
from lumen.routing.route_models import RouteAnalysis, RouteCandidate, RouteComparison, RouteDecisionSummary, RouteEvidence


@dataclass(slots=True)
class DomainRoute:
    mode: str
    kind: str
    normalized_prompt: str
    confidence: float
    reason: str
    source: str = "classifier"
    evidence: list[dict[str, object]] | None = None
    comparison: dict[str, object] | None = None
    decision_summary: dict[str, object] | None = None

    def should_clarify(self) -> bool:
        return RouteClarificationPolicy.base_should_clarify(self)

    def to_metadata(self) -> RouteMetadata:
        strength = self._strength()
        caution = self._caution(strength)
        ambiguity = self._ambiguity_summary()
        return RouteMetadata(
            confidence=self.confidence,
            reason=self.reason,
            source=self.source,
            strength=strength,
            caution=caution,
            evidence=list(self.evidence or []),
            decision_summary=self.decision_summary,
            ambiguity=ambiguity,
        )

    def _strength(self) -> str:
        if self.source in {"manifest_alias", "explicit_compare", "explicit_summary", "explicit_planning"}:
            return "high"
        if self.source == "active_intent":
            return "medium"
        if self.source == "active_topic":
            return "medium"
        if self.source == "active_entities":
            return "medium"
        if self.source in {"active_thread", "recent_interaction"} and self.confidence >= 0.65:
            return "medium"
        if self.source == "active_thread_bias":
            return "medium"
        if self.confidence >= 0.75:
            return "medium"
        return "low"

    def _caution(self, strength: str) -> str | None:
        if self.source == "active_thread_bias":
            return "Route selection relied on active-thread bias because planning and research cues were close."
        if self.source == "active_intent":
            return "Route selection leaned on active intent continuity from the current session."
        if self.source == "active_topic":
            return "Route selection leaned on active topic continuity from the current session."
        if self.source == "active_entities":
            return "Route selection leaned on active entity continuity from the current session."
        if self._ambiguity_summary() is not None:
            return "Route selection is near a tie with another candidate and should be treated cautiously."
        if self.source == "fallback":
            return "Route selection fell back to a general research response because stronger intent cues were not found."
        if strength == "low":
            return "Route selection is heuristic and should be treated as provisional."
        return None

    def _ambiguity_summary(self) -> dict[str, object] | None:
        decision = self.decision_summary or {}
        if not decision.get("ambiguous"):
            return None
        summary = {
            "ambiguous": True,
        }
        if decision.get("ambiguity_reason"):
            summary["reason"] = decision.get("ambiguity_reason")
        alternatives = decision.get("alternatives") or []
        if alternatives:
            top_alternative = alternatives[0]
            candidate = top_alternative.get("candidate") or {}
            summary["top_alternative"] = {
                "mode": candidate.get("mode"),
                "kind": candidate.get("kind"),
                "source": candidate.get("source"),
                "confidence": candidate.get("confidence"),
            }
        return summary


class DomainRouter:
    """Chooses a lightweight response mode for a free-form prompt."""

    def __init__(self, capability_manager: CapabilityManager | None = None):
        self.capability_manager = capability_manager
        self.signal_extractor = IntentSignalExtractor()

    def route(
        self,
        prompt: str | PromptUnderstanding,
        *,
        recent_interactions: list[dict[str, Any]] | None = None,
        active_thread: dict[str, Any] | None = None,
    ) -> DomainRoute:
        analysis = self.analyze(
            prompt,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        selected_comparison = analysis.decision_summary.selected
        selected = selected_comparison.candidate
        return DomainRoute(
            mode=selected.mode,
            kind=selected.kind,
            normalized_prompt=analysis.normalized_prompt,
            confidence=selected.confidence,
            reason=selected.reason,
            source=selected.source,
            evidence=[item.to_dict() for item in selected.evidence],
            comparison=selected_comparison.to_dict(),
            decision_summary=analysis.decision_summary.to_dict(),
        )

    def analyze(
        self,
        prompt: str | PromptUnderstanding,
        *,
        recent_interactions: list[dict[str, Any]] | None = None,
        active_thread: dict[str, Any] | None = None,
    ) -> RouteAnalysis:
        # DomainRouter is the route authority; downstream layers may consume these signals but must not re-select the route.
        signals = self.signal_extractor.extract(prompt)
        normalized = signals.normalized_prompt
        candidates: list[RouteCandidate] = []

        conversation_candidate = self._conversation_candidate(signals)
        if conversation_candidate is not None:
            candidates.append(conversation_candidate)

        tool_candidate = self._tool_candidate(signals)
        if tool_candidate is not None:
            candidates.append(tool_candidate)

        follow_up_candidate = self._follow_up_candidate(
            signals,
            recent_interactions or [],
            active_thread=active_thread,
        )
        if follow_up_candidate is not None:
            candidates.append(follow_up_candidate)

        planning_candidate = self._planning_candidate(signals)
        if planning_candidate is not None:
            candidates.append(planning_candidate)

        research_candidate = self._research_candidate(signals)
        if research_candidate is not None:
            candidates.append(research_candidate)

        active_thread_bias = self._active_thread_bias_candidate(
            signals,
            active_thread=active_thread,
            candidates=candidates,
        )
        if active_thread_bias is not None:
            candidates.append(active_thread_bias)

        active_intent_candidate = self._active_intent_candidate(
            signals,
            active_thread=active_thread,
        )
        if active_intent_candidate is not None:
            candidates.append(active_intent_candidate)

        active_topic_candidate = self._active_topic_candidate(
            signals,
            active_thread=active_thread,
            candidates=candidates,
        )
        if active_topic_candidate is not None:
            candidates.append(active_topic_candidate)

        active_entities_candidate = self._active_entities_candidate(
            signals,
            active_thread=active_thread,
            candidates=candidates,
        )
        if active_entities_candidate is not None:
            candidates.append(active_entities_candidate)

        candidates.append(
            RouteCandidate(
                mode="research",
                kind="research.general",
                confidence=0.35,
                reason="Defaulted to research mode because no stronger planning or tool cues matched",
                source="fallback",
                evidence=[],
            )
        )

        comparisons = self._rank_candidates(
            candidates,
            signals=signals,
            active_thread=active_thread,
        )
        decision_summary = self._decision_summary(comparisons)
        return RouteAnalysis(
            normalized_prompt=normalized,
            signals={
                "detected_language": signals.detected_language,
                "normalized_topic": signals.normalized_topic,
                "dominant_intent": signals.dominant_intent,
                "intent_confidence": signals.intent_confidence,
                "extracted_entities": list(signals.extracted_entities),
                "explicit_planning_kind": signals.explicit_planning_kind,
                "explicit_greeting": signals.explicit_greeting,
                "explicit_summary": signals.explicit_summary,
                "explicit_comparison": signals.explicit_comparison,
                "follow_up": signals.follow_up,
                "planning_score": signals.planning_score,
                "research_score": signals.research_score,
                "action_score": signals.action_score,
                "answer_score": signals.answer_score,
                "starter_category_scores": dict(signals.starter_category_scores),
                "starter_categories": list(signals.starter_categories),
                "migration_hints": list(signals.migration_hints),
                "comparison_hints": list(signals.comparison_hints),
                "structure_subject": signals.structure_subject,
                "structure_predicate": signals.structure_predicate,
                "structure_object": signals.structure_object,
                "structure_modifiers": list(signals.structure_modifiers),
                "structure_completeness": signals.structure_completeness,
                "structure_ambiguity_flags": list(signals.structure_ambiguity_flags),
                "structure_fragmentation_markers": list(signals.structure_fragmentation_markers),
                "structure_reconstruction_confidence": signals.structure_reconstruction_confidence,
            },
            candidates=candidates,
            comparisons=comparisons,
            decision_summary=decision_summary,
        )

    def _tool_candidate(self, signals: PromptIntentSignals) -> RouteCandidate | None:
        if self.capability_manager is None:
            return None
        reconstructed_parts = tuple(str(signals.reconstructed_prompt or "").split(maxsplit=1))
        if len(signals.prompt_parts) < 2 and len(reconstructed_parts) < 2:
            capability = None
        elif len(signals.prompt_parts) >= 2:
            capability = self.capability_manager.find_by_command(
                signals.prompt_parts[0],
                signals.prompt_parts[1],
            )
        else:
            capability = self.capability_manager.find_by_command(
                reconstructed_parts[0],
                reconstructed_parts[1],
            )
        if capability is not None:
            return RouteCandidate(
                mode="tool",
                kind="tool.command_alias",
                confidence=0.95,
                reason=f"Matched manifest-declared tool alias '{signals.normalized_prompt}'",
                source="manifest_alias",
                evidence=[
                    RouteEvidence(
                        label="tool_alias",
                        detail=signals.normalized_prompt,
                        weight=0.95,
                    )
                ],
            )
        inferred_alias = self.capability_manager.infer_command_alias(
            str(signals.reconstructed_prompt or signals.normalized_prompt)
        )
        if inferred_alias is not None:
            _, alias = inferred_alias
            return RouteCandidate(
                mode="tool",
                kind="tool.command_alias",
                confidence=0.99,
                reason=f"Matched manifest-declared tool alias '{alias}' from a longer prompt.",
                source="manifest_alias",
                evidence=[
                    RouteEvidence(
                        label="tool_alias_inference",
                        detail=alias,
                        weight=0.99,
                    )
                ],
            )
        if (
            signals.explicit_social_kind is not None
            or signals.explicit_summary
            or signals.explicit_comparison
        ):
            return None
        signal_prompt = str(signals.reconstructed_prompt or signals.normalized_prompt)
        signal_match = self.capability_manager.infer_by_signals(signal_prompt)
        if signal_match is None or signal_match.confidence < 0.82:
            return None
        evidence = [
            RouteEvidence(label="tool_signal_keywords", detail=", ".join(signal_match.matched_keywords), weight=0.2)
            for _ in [0]
            if signal_match.matched_keywords
        ]
        if signal_match.matched_patterns:
            evidence.append(
                RouteEvidence(
                    label="tool_signal_patterns",
                    detail=", ".join(signal_match.matched_patterns),
                    weight=0.34,
                )
            )
        if signal_match.matched_intent_hints:
            evidence.append(
                RouteEvidence(
                    label="tool_signal_intent",
                    detail=", ".join(signal_match.matched_intent_hints),
                    weight=0.18,
                )
            )
        return RouteCandidate(
            mode="tool",
            kind="tool.hybrid_signal",
            confidence=signal_match.confidence,
            reason="Hybrid tool-routing signals matched a live bundle capability.",
            source="hybrid_signal",
            evidence=evidence,
        )

    def _conversation_candidate(self, signals: PromptIntentSignals) -> RouteCandidate | None:
        if self._looks_like_self_referential_prompt(signals.normalized_prompt):
            return RouteCandidate(
                mode="conversation",
                kind="conversation.self_overview",
                confidence=0.91,
                reason="Self-referential assistant prompt matched conversational self-overview routing.",
                source="self_overview",
                evidence=[
                    RouteEvidence(
                        label="self_prompt",
                        detail=signals.normalized_prompt,
                        weight=0.91,
                    )
                ],
            )
        if not signals.explicit_social_kind:
            return None
        return RouteCandidate(
            mode="conversation",
            kind=str(signals.explicit_social_kind),
            confidence=0.93,
            reason="Simple social turn matched conversational routing",
            source="explicit_greeting",
            evidence=[
                RouteEvidence(
                    label="social_turn",
                    detail=signals.normalized_prompt,
                    weight=0.93,
                )
            ],
        )

    def _follow_up_candidate(
        self,
        signals: PromptIntentSignals,
        recent_interactions: list[dict[str, Any]],
        *,
        active_thread: dict[str, Any] | None,
    ) -> RouteCandidate | None:
        if not signals.follow_up:
            return None

        strong_reference_follow_up = looks_like_reference_follow_up(signals.normalized_prompt)
        if not strong_reference_follow_up and self._has_strong_fresh_route_intent(signals):
            return None

        if active_thread is not None:
            return self._build_follow_up_candidate(
                signals.normalized_prompt,
                mode=str(active_thread.get("mode", "")).strip(),
                kind=str(active_thread.get("kind", "")).strip(),
                summary=str(active_thread.get("summary", "")).strip(),
                source_label="active session thread",
                source="active_thread",
                confidence=0.68 if strong_reference_follow_up else 0.58,
            )

        if not recent_interactions:
            return None

        previous = recent_interactions[0]
        return self._build_follow_up_candidate(
            signals.normalized_prompt,
            mode=str(previous.get("mode", "")).strip(),
            kind=str(previous.get("kind", "")).strip(),
            summary=str(previous.get("summary", "")).strip(),
            source_label="recent session interaction",
            source="recent_interaction",
            confidence=0.66 if strong_reference_follow_up else 0.56,
        )

    def _planning_candidate(self, signals: PromptIntentSignals) -> RouteCandidate | None:
        explicit_kind = signals.explicit_planning_kind
        if explicit_kind is not None:
            return RouteCandidate(
                mode="planning",
                kind=explicit_kind,
                confidence=0.88,
                reason="Explicit planning prompt matched planning-intent routing",
                source="explicit_planning",
                evidence=[
                    RouteEvidence(
                        label="planning_prefix",
                        detail=signals.normalized_prompt,
                        weight=0.88,
                    )
                ],
            )
        planning_score = signals.planning_score + signals.action_score
        research_score = signals.research_score + signals.answer_score
        if planning_score <= research_score or planning_score <= 0:
            return None
        kind = "planning.migration" if signals.migration_hints else "planning.architecture"
        evidence = [
            RouteEvidence(
                label="planning_score",
                detail=f"planning={planning_score}, research={research_score}",
                weight=float(planning_score),
            )
        ]
        if signals.action_score > 0:
            evidence.append(
                RouteEvidence(
                    label="action_intent",
                    detail=f"action_bias={signals.action_score}",
                    weight=float(signals.action_score),
                )
            )
        return RouteCandidate(
            mode="planning",
            kind=kind,
            confidence=min(0.9, 0.45 + (planning_score * 0.1)),
            reason=f"Planning cues matched with score {planning_score}",
            source="heuristic_planning",
            evidence=evidence,
        )

    def _research_candidate(self, signals: PromptIntentSignals) -> RouteCandidate | None:
        self_referential_prompt = self._looks_like_self_referential_prompt(signals.normalized_prompt)
        if signals.explicit_comparison:
            return RouteCandidate(
                mode="research",
                kind="research.comparison",
                confidence=0.82,
                reason="Explicit comparison prompt matched comparison-research routing",
                source="explicit_compare",
                evidence=[
                    RouteEvidence(
                        label="comparison_prefix",
                        detail="Prompt starts with 'compare'",
                        weight=0.82,
                    )
                ],
            )

        if self_referential_prompt:
            return None

        if signals.explicit_summary:
            return RouteCandidate(
                mode="research",
                kind="research.summary",
                confidence=0.8,
                reason="Explicit summary/explanation prompt matched summary-research routing",
                source="explicit_summary",
                evidence=[
                    RouteEvidence(
                        label="summary_prefix",
                        detail=signals.normalized_prompt,
                        weight=0.8,
                    )
                ],
            )

        research_score = signals.research_score + signals.answer_score
        if research_score <= 0:
            return None
        kind = "research.comparison" if signals.comparison_hints else (
            "research.summary"
            if signals.answer_score > 0
            or any(hint in signals.normalized_prompt for hint in ("summarize", "summary", "explain", "tell me about"))
            else "research.general"
        )
        evidence = [
            RouteEvidence(
                label="research_score",
                detail=f"research={research_score}",
                weight=float(research_score),
            )
        ]
        if signals.answer_score > 0:
            evidence.append(
                RouteEvidence(
                    label="answer_intent",
                    detail=f"answer_bias={signals.answer_score}",
                    weight=float(signals.answer_score),
                )
            )
        return RouteCandidate(
            mode="research",
            kind=kind,
            confidence=min(0.85, 0.4 + (research_score * 0.08)),
            reason=f"Research cues matched with score {research_score}",
            source="heuristic_research",
            evidence=evidence,
        )

    @staticmethod
    def _looks_like_self_referential_prompt(normalized_prompt: str) -> bool:
        if SelfOverviewSurfaceSupport.looks_like_self_referential_prompt(prompt=normalized_prompt):
            return True
        return normalized_prompt in {
            "tell me about yourself",
            "tell me more about yourself",
            "explain yourself",
            "what about you",
            "how about you",
        } or (
            normalized_prompt.startswith(("tell me about ", "explain "))
            and any(token in normalized_prompt for token in ("yourself", "you"))
        )

    def _active_thread_bias_candidate(
        self,
        signals: PromptIntentSignals,
        *,
        active_thread: dict[str, Any] | None,
        candidates: list[RouteCandidate],
    ) -> RouteCandidate | None:
        if active_thread is None:
            return None

        mode = str(active_thread.get("mode", "")).strip()
        kind = str(active_thread.get("kind", "")).strip()
        summary = str(active_thread.get("summary", "")).strip()
        if mode not in {"planning", "research"} or not kind:
            return None

        if self._tool_candidate(signals) is not None:
            return None
        if signals.follow_up:
            return None
        if signals.explicit_planning_kind is not None:
            return None
        if signals.explicit_summary:
            return None
        if signals.explicit_comparison:
            return None

        planning_score = signals.planning_score
        research_score = signals.research_score
        heuristic_candidates = [
            candidate
            for candidate in candidates
            if candidate.source in {"heuristic_planning", "heuristic_research"}
        ]
        if planning_score <= 0 or research_score <= 0:
            return None

        if abs(planning_score - research_score) > 1:
            return None

        top_confidence = max(
            (
                candidate.confidence
                for candidate in heuristic_candidates
                if candidate.mode == mode
            ),
            default=0.58,
        )

        reason = "Ambiguous prompt was biased toward the active session thread"
        if summary:
            reason += f" ({summary})"
        return RouteCandidate(
            mode=mode,
            kind=kind,
            confidence=min(0.86, top_confidence + 0.03),
            reason=reason,
            source="active_thread_bias",
            evidence=[
                RouteEvidence(
                    label="active_thread_bias",
                    detail=signals.normalized_prompt,
                    weight=min(0.86, top_confidence + 0.03),
                )
            ],
        )

    def _active_topic_candidate(
        self,
        signals: PromptIntentSignals,
        *,
        active_thread: dict[str, Any] | None,
        candidates: list[RouteCandidate],
    ) -> RouteCandidate | None:
        if active_thread is None:
            return None

        mode = str(active_thread.get("mode", "")).strip()
        kind = str(active_thread.get("kind", "")).strip()
        normalized_topic = str(active_thread.get("normalized_topic") or "").strip().lower()
        if mode not in {"planning", "research"} or not kind or not normalized_topic:
            return None

        if self._tool_candidate(signals) is not None:
            return None
        if signals.follow_up:
            return None
        if signals.explicit_planning_kind is not None:
            return None
        if signals.explicit_summary:
            return None
        if signals.explicit_comparison:
            return None

        prompt_tokens = self._meaningful_tokens(signals.normalized_prompt)
        topic_tokens = self._meaningful_tokens(normalized_topic)
        overlap = sorted(prompt_tokens & topic_tokens)
        if not overlap:
            return None

        heuristic_candidates = [
            candidate
            for candidate in candidates
            if candidate.source in {"heuristic_planning", "heuristic_research"} and candidate.mode == mode
        ]
        if heuristic_candidates:
            top_confidence = max(candidate.confidence for candidate in heuristic_candidates)
            confidence = min(0.84, top_confidence + 0.02)
            reason = (
                "Prompt overlapped the active session topic and reinforced the current route"
            )
        else:
            confidence = 0.62
            reason = (
                "Prompt overlapped the active session topic without stronger direct intent cues"
            )

        return RouteCandidate(
            mode=mode,
            kind=kind,
            confidence=confidence,
            reason=reason,
            source="active_topic",
            evidence=[
                RouteEvidence(
                    label="active_topic",
                    detail=", ".join(overlap[:3]),
                    weight=confidence,
                )
            ],
        )

    def _active_intent_candidate(
        self,
        signals: PromptIntentSignals,
        *,
        active_thread: dict[str, Any] | None,
    ) -> RouteCandidate | None:
        if active_thread is None:
            return None

        mode = str(active_thread.get("mode", "")).strip()
        kind = str(active_thread.get("kind", "")).strip()
        dominant_intent = str(active_thread.get("dominant_intent") or "").strip()
        if mode not in {"planning", "research"} or not kind or dominant_intent == "unknown":
            return None

        if self._tool_candidate(signals) is not None:
            return None
        if signals.follow_up:
            return None
        if signals.explicit_planning_kind is not None:
            return None
        if signals.explicit_summary:
            return None
        if signals.explicit_comparison:
            return None

        if dominant_intent == "planning" and signals.dominant_intent == "planning":
            return RouteCandidate(
                mode=mode,
                kind=kind,
                confidence=0.64,
                reason="Prompt intent aligned with the active session intent",
                source="active_intent",
                evidence=[
                    RouteEvidence(
                        label="active_intent",
                        detail="planning",
                        weight=0.64,
                    )
                ],
            )
        if dominant_intent == "research" and signals.dominant_intent == "research":
            return RouteCandidate(
                mode=mode,
                kind=kind,
                confidence=0.64,
                reason="Prompt intent aligned with the active session intent",
                source="active_intent",
                evidence=[
                    RouteEvidence(
                        label="active_intent",
                        detail="research",
                        weight=0.64,
                    )
                ],
            )
        return None

    def _active_entities_candidate(
        self,
        signals: PromptIntentSignals,
        *,
        active_thread: dict[str, Any] | None,
        candidates: list[RouteCandidate],
    ) -> RouteCandidate | None:
        if active_thread is None:
            return None

        mode = str(active_thread.get("mode", "")).strip()
        kind = str(active_thread.get("kind", "")).strip()
        if mode not in {"planning", "research"} or not kind:
            return None

        if self._tool_candidate(signals) is not None:
            return None
        if signals.follow_up:
            return None
        if signals.explicit_planning_kind is not None:
            return None
        if signals.explicit_summary:
            return None
        if signals.explicit_comparison:
            return None

        active_entities = {
            str(entity.get("value")).strip().lower()
            for entity in (active_thread.get("extracted_entities") or [])
            if isinstance(entity, dict) and entity.get("value") is not None
        }
        signal_entities = {
            str(entity.get("value")).strip().lower()
            for entity in signals.extracted_entities
            if isinstance(entity, dict) and entity.get("value") is not None
        }
        overlap = sorted(active_entities & signal_entities)
        if not overlap:
            return None

        heuristic_candidates = [
            candidate
            for candidate in candidates
            if candidate.source in {"heuristic_planning", "heuristic_research"} and candidate.mode == mode
        ]
        if heuristic_candidates:
            top_confidence = max(candidate.confidence for candidate in heuristic_candidates)
            confidence = min(0.83, top_confidence + 0.01)
            reason = "Prompt reused entities from the active session and reinforced the current route"
        else:
            confidence = 0.61
            reason = "Prompt reused entities from the active session without stronger direct intent cues"

        return RouteCandidate(
            mode=mode,
            kind=kind,
            confidence=confidence,
            reason=reason,
            source="active_entities",
            evidence=[
                RouteEvidence(
                    label="active_entities",
                    detail=", ".join(overlap[:3]),
                    weight=confidence,
                )
            ],
        )

    @staticmethod
    def _build_follow_up_candidate(
        normalized_prompt: str,
        *,
        mode: str,
        kind: str,
        summary: str,
        source_label: str,
        source: str,
        confidence: float,
    ) -> RouteCandidate | None:
        if not mode or not kind:
            return None
        reason = f"Follow-up prompt matched {source_label}"
        if summary:
            reason += f" ({summary})"
        return RouteCandidate(
            mode=mode,
            kind=kind,
            confidence=confidence,
            reason=reason,
            source=source,
            evidence=[
                RouteEvidence(
                    label="follow_up",
                    detail=normalized_prompt,
                    weight=confidence,
                )
            ],
        )

    def _rank_candidates(
        self,
        candidates: list[RouteCandidate],
        *,
        signals: PromptIntentSignals,
        active_thread: dict[str, Any] | None,
    ) -> list[RouteComparison]:
        source_priority = {
            "manifest_alias": 5,
            "explicit_greeting": 4,
            "self_overview": 4,
            "explicit_compare": 4,
            "explicit_summary": 4,
            "explicit_planning": 4,
            "active_thread": 3,
            "active_thread_bias": 3,
            "active_intent": 3,
            "active_topic": 3,
            "active_entities": 3,
            "recent_interaction": 3,
            "heuristic_planning": 2,
            "heuristic_research": 2,
            "hybrid_signal": 3,
            "fallback": 1,
        }
        priority = {
            "tool": 3,
            "conversation": 2,
            "planning": 2,
            "research": 1,
        }
        comparisons = [
            RouteComparison(
                candidate=item,
                source_priority=source_priority.get(item.source, 0),
                mode_priority=priority.get(item.mode, 0),
                semantic_bonus=self._semantic_bonus(
                    item,
                    signals=signals,
                    active_thread=active_thread,
                ),
                intent_weight=self._intent_weight(item, signals=signals),
                context_decay=self._context_decay(item, signals=signals, active_thread=active_thread),
                normalized_score=0.0,
            )
            for item in candidates
        ]
        comparisons = [
            RouteComparison(
                candidate=item.candidate,
                source_priority=item.source_priority,
                mode_priority=item.mode_priority,
                semantic_bonus=item.semantic_bonus,
                intent_weight=item.intent_weight,
                context_decay=item.context_decay,
                normalized_score=self._normalized_route_score(item),
            )
            for item in comparisons
        ]
        return sorted(
            comparisons,
            key=lambda item: item.rank_tuple,
            reverse=True,
        )

    def _semantic_bonus(
        self,
        candidate: RouteCandidate,
        *,
        signals: PromptIntentSignals,
        active_thread: dict[str, Any] | None,
    ) -> float:
        bonus = 0.0
        entity_values = {
            str(entity.get("value")).strip().lower()
            for entity in signals.extracted_entities
            if isinstance(entity, dict) and entity.get("value") is not None
        }

        # Exact manifest aliases should win over generic summary phrasing.
        # This keeps explicit tool commands like "describe data" from
        # collapsing back into research.summary just because they are also
        # explanation-shaped in plain English.
        if candidate.mode == "tool" and candidate.source == "manifest_alias":
            bonus += 0.06

        if candidate.mode == "planning" and signals.dominant_intent == "planning":
            bonus += min(0.2, 0.08 + (signals.intent_confidence * 0.12))
        elif candidate.mode == "research" and signals.dominant_intent == "research":
            bonus += min(0.2, 0.08 + (signals.intent_confidence * 0.12))

        if candidate.kind == "planning.migration" and signals.migration_hints:
            bonus += 0.08
        if candidate.kind == "planning.architecture" and "architecture" in signals.normalized_prompt:
            bonus += 0.08
        if candidate.kind == "research.comparison" and signals.comparison_hints:
            bonus += 0.08
        if candidate.kind == "research.summary" and signals.explicit_summary:
            bonus += 0.08

        if active_thread is not None:
            active_mode = str(active_thread.get("mode", "")).strip()
            active_topic = str(active_thread.get("normalized_topic") or "").strip().lower()
            active_intent = str(active_thread.get("dominant_intent") or "").strip().lower()
            active_entities = {
                str(entity.get("value")).strip().lower()
                for entity in (active_thread.get("extracted_entities") or [])
                if isinstance(entity, dict) and entity.get("value") is not None
            }
            if candidate.mode == active_mode and signals.dominant_intent and signals.dominant_intent == active_intent:
                bonus += min(0.08, 0.02 + (signals.intent_confidence * 0.04))
            if candidate.mode == active_mode and signals.normalized_topic and active_topic:
                topic_overlap = self._meaningful_tokens(signals.normalized_topic) & self._meaningful_tokens(active_topic)
                if topic_overlap:
                    bonus += min(0.12, 0.04 * len(topic_overlap))
            if candidate.mode == active_mode and entity_values and active_entities:
                entity_overlap = entity_values & active_entities
                if entity_overlap:
                    bonus += min(0.12, 0.06 * len(entity_overlap))

        return round(bonus, 3)

    @staticmethod
    def _intent_weight(
        candidate: RouteCandidate,
        *,
        signals: PromptIntentSignals,
    ) -> float:
        weight = 0.0
        social_override = signals.explicit_social_kind is not None and not DomainRouter._has_strong_explicit_task_intent(signals)
        explanatory_override = DomainRouter._has_strong_explanatory_intent(signals)
        if candidate.mode == "conversation" and signals.explicit_social_kind is not None:
            weight += 0.26 if social_override else 0.16
        if candidate.source in {"active_thread", "recent_interaction"} and signals.follow_up:
            weight += 0.18
        if candidate.mode == "research" and candidate.kind == "research.summary":
            if signals.explicit_summary:
                weight += 0.14
            elif signals.answer_score > 0:
                weight += 0.08
            if explanatory_override:
                weight += 0.08
        elif candidate.mode == "research" and signals.answer_score > 0:
            weight += 0.04
        elif candidate.mode == "planning" and signals.planning_score > signals.research_score:
            weight += 0.04
        return round(weight, 3)

    @staticmethod
    def _context_decay(
        candidate: RouteCandidate,
        *,
        signals: PromptIntentSignals,
        active_thread: dict[str, Any] | None,
    ) -> float:
        if active_thread is None:
            return 0.0
        if candidate.source not in {"active_thread", "active_thread_bias", "active_intent", "active_topic", "active_entities"}:
            return 0.0

        decay = 0.0
        social_override = signals.explicit_social_kind is not None and not DomainRouter._has_strong_explicit_task_intent(signals)
        explanatory_override = DomainRouter._has_strong_explanatory_intent(signals)
        strong_reference_follow_up = signals.follow_up and looks_like_reference_follow_up(signals.normalized_prompt)

        if social_override and candidate.mode != "conversation":
            decay += 0.24 if candidate.source.startswith("active_") else 0.14
        elif signals.explicit_social_kind is not None:
            decay += 0.2

        if signals.follow_up and candidate.source not in {"active_thread", "recent_interaction"}:
            decay += 0.16 if candidate.source.startswith("active_") else 0.08

        if explanatory_override and candidate.mode == "planning" and not strong_reference_follow_up:
            decay += 0.14 if candidate.source.startswith("active_") else 0.08
        if explanatory_override and candidate.mode == "research" and candidate.kind != "research.summary":
            decay += 0.06

        if (signals.explicit_summary or signals.answer_score > 0) and not strong_reference_follow_up:
            if candidate.mode != "research":
                decay += 0.1
        if signals.explicit_planning_kind is not None or signals.planning_score > signals.research_score:
            if candidate.mode != "planning":
                decay += 0.1
        if not signals.follow_up and signals.dominant_intent == "unknown":
            decay += 0.04
        return round(decay, 3)

    @staticmethod
    def _has_strong_explicit_task_intent(signals: PromptIntentSignals) -> bool:
        if signals.explicit_planning_kind is not None:
            return True
        if signals.explicit_comparison or signals.explicit_summary:
            return True
        if signals.planning_score >= 2:
            return True
        return signals.answer_score >= 2 and signals.research_score >= 2

    @staticmethod
    def _has_strong_explanatory_intent(signals: PromptIntentSignals) -> bool:
        if signals.explicit_summary:
            return True
        if signals.explicit_comparison:
            return False
        entity_labels = {
            str(entity.get("label") or "").strip().lower()
            for entity in signals.extracted_entities
            if isinstance(entity, dict)
        }
        explanatory_entities = {"person", "place", "event", "concept", "object", "formula", "system", "process"}
        if entity_labels & explanatory_entities and signals.answer_score >= 2:
            return True
        if signals.answer_score >= 3 and signals.research_score >= 2:
            return True
        return "science" in signals.starter_categories and signals.answer_score >= 2

    @staticmethod
    def _has_strong_fresh_route_intent(signals: PromptIntentSignals) -> bool:
        if signals.explicit_social_kind is not None and not DomainRouter._has_strong_explicit_task_intent(signals):
            return True
        if signals.explicit_planning_kind is not None:
            return True
        if signals.explicit_summary or signals.explicit_comparison:
            return True
        if DomainRouter._has_strong_explanatory_intent(signals):
            return True
        if signals.dominant_intent in {"planning", "research"} and signals.intent_confidence >= 0.8:
            return True
        return False

    @staticmethod
    def _normalized_route_score(comparison: RouteComparison) -> float:
        source_component = comparison.source_priority * 0.25
        confidence_component = comparison.candidate.confidence
        semantic_component = comparison.semantic_bonus
        intent_component = comparison.intent_weight
        decay_component = comparison.context_decay
        mode_component = comparison.mode_priority * 0.02
        return round(
            source_component + confidence_component + semantic_component + intent_component + mode_component - decay_component,
            4,
        )

    @staticmethod
    def _decision_summary(comparisons: list[RouteComparison]) -> RouteDecisionSummary:
        selected = comparisons[0]
        alternatives = comparisons[1:3]
        ambiguous = False
        ambiguity_reason = None
        if alternatives:
            top_alternative = alternatives[0]
            close_total_score = abs(selected.normalized_score - top_alternative.normalized_score) <= 0.1
            close_source_priority = abs(selected.source_priority - top_alternative.source_priority) <= 1
            close_semantic_bonus = abs(selected.semantic_bonus - top_alternative.semantic_bonus) <= 0.08
            close_confidence = abs(selected.candidate.confidence - top_alternative.candidate.confidence) <= 0.08
            if selected.candidate.source == "active_thread_bias":
                ambiguous = True
                ambiguity_reason = (
                    "Route selection relied on active-thread bias after close competing cues."
                )
            elif close_total_score and close_source_priority and close_confidence and close_semantic_bonus:
                ambiguous = True
                ambiguity_reason = (
                    "Top route candidates had very similar normalized score, confidence, semantic reinforcement, and source priority."
                )
        return RouteDecisionSummary(
            selected=selected,
            alternatives=alternatives,
            ambiguous=ambiguous,
            ambiguity_reason=ambiguity_reason,
        )

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        stopwords = {"the", "a", "an", "for", "of", "and", "to", "with", "that", "this", "it"}
        return {
            token
            for token in text.split()
            if token not in stopwords and len(token) > 2
        }
