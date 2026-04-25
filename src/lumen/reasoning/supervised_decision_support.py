from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from lumen.reasoning.pipeline_models import IntentDomainResult
from lumen.reasoning.reasoning_language import ReasoningResponseLanguage
from lumen.reasoning.tool_threshold_gate import ToolThresholdDecision


SUPPORTED_INTENT_DOMAINS = {
    "conversational",
    "learning_teaching",
    "decision_support",
    "problem_solving",
    "creative_ideation",
    "emotional_support_grounded",
    "reflection_self_analysis",
    "planning_strategy",
    "technical_engineering",
    "research_investigation",
}


@dataclass(slots=True, frozen=True)
class SupervisedExample:
    surface: str
    label: str
    prompt_terms: tuple[str, ...] = ()
    route_modes: tuple[str, ...] = ()
    route_kinds: tuple[str, ...] = ()
    tool_id: str | None = None
    notes: str | None = None


@dataclass(slots=True, frozen=True)
class LearnedRecommendation:
    surface: str
    recommended_label: str
    confidence: float
    rationale: str
    features: dict[str, object] = field(default_factory=dict)
    applied: bool = False
    applied_reason: str | None = None

    def with_application(self, *, applied: bool, reason: str) -> "LearnedRecommendation":
        return LearnedRecommendation(
            surface=self.surface,
            recommended_label=self.recommended_label,
            confidence=self.confidence,
            rationale=self.rationale,
            features=dict(self.features),
            applied=applied,
            applied_reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "recommended_label": self.recommended_label,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "features": dict(self.features),
            "applied": self.applied,
            "applied_reason": self.applied_reason,
        }


class SupervisedDecisionSupport:
    """Optional local-first learned support that advises bounded decision surfaces."""

    def __init__(
        self,
        *,
        examples_by_surface: dict[str, Iterable[SupervisedExample]] | None = None,
    ) -> None:
        self._examples_by_surface: dict[str, tuple[SupervisedExample, ...]] = {
            str(surface): tuple(examples)
            for surface, examples in (examples_by_surface or {}).items()
            if examples
        }

    def is_enabled(self) -> bool:
        return bool(self._examples_by_surface)

    def empty_trace(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "enabled": self.is_enabled(),
            "surfaces_with_examples": sorted(self._examples_by_surface),
            "recommendations": {},
            "applied_surfaces": [],
            "deterministic_authority_preserved": True,
        }

    def record_recommendation(
        self,
        *,
        trace: dict[str, object] | None,
        recommendation: LearnedRecommendation | None,
    ) -> dict[str, object]:
        payload = dict(trace or self.empty_trace())
        payload.setdefault("schema_version", "1")
        payload.setdefault("enabled", self.is_enabled())
        payload.setdefault("surfaces_with_examples", sorted(self._examples_by_surface))
        payload.setdefault("recommendations", {})
        payload.setdefault("applied_surfaces", [])
        payload.setdefault("deterministic_authority_preserved", True)
        if recommendation is None:
            return payload
        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, dict):
            recommendations = {}
            payload["recommendations"] = recommendations
        recommendations[recommendation.surface] = recommendation.to_dict()
        if recommendation.applied:
            applied_surfaces = payload.get("applied_surfaces")
            if not isinstance(applied_surfaces, list):
                applied_surfaces = []
                payload["applied_surfaces"] = applied_surfaces
            if recommendation.surface not in applied_surfaces:
                applied_surfaces.append(recommendation.surface)
        return payload

    def assist_intent_domain(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        current: IntentDomainResult | None,
    ) -> tuple[IntentDomainResult | None, LearnedRecommendation | None]:
        recommendation = self._recommend_for_surface(
            surface="intent_domain_classification",
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
        )
        if recommendation is None or current is None:
            return current, recommendation
        if recommendation.recommended_label not in SUPPORTED_INTENT_DOMAINS:
            return current, recommendation.with_application(
                applied=False,
                reason="Recommended domain was outside the supported intent-domain set.",
            )
        if current.domain == recommendation.recommended_label:
            return current, recommendation.with_application(
                applied=False,
                reason="Deterministic intent-domain result already matches the learned recommendation.",
            )
        if float(current.confidence or 0.0) >= 0.72:
            return current, recommendation.with_application(
                applied=False,
                reason="Deterministic intent-domain confidence is already strong enough to keep authority local.",
            )
        if float(recommendation.confidence) < max(float(current.confidence or 0.0) + 0.08, 0.78):
            return current, recommendation.with_application(
                applied=False,
                reason="Learned recommendation was not strong enough to justify changing the deterministic domain result.",
            )
        assisted = IntentDomainResult(
            domain=recommendation.recommended_label,
            confidence=max(float(current.confidence or 0.0), float(recommendation.confidence)),
            rationale=(
                f"{current.rationale or ''} | Learned support favored "
                f"{recommendation.recommended_label} from matched local examples."
            ).strip(" |"),
            signals=list(current.signals) + ["learned_support"],
        )
        return assisted, recommendation.with_application(
            applied=True,
            reason="Applied as a bounded tie-break because deterministic domain confidence was still weak.",
        )

    def advise_route_decision(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        current_mode: str,
        current_kind: str,
    ) -> LearnedRecommendation | None:
        recommendation = self._recommend_for_surface(
            surface="route_recommendation_support",
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
        )
        if recommendation is None:
            return None
        current_label = f"{current_mode}:{current_kind}".strip(":")
        if recommendation.recommended_label == current_label:
            return recommendation.with_application(
                applied=False,
                reason="Learned support agrees with the deterministic route decision.",
            )
        return recommendation.with_application(
            applied=False,
            reason="Recorded for evaluation only; route authority stays deterministic in Phase 11.",
        )

    def advise_tool_decision(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        tool_id: str | None,
        current_decision: ToolThresholdDecision,
    ) -> LearnedRecommendation | None:
        recommendation = self._recommend_for_surface(
            surface="tool_use_decision_support",
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
            tool_id=tool_id,
        )
        if recommendation is None:
            return None
        expected = recommendation.recommended_label == "use_tool"
        current = bool(current_decision.should_use_tool)
        if current == expected:
            return recommendation.with_application(
                applied=False,
                reason="Learned support agrees with the deterministic tool-threshold decision.",
            )
        return recommendation.with_application(
            applied=False,
            reason="Recorded for traceability only; tool-threshold authority stays deterministic in Phase 5.",
        )

    def advise_response_style(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        current_structure: str,
    ) -> LearnedRecommendation | None:
        recommendation = self._recommend_for_surface(
            surface="response_style_selection",
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
        )
        if recommendation is None:
            return None
        if str(recommendation.recommended_label or "").strip() == str(current_structure or "").strip():
            return recommendation.with_application(
                applied=False,
                reason="Learned support agrees with the current deterministic response-structure selection.",
            )
        return recommendation.with_application(
            applied=False,
            reason="Recorded for evaluation only; response-style packaging stays deterministic in Phase 11.",
        )

    def assist_confidence_calibration(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        current_tier: str,
        current_posture: str,
        route_status: str,
        support_status: str,
    ) -> tuple[str, str, LearnedRecommendation | None]:
        recommendation = self._recommend_for_surface(
            surface="confidence_calibration_support",
            prompt=prompt,
            route_mode=route_mode,
            route_kind=route_kind,
        )
        normalized_tier = self._normalize_confidence_tier(current_tier)
        normalized_posture = self._normalize_confidence_posture(current_posture)
        if recommendation is None:
            return normalized_tier, normalized_posture, None
        recommended_posture = self._normalize_confidence_posture(recommendation.recommended_label)
        if recommended_posture is None:
            return normalized_tier, normalized_posture, recommendation.with_application(
                applied=False,
                reason="Recommended confidence posture was outside the bounded calibration set.",
            )
        if recommended_posture == normalized_posture:
            return normalized_tier, normalized_posture, recommendation.with_application(
                applied=False,
                reason="Learned support agrees with the current deterministic confidence posture.",
            )
        if self._posture_rank(recommended_posture) >= self._posture_rank(normalized_posture):
            return normalized_tier, normalized_posture, recommendation.with_application(
                applied=False,
                reason="Phase 11 confidence calibration only permits more cautious adjustments, never more confident ones.",
            )
        if float(recommendation.confidence) < 0.8:
            return normalized_tier, normalized_posture, recommendation.with_application(
                applied=False,
                reason="Learned confidence calibration was not strong enough to justify adjusting the deterministic posture.",
            )
        if route_status not in {"weakened", "under_tension", "unresolved"} and support_status not in {
            "moderately_supported",
            "insufficiently_grounded",
        }:
            return normalized_tier, normalized_posture, recommendation.with_application(
                applied=False,
                reason="Deterministic route/support signals were already strong enough to keep confidence calibration local.",
            )
        adjusted_tier = self._tier_from_posture(recommended_posture)
        return adjusted_tier, recommended_posture, recommendation.with_application(
            applied=True,
            reason="Applied as a bounded caution adjustment because learned support recommended a more conservative confidence posture under weak support.",
        )

    def _recommend_for_surface(
        self,
        *,
        surface: str,
        prompt: str,
        route_mode: str,
        route_kind: str,
        tool_id: str | None = None,
    ) -> LearnedRecommendation | None:
        examples = self._examples_by_surface.get(surface)
        if not examples:
            return None
        prompt_tokens = self._prompt_tokens(prompt)
        best: tuple[float, SupervisedExample] | None = None
        for example in examples:
            score = self._score_example(
                example=example,
                prompt_tokens=prompt_tokens,
                route_mode=route_mode,
                route_kind=route_kind,
                tool_id=tool_id,
            )
            if score <= 0.0:
                continue
            if best is None or score > best[0]:
                best = (score, example)
        if best is None:
            return None
        confidence = min(0.96, 0.45 + best[0] * 0.45)
        example = best[1]
        rationale = (
            f"Matched local supervised example support for {surface} using "
            f"route={route_mode}:{route_kind} and prompt-term overlap."
        )
        return LearnedRecommendation(
            surface=surface,
            recommended_label=example.label,
            confidence=confidence,
            rationale=rationale,
            features={
                "route_mode": route_mode,
                "route_kind": route_kind,
                "tool_id": tool_id,
                "matched_terms": [
                    term for term in example.prompt_terms
                    if term and term.lower() in prompt_tokens
                ],
                "example_notes": example.notes,
            },
        )

    @staticmethod
    def _score_example(
        *,
        example: SupervisedExample,
        prompt_tokens: set[str],
        route_mode: str,
        route_kind: str,
        tool_id: str | None,
    ) -> float:
        score = 0.0
        if example.route_modes and route_mode not in example.route_modes:
            return 0.0
        if example.route_kinds and route_kind not in example.route_kinds:
            return 0.0
        if example.tool_id and example.tool_id != tool_id:
            return 0.0
        if example.route_modes:
            score += 0.2
        if example.route_kinds:
            score += 0.15
        if example.tool_id:
            score += 0.15
        if example.prompt_terms:
            matched = [
                term for term in example.prompt_terms
                if term and term.lower() in prompt_tokens
            ]
            if not matched:
                return 0.0
            score += min(0.7, len(matched) / max(len(example.prompt_terms), 1))
        else:
            score += 0.3
        return score

    @staticmethod
    def _prompt_tokens(prompt: str) -> set[str]:
        normalized = [
            token.strip(".,!?():;[]{}\"'")
            for token in str(prompt or "").lower().split()
        ]
        return {token for token in normalized if token}

    @staticmethod
    def _normalize_confidence_tier(tier: str) -> str:
        normalized = str(tier or "").strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return "medium"

    @staticmethod
    def _normalize_confidence_posture(posture: str) -> str | None:
        normalized = str(posture or "").strip().lower()
        if normalized in {"strong", "supported", "tentative", "conflicted"}:
            return normalized
        return None

    @staticmethod
    def _posture_rank(posture: str | None) -> int:
        if posture == "strong":
            return 3
        if posture == "supported":
            return 2
        if posture in {"tentative", "conflicted"}:
            return 1
        return 0

    @staticmethod
    def _tier_from_posture(posture: str) -> str:
        tier = ReasoningResponseLanguage.language_confidence_tier(
            confidence_posture=posture,
        )
        return tier if tier in {"low", "medium", "high"} else "medium"
