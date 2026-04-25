from __future__ import annotations

from dataclasses import replace

from lumen.routing.domain_router import DomainRoute


class RouteAdaptationPolicy:
    """Applies session-level softening or reinforcement to ambiguous heuristic routes."""

    @staticmethod
    def adapt_for_session_intent(
        route: DomainRoute,
        *,
        interaction_summary: dict[str, object],
    ) -> DomainRoute:
        decision = route.decision_summary or {}
        if not decision.get("ambiguous"):
            return route
        if route.source not in {"heuristic_planning", "heuristic_research", "active_thread_bias"}:
            return route

        dominant_intent, ratio = RouteAdaptationPolicy.session_dominant_intent(interaction_summary)
        if dominant_intent not in {"planning", "research"}:
            return route

        adjusted_evidence = list(route.evidence or [])
        if dominant_intent == route.mode:
            adjusted_evidence.append(
                {
                    "label": "session_intent_bias",
                    "detail": (
                        f"Recent session intent has been mostly {dominant_intent}, so this ambiguous route is being reinforced slightly."
                    ),
                    "weight": round(ratio, 2),
                }
            )
            return replace(
                route,
                confidence=min(1.0, route.confidence + 0.03),
                reason=(
                    f"{route.reason} Session intent continuity reinforced this route because recent turns have mostly behaved like {dominant_intent} requests."
                ),
                evidence=adjusted_evidence,
            )

        adjusted_evidence.append(
            {
                "label": "session_intent_caution",
                "detail": (
                    f"Recent session intent has been mostly {dominant_intent}, so this ambiguous route is being softened slightly."
                ),
                "weight": round(ratio, 2),
            }
        )
        return replace(
            route,
            confidence=max(0.0, route.confidence - 0.03),
            reason=(
                f"{route.reason} Session intent caution lowered route confidence slightly because recent turns have mostly behaved like {dominant_intent} requests."
            ),
            evidence=adjusted_evidence,
        )

    @staticmethod
    def adapt_for_retrieval_bias(
        route: DomainRoute,
        *,
        interaction_summary: dict[str, object],
    ) -> DomainRoute:
        decision = route.decision_summary or {}
        if not decision.get("ambiguous"):
            return route
        if route.source not in {"heuristic_planning", "heuristic_research"}:
            return route
        if not RouteAdaptationPolicy.session_retrieval_semantic_bias_high(interaction_summary):
            return route

        adjusted_evidence = list(route.evidence or [])
        adjusted_evidence.append(
            {
                "label": "retrieval_bias_caution",
                "detail": "Recent retrieval has been semantic-led, so ambiguous heuristic routing is being treated more cautiously.",
                "weight": 0.03,
            }
        )
        return replace(
            route,
            confidence=max(0.0, route.confidence - 0.03),
            reason=(
                f"{route.reason} Retrieval bias caution lowered route confidence slightly because recent session matches have been too semantic-led."
            ),
            evidence=adjusted_evidence,
        )

    @staticmethod
    def session_dominant_intent(interaction_summary: dict[str, object]) -> tuple[str | None, float]:
        interaction_count = int(interaction_summary.get("interaction_count", 0))
        if interaction_count < 3:
            return None, 0.0
        dominant_intent_counts = dict(interaction_summary.get("dominant_intent_counts") or {})
        scoped_counts = {
            intent: int(count)
            for intent, count in dominant_intent_counts.items()
            if intent in {"planning", "research"} and int(count) > 0
        }
        if not scoped_counts:
            return None, 0.0
        dominant_intent, dominant_count = max(scoped_counts.items(), key=lambda item: item[1])
        ratio = dominant_count / interaction_count
        if ratio < 0.6:
            return None, ratio
        return dominant_intent, ratio

    @staticmethod
    def session_retrieval_semantic_bias_high(interaction_summary: dict[str, object]) -> bool:
        observation_count = int(interaction_summary.get("retrieval_observation_count", 0))
        retrieval_lead_counts = dict(interaction_summary.get("retrieval_lead_counts") or {})
        semantic_count = int(retrieval_lead_counts.get("semantic", 0))
        keyword_count = int(retrieval_lead_counts.get("keyword", 0))
        blended_count = int(retrieval_lead_counts.get("blended", 0))
        return observation_count >= 3 and semantic_count > (keyword_count + blended_count)
