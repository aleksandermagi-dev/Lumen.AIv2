from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.app.models import InteractionProfile
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    ConversationPhaseResult,
    DialogueManagementResult,
    EmpathyModelResult,
    IntentDomainResult,
    NLUExtraction,
    ResponseDepthResult,
    StateControlResult,
)
from lumen.routing.domain_router import DomainRoute


@dataclass(slots=True)
class IntentDomainPolicyInput:
    prompt: str
    route: DomainRoute
    nlu: NLUExtraction
    interaction_profile: InteractionProfile
    dialogue_management: DialogueManagementResult | None = None
    conversation_awareness: ConversationAwarenessResult | None = None
    empathy_model: EmpathyModelResult | None = None
    state_control: StateControlResult | None = None
    active_thread: dict[str, Any] | None = None
    recent_interactions: list[dict[str, Any]] = field(default_factory=list)


class IntentDomainPolicy:
    """Infer intent domain, adaptive depth, and conversation phase without replacing route authority."""

    _DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
        "learning_teaching": (
            "teach",
            "explain",
            "walk me through",
            "step by step",
            "break down",
            "how does",
            "how do",
            "why does",
        ),
        "decision_support": (
            "should i",
            "which should",
            "compare",
            "pros and cons",
            "tradeoff",
            "trade-off",
            "choose between",
            "better option",
        ),
        "problem_solving": (
            "solve",
            "fix",
            "figure out",
            "troubleshoot",
            "problem",
            "stuck",
            "blocked",
            "constraint",
        ),
        "creative_ideation": (
            "brainstorm",
            "ideas",
            "worldbuild",
            "world building",
            "story",
            "song",
            "music",
            "creative",
            "concepts",
        ),
        "reflection_self_analysis": (
            "why do i feel",
            "pattern",
            "what does this say about me",
            "self",
            "internally",
            "reflection",
            "why am i",
        ),
        "planning_strategy": (
            "plan",
            "roadmap",
            "prioritize",
            "strategy",
            "milestone",
            "timeline",
            "execution plan",
        ),
        "technical_engineering": (
            "debug",
            "architecture",
            "system design",
            "code",
            "refactor",
            "api",
            "implementation",
            "bug",
            "technical",
            "engineering",
        ),
        "research_investigation": (
            "research",
            "investigate",
            "evidence",
            "hypothesis",
            "findings",
            "source",
            "analyze",
            "deep dive",
        ),
        "emotional_support_grounded": (
            "overwhelmed",
            "anxious",
            "stressed",
            "sad",
            "upset",
            "angry",
            "lonely",
            "i feel",
            "i'm feeling",
        ),
    }

    def infer(
        self,
        *,
        prompt: str,
        route: DomainRoute,
        nlu: NLUExtraction,
        interaction_profile: InteractionProfile,
        dialogue_management: DialogueManagementResult | None = None,
        conversation_awareness: ConversationAwarenessResult | None = None,
        empathy_model: EmpathyModelResult | None = None,
        state_control: StateControlResult | None = None,
        active_thread: dict[str, Any] | None = None,
        recent_interactions: list[dict[str, Any]] | None = None,
    ) -> tuple[IntentDomainResult, ResponseDepthResult, ConversationPhaseResult]:
        policy_input = IntentDomainPolicyInput(
            prompt=prompt,
            route=route,
            nlu=nlu,
            interaction_profile=interaction_profile,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            empathy_model=empathy_model,
            state_control=state_control,
            active_thread=active_thread,
            recent_interactions=list(recent_interactions or []),
        )
        intent_domain = self._infer_domain(policy_input)
        response_depth = self._infer_depth(policy_input, intent_domain)
        conversation_phase = self._infer_phase(policy_input, intent_domain)
        return intent_domain, response_depth, conversation_phase

    def _infer_domain(self, policy_input: IntentDomainPolicyInput) -> IntentDomainResult:
        normalized = " ".join(policy_input.prompt.lower().split())
        signals: list[str] = []

        if self._matches_any(normalized, self._DOMAIN_KEYWORDS["reflection_self_analysis"]):
            signals.append("self_reflection_cues")
            return IntentDomainResult(
                domain="reflection_self_analysis",
                confidence=0.86,
                rationale="Detected self-analysis language in the prompt.",
                signals=signals,
            )
        if (
            policy_input.empathy_model is not None
            and policy_input.empathy_model.emotional_signal_detected
            and self._matches_any(normalized, self._DOMAIN_KEYWORDS["emotional_support_grounded"])
        ):
            signals.append("emotional_support_cues")
            return IntentDomainResult(
                domain="emotional_support_grounded",
                confidence=0.88,
                rationale="Detected emotional-support language with empathy signals.",
                signals=signals,
            )

        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            if domain in {"reflection_self_analysis", "emotional_support_grounded"}:
                continue
            if self._matches_any(normalized, keywords):
                signals.append(f"{domain}_keywords")
                return IntentDomainResult(
                    domain=domain,
                    confidence=0.82,
                    rationale=f"Detected language that matches the {domain} domain.",
                    signals=signals,
                )

        route_mode = str(policy_input.route.mode or "").strip().lower()
        if route_mode == "planning":
            if self._matches_any(normalized, self._DOMAIN_KEYWORDS["technical_engineering"]):
                return IntentDomainResult(
                    domain="technical_engineering",
                    confidence=0.78,
                    rationale="Planning route with technical/architecture cues.",
                    signals=["planning_route", "technical_cues"],
                )
            return IntentDomainResult(
                domain="planning_strategy",
                confidence=0.8,
                rationale="Planning route defaults to planning strategy behavior.",
                signals=["planning_route"],
            )
        if route_mode == "research":
            if self._matches_any(normalized, self._DOMAIN_KEYWORDS["learning_teaching"]):
                return IntentDomainResult(
                    domain="learning_teaching",
                    confidence=0.76,
                    rationale="Research route with explanatory teaching cues.",
                    signals=["research_route", "teaching_cues"],
                )
            return IntentDomainResult(
                domain="research_investigation",
                confidence=0.77,
                rationale="Research route defaults to investigation behavior.",
                signals=["research_route"],
            )
        if route_mode == "tool":
            return IntentDomainResult(
                domain="technical_engineering",
                confidence=0.74,
                rationale="Tool routes usually require technical execution framing.",
                signals=["tool_route"],
            )

        prior_domain = str((policy_input.active_thread or {}).get("intent_domain") or "").strip()
        if prior_domain:
            return IntentDomainResult(
                domain=prior_domain,
                confidence=0.68,
                rationale="Inherited the active thread domain for continuity.",
                signals=["active_thread_continuity"],
            )
        return IntentDomainResult(
            domain="conversational",
            confidence=0.65,
            rationale="No stronger domain signal was detected, so defaulting to conversational.",
            signals=["default_conversational"],
        )

    def _infer_depth(
        self,
        policy_input: IntentDomainPolicyInput,
        intent_domain: IntentDomainResult,
    ) -> ResponseDepthResult:
        normalized = " ".join(policy_input.prompt.lower().split())
        depth = "standard"
        rationale = "Default response depth."
        profile_depth = str(policy_input.interaction_profile.reasoning_depth or "").strip().lower()
        deep_cues = (
            "deep",
            "deeper",
            "thorough",
            "step by step",
            "walk me through",
            "layered",
            "detailed",
        )
        concise_cues = ("brief", "quick", "short", "concise", "one line")

        if self._matches_any(normalized, deep_cues) or profile_depth == "deep":
            depth = "deep"
            rationale = "Explicit or profile-based request for deeper reasoning."
        elif self._matches_any(normalized, concise_cues):
            depth = "concise"
            rationale = "Prompt asks for a concise answer."
        elif (
            policy_input.active_thread
            and intent_domain.domain == str(policy_input.active_thread.get("intent_domain") or "").strip()
            and str((policy_input.conversation_awareness or ConversationAwarenessResult()).conversation_momentum or "").strip()
            in {"building", "doubting"}
        ):
            depth = "deep"
            rationale = "Follow-up continuation on an active thread should deepen the answer."
        elif intent_domain.domain in {"technical_engineering", "research_investigation"} and len(normalized.split()) >= 10:
            depth = "deep"
            rationale = "Complex technical or research prompt benefits from layered depth."
        return ResponseDepthResult(level=depth, rationale=rationale)

    def _infer_phase(
        self,
        policy_input: IntentDomainPolicyInput,
        intent_domain: IntentDomainResult,
    ) -> ConversationPhaseResult:
        normalized = " ".join(policy_input.prompt.lower().split())
        awareness = policy_input.conversation_awareness or ConversationAwarenessResult()
        dialogue = policy_input.dialogue_management or DialogueManagementResult(
            interaction_mode="unknown",
            idea_state="introduced",
            response_strategy="answer",
        )

        if intent_domain.domain in {"reflection_self_analysis", "emotional_support_grounded"}:
            return ConversationPhaseResult(
                phase="reflection",
                rationale="Reflection and emotional-support domains map to the reflection phase.",
            )
        if any(token in normalized for token in ("should i", "which", "choose", "decide")):
            return ConversationPhaseResult(
                phase="decision",
                rationale="Detected explicit decision language.",
            )
        if intent_domain.domain in {"planning_strategy", "technical_engineering"} and any(
            token in normalized for token in ("implement", "do this", "first step", "execute", "run")
        ):
            return ConversationPhaseResult(
                phase="execution",
                rationale="Detected execution-oriented wording.",
            )
        if awareness.unresolved_thread_open or dialogue.idea_state in {"exploring", "branching"}:
            return ConversationPhaseResult(
                phase="exploration",
                rationale="Active unresolved thread or exploration state detected.",
            )
        if policy_input.active_thread is not None:
            return ConversationPhaseResult(
                phase="follow_up",
                rationale="Active thread indicates this is a follow-up turn.",
            )
        return ConversationPhaseResult(
            phase="intake",
            rationale="Fresh prompt with no stronger phase cues detected.",
        )

    @staticmethod
    def _matches_any(normalized_prompt: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in normalized_prompt for keyword in keywords)
