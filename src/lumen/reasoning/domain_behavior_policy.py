from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DomainBehaviorProfile:
    intent_domain: str
    response_depth: str
    conversation_phase: str
    interaction_style: str
    structure_shape: str
    decomposition_level: str
    emphasize_tradeoffs: bool = False
    emphasize_options: bool = False
    emphasize_hypotheses: bool = False
    emphasize_validation: bool = False
    emphasize_reflection: bool = False
    append_next_steps: bool = False
    prefer_clarification: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_domain": self.intent_domain,
            "response_depth": self.response_depth,
            "conversation_phase": self.conversation_phase,
            "interaction_style": self.interaction_style,
            "structure_shape": self.structure_shape,
            "decomposition_level": self.decomposition_level,
            "emphasize_tradeoffs": self.emphasize_tradeoffs,
            "emphasize_options": self.emphasize_options,
            "emphasize_hypotheses": self.emphasize_hypotheses,
            "emphasize_validation": self.emphasize_validation,
            "emphasize_reflection": self.emphasize_reflection,
            "append_next_steps": self.append_next_steps,
            "prefer_clarification": self.prefer_clarification,
        }


@dataclass(slots=True)
class NextStepState:
    should_offer: bool
    suggestions: list[str] = field(default_factory=list)
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "should_offer": self.should_offer,
            "suggestions": list(self.suggestions),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ToolSuggestionState:
    should_suggest: bool
    suggestions: list[str] = field(default_factory=list)
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "should_suggest": self.should_suggest,
            "suggestions": list(self.suggestions),
            "rationale": self.rationale,
        }


class DomainBehaviorPolicy:
    """Map domain, depth, and phase into a unified response behavior profile."""

    def build_profile(
        self,
        *,
        route_mode: str,
        intent_domain: str,
        interaction_style: str,
        response_depth: str,
        conversation_phase: str,
        context_state: dict[str, Any] | None = None,
    ) -> DomainBehaviorProfile:
        shape = "guided"
        decomposition = "moderate"
        profile = DomainBehaviorProfile(
            intent_domain=intent_domain,
            response_depth=response_depth,
            conversation_phase=conversation_phase,
            interaction_style=interaction_style,
            structure_shape=shape,
            decomposition_level=decomposition,
        )
        if response_depth == "concise":
            profile.structure_shape = "compact"
            profile.decomposition_level = "light"
        elif response_depth == "deep":
            profile.structure_shape = "layered"
            profile.decomposition_level = "high"

        if intent_domain == "conversational":
            profile.structure_shape = "natural"
            profile.append_next_steps = conversation_phase in {"decision", "execution"}
        elif intent_domain == "learning_teaching":
            profile.structure_shape = "layered"
            profile.decomposition_level = "high" if response_depth != "concise" else "moderate"
        elif intent_domain == "decision_support":
            profile.structure_shape = "tradeoff_matrix"
            profile.emphasize_tradeoffs = True
            profile.emphasize_options = True
            profile.append_next_steps = True
        elif intent_domain == "problem_solving":
            profile.structure_shape = "decomposition"
            profile.emphasize_validation = True
            profile.append_next_steps = True
        elif intent_domain == "creative_ideation":
            profile.structure_shape = "expansive"
            profile.emphasize_options = True
            profile.append_next_steps = response_depth != "concise"
        elif intent_domain == "emotional_support_grounded":
            profile.structure_shape = "steady_support"
            profile.emphasize_reflection = True
            profile.prefer_clarification = conversation_phase == "intake"
        elif intent_domain == "reflection_self_analysis":
            profile.structure_shape = "reflective"
            profile.emphasize_reflection = True
            profile.prefer_clarification = True
        elif intent_domain == "planning_strategy":
            profile.structure_shape = "roadmap"
            profile.emphasize_options = True
            profile.append_next_steps = True
        elif intent_domain == "technical_engineering":
            profile.structure_shape = "systematic"
            profile.emphasize_validation = True
            profile.append_next_steps = True
        elif intent_domain == "research_investigation":
            profile.structure_shape = "evidence_led"
            profile.emphasize_hypotheses = True
            profile.emphasize_validation = True
            profile.append_next_steps = True

        if route_mode == "clarification":
            profile.prefer_clarification = True
            profile.append_next_steps = False
        return profile


class NextStepEngine:
    """Generate domain-aware actionable next steps and optional tool suggestions."""

    _NEXT_STEP_TEMPLATES: dict[str, tuple[str, ...]] = {
        "decision_support": (
            "Compare the top two options side by side.",
            "Simulate the most likely outcome for each option.",
            "Gather one missing fact that would change the decision.",
        ),
        "planning_strategy": (
            "Choose the first milestone and make it concrete.",
            "Identify the blocker most likely to delay execution.",
            "Turn the roadmap into the next three actions.",
        ),
        "research_investigation": (
            "Validate the strongest assumption with another source.",
            "Test the leading hypothesis against a competing explanation.",
            "List the evidence gap that still limits confidence.",
        ),
        "technical_engineering": (
            "Reproduce the issue with the smallest reliable test case.",
            "Inspect the subsystem with the highest failure risk first.",
            "Isolate the next module or dependency to verify.",
        ),
        "problem_solving": (
            "Name the tightest constraint before picking a fix.",
            "Test the highest-leverage solution first.",
            "Eliminate one likely failure point before expanding scope.",
        ),
        "creative_ideation": (
            "Pick one direction and expand it into three stronger variants.",
            "Combine the strongest elements into a single concept.",
            "Stress-test the idea against tone or audience fit.",
        ),
        "learning_teaching": (
            "Zoom in on the part that still feels fuzzy.",
            "Test the idea with a concrete example.",
            "Compare this concept to the closest familiar one.",
        ),
    }

    _TOOL_TEMPLATES: dict[str, tuple[str, ...]] = {
        "decision_support": ("Want me to build a structured comparison?", "Want me to simulate the scenarios?"),
        "planning_strategy": ("Want me to turn this into a step-by-step plan?",),
        "research_investigation": ("Want me to turn this into a structured report?",),
        "technical_engineering": ("Want me to map this into a debugging checklist?",),
        "problem_solving": ("Want me to break this into a tighter troubleshooting path?",),
    }

    def build_next_steps(
        self,
        *,
        behavior_profile: DomainBehaviorProfile,
        response: dict[str, Any],
        prompt: str,
    ) -> NextStepState:
        if not behavior_profile.append_next_steps:
            return NextStepState(
                should_offer=False,
                rationale="This domain/phase does not benefit from automatic next steps.",
            )
        suggestions = list(self._NEXT_STEP_TEMPLATES.get(behavior_profile.intent_domain, ()))
        if not suggestions:
            return NextStepState(
                should_offer=False,
                rationale="No domain-specific next-step template matched.",
            )
        return NextStepState(
            should_offer=True,
            suggestions=suggestions[: (1 if behavior_profile.response_depth == "concise" else 2)],
            rationale=f"Generated next steps for the {behavior_profile.intent_domain} domain.",
        )

    def build_tool_suggestions(
        self,
        *,
        behavior_profile: DomainBehaviorProfile,
        route_mode: str,
        prompt: str,
    ) -> ToolSuggestionState:
        if route_mode == "tool":
            return ToolSuggestionState(
                should_suggest=False,
                rationale="Tool route already selected.",
            )
        suggestions = list(self._TOOL_TEMPLATES.get(behavior_profile.intent_domain, ()))
        if not suggestions:
            return ToolSuggestionState(
                should_suggest=False,
                rationale="No natural tool suggestion for this domain.",
            )
        return ToolSuggestionState(
            should_suggest=True,
            suggestions=suggestions[:1],
            rationale=f"Tool suggestion is useful for the {behavior_profile.intent_domain} domain.",
        )
