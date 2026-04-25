from __future__ import annotations

from lumen.reasoning.pipeline_models import DialogueManagementResult, NLUExtraction
from lumen.routing.domain_router import DomainRoute


class DialogueManager:
    """Classifies dialogue state between NLU and later response stages."""

    EXPLORATORY_CUES = (
        "what do you think",
        "maybe",
        "could we",
        "brainstorm",
        "explore",
        "idea",
        "hypothesis",
    )
    SYNTHESIS_CUES = (
        "summarize",
        "recap",
        "synthesize",
        "where are we",
        "pull this together",
    )
    REORIENTATION_CUES = (
        "where are we now",
        "where were we",
        "what were we doing",
        "what still matters here",
        "what is still unresolved",
        "what's still unresolved",
        "what were we testing",
        "what are we testing",
    )
    CHALLENGE_CUES = (
        "but",
        "however",
        "actually",
        "i disagree",
        "challenge",
        "counterpoint",
    )
    BRANCH_CUES = (
        "what about",
        "another option",
        "alternatively",
        "or maybe",
        "branch",
    )
    REFINEMENT_CUES = (
        "refine",
        "tighten",
        "improve",
        "narrow",
        "revise",
        "adjust",
    )
    VALIDATION_CUES = (
        "that makes sense",
        "agreed",
        "confirmed",
        "yes, that fits",
        "validated",
    )
    HESITATION_CUES = (
        "not sure",
        "unsure",
        "i wonder",
        "maybe",
        "i'm not convinced",
        "hesitant",
    )
    PARKING_CUES = (
        "park that",
        "leave that for now",
        "come back later",
        "set that aside",
    )
    WRAP_UP_CUES = (
        "wrap this up",
        "that's enough",
        "we can stop here",
        "good for now",
        "done for now",
    )

    def manage(
        self,
        *,
        prompt: str,
        nlu: NLUExtraction,
        route: DomainRoute,
        interaction_count: int,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> DialogueManagementResult:
        normalized = " ".join(prompt.strip().lower().split())
        interaction_mode = self._interaction_mode(normalized=normalized, nlu=nlu, route=route)
        idea_state = self._idea_state(
            normalized=normalized,
            nlu=nlu,
            interaction_mode=interaction_mode,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        checkpoint_due, checkpoint_reason = self._checkpoint(
            normalized=normalized,
            interaction_mode=interaction_mode,
            idea_state=idea_state,
            interaction_count=interaction_count,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        response_strategy = self._response_strategy(
            interaction_mode=interaction_mode,
            idea_state=idea_state,
            checkpoint_due=checkpoint_due,
        )
        return DialogueManagementResult(
            interaction_mode=interaction_mode,
            idea_state=idea_state,
            response_strategy=response_strategy,
            synthesis_checkpoint_due=checkpoint_due,
            checkpoint_reason=checkpoint_reason,
        )

    def _interaction_mode(
        self,
        *,
        normalized: str,
        nlu: NLUExtraction,
        route: DomainRoute,
    ) -> str:
        if route.mode == "conversation":
            return "social"
        if nlu.ambiguity_flags or route.should_clarify():
            return "clarification"
        if any(cue in normalized for cue in self.REORIENTATION_CUES):
            return "synthesis"
        if any(cue in normalized for cue in self.SYNTHESIS_CUES):
            return "synthesis"
        if any(cue in normalized for cue in self.EXPLORATORY_CUES):
            if nlu.dominant_intent in {"planning", "research"}:
                return "hybrid"
            return "exploratory"
        return "analytical"

    def _idea_state(
        self,
        *,
        normalized: str,
        nlu: NLUExtraction,
        interaction_mode: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        if any(cue in normalized for cue in self.PARKING_CUES):
            return "parked"
        if any(cue in normalized for cue in self.VALIDATION_CUES):
            return "validated"
        if any(cue in normalized for cue in self.CHALLENGE_CUES):
            return "challenged"
        if any(cue in normalized for cue in self.BRANCH_CUES):
            return "branching"
        if any(cue in normalized for cue in self.REFINEMENT_CUES):
            return "refining"
        if nlu.ambiguity_flags:
            return "uncertain"
        if active_thread is None and not recent_interactions:
            return "introduced"
        if interaction_mode in {"exploratory", "hybrid"}:
            return "exploring"
        return "refining"

    @staticmethod
    def _checkpoint(
        *,
        normalized: str,
        interaction_mode: str,
        idea_state: str,
        interaction_count: int,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> tuple[bool, str | None]:
        next_turn_index = interaction_count + 1
        periodic_due = (
            interaction_mode in {"exploratory", "analytical", "hybrid", "synthesis"}
            and idea_state in {"exploring", "refining", "challenged", "branching"}
            and next_turn_index >= 4
            and next_turn_index % 4 == 0
        )
        if periodic_due:
            return True, "Extended multi-turn exploration suggests a synthesis checkpoint."

        state_pressure_due = DialogueManager._state_pressure_checkpoint(
            normalized=normalized,
            interaction_mode=interaction_mode,
            idea_state=idea_state,
            interaction_count=interaction_count,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        if state_pressure_due:
            return True, state_pressure_due
        return False, None

    @staticmethod
    def _state_pressure_checkpoint(
        *,
        normalized: str,
        interaction_mode: str,
        idea_state: str,
        interaction_count: int,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str | None:
        if interaction_mode not in {"exploratory", "analytical", "hybrid", "synthesis"}:
            return None

        same_mode_recent = 0
        for item in recent_interactions[:2]:
            if str(item.get("mode") or "").strip() in {"planning", "research"}:
                same_mode_recent += 1

        if idea_state in {"branching", "challenged"} and active_thread is not None and interaction_count >= 2:
            return "The thread has enough branching or challenge pressure to justify a synthesis checkpoint."

        if (
            active_thread is not None
            and same_mode_recent >= 1
            and idea_state in {"exploring", "refining"}
            and interaction_count >= 2
            and any(cue in normalized for cue in DialogueManager.EXPLORATORY_CUES + DialogueManager.REFINEMENT_CUES)
        ):
            return "The main thread is carrying enough live continuity to justify an earlier synthesis checkpoint."

        return None

    @staticmethod
    def _response_strategy(
        *,
        interaction_mode: str,
        idea_state: str,
        checkpoint_due: bool,
    ) -> str:
        if interaction_mode == "social":
            return "answer"
        if interaction_mode == "clarification" or idea_state == "uncertain":
            return "ask_question"
        if checkpoint_due or interaction_mode == "synthesis" or idea_state in {"validated", "parked"}:
            return "summarize"
        if idea_state == "challenged":
            return "challenge"
        if idea_state == "branching":
            return "expand"
        if interaction_mode in {"exploratory", "hybrid"}:
            return "expand"
        return "answer"
