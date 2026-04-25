from __future__ import annotations

from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    DialogueManagementResult,
    NLUExtraction,
    StateControlResult,
    ThoughtCheckpointSummary,
    ThoughtFramingResult,
)
from lumen.routing.domain_router import DomainRoute


class ThoughtFramingLayer:
    """Translates dialogue state into a response framing and research-question plan."""

    def frame(
        self,
        *,
        prompt: str,
        nlu: NLUExtraction,
        route: DomainRoute,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        state_control: StateControlResult | None,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> ThoughtFramingResult:
        normalized = " ".join(str(prompt).strip().lower().split())
        research_questions = self._research_questions(
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            nlu=nlu,
        )
        checkpoint_summary = self._checkpoint_summary(
            prompt=prompt,
            normalized=normalized,
            nlu=nlu,
            dialogue_management=dialogue_management,
            conversation_awareness=conversation_awareness,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
            research_questions=research_questions,
        )
        return ThoughtFramingResult(
            response_kind_label=self._response_kind_label(
                dialogue_management=dialogue_management,
                normalized=normalized,
            ),
            conversation_activity=self._conversation_activity(
                dialogue_management=dialogue_management,
                conversation_awareness=conversation_awareness,
                route=route,
                state_control=state_control,
                normalized=normalized,
            ),
            research_questions=research_questions,
            checkpoint_summary=checkpoint_summary,
            branch_return_hint=self._branch_return_hint(conversation_awareness=conversation_awareness),
        )

    @staticmethod
    def _response_kind_label(*, dialogue_management: DialogueManagementResult, normalized: str) -> str:
        if dialogue_management.interaction_mode == "social":
            return "lightweight_social"
        if ThoughtFramingLayer._is_reorientation_prompt(normalized):
            return "thread_reorientation"
        if dialogue_management.synthesis_checkpoint_due:
            return "checkpoint_summary"
        if dialogue_management.response_strategy == "ask_question":
            return "research_question"
        if dialogue_management.response_strategy == "challenge":
            return "challenge_response"
        if dialogue_management.response_strategy == "expand":
            return "exploratory_expansion"
        if dialogue_management.response_strategy == "summarize":
            return "synthesis_response"
        return "direct_answer"

    @staticmethod
    def _conversation_activity(
        *,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        route: DomainRoute,
        state_control: StateControlResult | None,
        normalized: str,
    ) -> str:
        if dialogue_management.interaction_mode == "social":
            return "opening or maintaining a lightweight social turn"
        if dialogue_management.interaction_mode == "clarification":
            return "reducing ambiguity before committing to a stronger answer"
        if ThoughtFramingLayer._is_reorientation_prompt(normalized):
            return "re-orienting the conversation around the live thread"
        if dialogue_management.interaction_mode == "synthesis":
            return "pulling the discussion into a synthesis checkpoint"
        if dialogue_management.interaction_mode == "exploratory":
            return "exploring a new idea and testing its shape"
        if (
            conversation_awareness.branch_state == "side_branch_open"
            and conversation_awareness.return_target
        ):
            return "testing a side branch while keeping the main thread in view"
        if (
            conversation_awareness.branch_state == "returning_to_main"
            and conversation_awareness.return_target
        ):
            return "rejoining the main thread and tightening the live line of work"
        if dialogue_management.interaction_mode == "hybrid":
            return "exploring an idea while keeping analytical structure"
        if route.mode == "planning":
            if state_control is not None and state_control.anti_spiral_active:
                return "slowing the discussion down so the next planning step is clearer and more grounded"
            return "turning the current idea into a more workable plan"
        if route.mode == "research":
            if state_control is not None and state_control.anti_spiral_active:
                return "stabilizing the discussion and separating what is known from what still needs evidence"
            return "analyzing and testing the current idea directly"
        return "working through the current idea directly"

    def _research_questions(
        self,
        *,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        nlu: NLUExtraction,
    ) -> list[str]:
        if dialogue_management.interaction_mode == "social":
            return []
        carried_question = str(conversation_awareness.live_unresolved_question or "").strip()
        if dialogue_management.interaction_mode == "clarification" or dialogue_management.idea_state == "uncertain":
            questions = [
                "What do you mean by the core assumption here?",
                "Do you want me to expand the idea or critique it?",
                "What would tell us this line is wrong?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if dialogue_management.idea_state == "challenged":
            questions = [
                "Which assumption should we test first?",
                "What would tell us this line doesn't hold?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if dialogue_management.idea_state == "branching":
            questions = [
                "Which branch feels most worth following first?",
                "Do you want me to expand one branch or compare them?",
            ]
            if conversation_awareness.return_target:
                questions.append("What do you want to carry back into the main thread once we test this branch?")
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if conversation_awareness.return_requested and conversation_awareness.return_target:
            questions = [
                "What do you want to resolve first as we return to the main thread?",
                "Which part of the main thread should we tighten next?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if dialogue_management.idea_state == "refining":
            questions = [
                "What assumption should we tighten first?",
                "What evidence would make this sharper?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if dialogue_management.interaction_mode in {"exploratory", "hybrid"}:
            questions = [
                "What assumption are we actually testing here?",
                "Have you considered the strongest alternative explanation?",
                "Are you trying to explain the idea, test it, or turn it into something usable?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        if nlu.dominant_intent in {"planning", "research"}:
            questions = [
                "What would count as a convincing next step here?",
            ]
            return self._merge_carried_question(carried_question=carried_question, questions=questions)
        return self._merge_carried_question(carried_question=carried_question, questions=[])

    @staticmethod
    def _merge_carried_question(*, carried_question: str, questions: list[str]) -> list[str]:
        if not carried_question:
            return questions
        normalized = carried_question.strip().lower()
        deduped = [question for question in questions if question.strip().lower() != normalized]
        return [carried_question, *deduped]

    @staticmethod
    def _branch_return_hint(*, conversation_awareness: ConversationAwarenessResult) -> str | None:
        if conversation_awareness.branch_state == "returning_to_main":
            return_target = str(conversation_awareness.return_target or "").strip()
            if not return_target:
                return "We're back on the main thread."
            return f"We're back on the main thread: {return_target}"
        if conversation_awareness.branch_state != "side_branch_open":
            return None
        return_target = str(conversation_awareness.return_target or "").strip()
        if not return_target:
            return "We can follow this branch without losing the main thread."
        return f"We can follow this branch, but the main thread to return to is: {return_target}"

    def _checkpoint_summary(
        self,
        *,
        prompt: str,
        normalized: str,
        nlu: NLUExtraction,
        dialogue_management: DialogueManagementResult,
        conversation_awareness: ConversationAwarenessResult,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        research_questions: list[str],
    ) -> ThoughtCheckpointSummary | None:
        if (
            not dialogue_management.synthesis_checkpoint_due
            and dialogue_management.interaction_mode != "synthesis"
            and not self._is_reorientation_prompt(normalized)
        ):
            return None
        open_questions = list(research_questions[:2])
        if not open_questions:
            open_questions = ["What should we resolve before moving forward?"]
        return ThoughtCheckpointSummary(
            current_direction=self._current_direction(
                prompt=prompt,
                nlu=nlu,
                active_thread=active_thread,
            ),
            strongest_point=self._strongest_point(
                prompt=prompt,
                recent_interactions=recent_interactions,
                active_thread=active_thread,
                nlu=nlu,
            ),
            weakest_point=self._weakest_point(
                dialogue_management=dialogue_management,
                reorientation=self._is_reorientation_prompt(normalized),
                live_unresolved_question=str(conversation_awareness.live_unresolved_question or "").strip(),
            ),
            open_questions=open_questions,
            next_step=self._next_step(
                dialogue_management=dialogue_management,
                reorientation=self._is_reorientation_prompt(normalized),
            ),
        )

    @staticmethod
    def _current_direction(
        *,
        prompt: str,
        nlu: NLUExtraction,
        active_thread: dict[str, object] | None,
    ) -> str:
        if active_thread is not None:
            summary = str(active_thread.get("thread_summary") or active_thread.get("summary") or "").strip()
            if summary:
                return summary
        if nlu.topic:
            return f"We are currently centered on {nlu.topic}."
        return f"We are currently working through: {prompt.strip()}."

    @staticmethod
    def _strongest_point(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        nlu: NLUExtraction,
    ) -> str:
        if recent_interactions:
            summary = str(recent_interactions[0].get("summary") or "").strip()
            if summary:
                return summary
        if active_thread is not None:
            objective = str(active_thread.get("objective") or "").strip()
            if objective:
                return objective
        if nlu.topic:
            return f"The main thread is staying anchored on {nlu.topic}."
        return f"The conversation still has a clear live direction around: {prompt.strip()}."

    @staticmethod
    def _weakest_point(
        *,
        dialogue_management: DialogueManagementResult,
        reorientation: bool,
        live_unresolved_question: str,
    ) -> str:
        if reorientation and live_unresolved_question:
            return f"The main unresolved point is still: {live_unresolved_question}"
        if reorientation:
            return "The main unresolved point still needs to be tightened before the thread can close cleanly."
        if dialogue_management.idea_state == "challenged":
            return "The current idea is under active challenge and still needs a cleaner stress test."
        if dialogue_management.idea_state == "branching":
            return "There are multiple live branches and no single line has fully won yet."
        if dialogue_management.idea_state == "uncertain":
            return "The main assumption is still too ambiguous to treat as settled."
        if dialogue_management.idea_state == "exploring":
            return "The central idea is still broad and has not been tightened enough yet."
        return "The current line of thought still needs sharper validation."

    @staticmethod
    def _next_step(*, dialogue_management: DialogueManagementResult, reorientation: bool) -> str:
        if reorientation:
            return "Re-anchor on the live thread, then either resolve the main open question or tighten the next step."
        if dialogue_management.response_strategy == "ask_question":
            return "Resolve the main open assumption before extending the answer."
        if dialogue_management.response_strategy == "challenge":
            return "Stress-test the strongest competing assumption next."
        if dialogue_management.response_strategy == "expand":
            return "Push the strongest branch one step further."
        if dialogue_management.response_strategy == "summarize":
            return "Capture the current synthesis, then decide whether to refine or branch."
        return "Turn the current direction into a more concrete answer or plan."

    @staticmethod
    def _is_reorientation_prompt(normalized: str) -> bool:
        cues = (
            "where are we now",
            "where were we",
            "what were we doing",
            "what still matters here",
            "what is still unresolved",
            "what's still unresolved",
            "what were we testing",
            "what are we testing",
        )
        return any(cue in normalized for cue in cues)
