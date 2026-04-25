from __future__ import annotations

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.pipeline_models import ConversationAwarenessResult, DialogueManagementResult


class ConversationAwarenessLayer:
    """Tracks lightweight live-thread awareness between dialogue state and response shaping."""

    CHALLENGE_CUES = (
        "but",
        "however",
        "actually",
        "i disagree",
        "challenge",
        "counterpoint",
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
    FOLLOW_THROUGH_CUES = (
        "keep going",
        "go on",
        "continue",
        "what else",
        "and then",
        "go deeper",
        "expand that",
        "what about that",
    )
    WRAP_UP_CUES = (
        "wrap this up",
        "that's enough",
        "we can stop here",
        "good for now",
        "done for now",
    )
    RETURN_CUES = (
        "go back",
        "back to the main",
        "back to the main thread",
        "back to the main idea",
        "return to the main",
        "return to the earlier thread",
        "return to the main thread",
        "pick back up",
        "continue the main thread",
    )

    def assess(
        self,
        *,
        prompt: str,
        dialogue_management: DialogueManagementResult,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> ConversationAwarenessResult:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        recent_intent_pattern = self._recent_intent_pattern(
            normalized=normalized,
            recent_interactions=recent_interactions,
        )
        conversation_momentum = self._conversation_momentum(
            normalized=normalized,
            interaction_mode=dialogue_management.interaction_mode,
            idea_state=dialogue_management.idea_state,
            recent_intent_pattern=recent_intent_pattern,
        )
        unresolved_thread_open, unresolved_thread_reason = self._unresolved_thread(
            interaction_mode=dialogue_management.interaction_mode,
            idea_state=dialogue_management.idea_state,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        )
        live_unresolved_question = self._live_unresolved_question(
            interaction_mode=dialogue_management.interaction_mode,
            recent_intent_pattern=recent_intent_pattern,
            unresolved_thread_open=unresolved_thread_open,
            recent_interactions=recent_interactions,
        )
        return_requested = self._return_requested(
            normalized=normalized,
            active_thread=active_thread,
        )
        branch_state, return_target = self._branch_state(
            idea_state=dialogue_management.idea_state,
            return_requested=return_requested,
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        )
        adaptive_posture = self._adaptive_posture(
            interaction_mode=dialogue_management.interaction_mode,
            idea_state=dialogue_management.idea_state,
            response_strategy=dialogue_management.response_strategy,
            recent_intent_pattern=recent_intent_pattern,
            conversation_momentum=conversation_momentum,
            unresolved_thread_open=unresolved_thread_open,
        )
        return ConversationAwarenessResult(
            recent_intent_pattern=recent_intent_pattern,
            conversation_momentum=conversation_momentum,
            unresolved_thread_open=unresolved_thread_open,
            unresolved_thread_reason=unresolved_thread_reason,
            live_unresolved_question=live_unresolved_question,
            branch_state=branch_state,
            return_target=return_target,
            return_requested=return_requested,
            adaptive_posture=adaptive_posture,
        )

    def _return_requested(
        self,
        *,
        normalized: str,
        active_thread: dict[str, object] | None,
    ) -> bool:
        if active_thread is None:
            return False
        return any(cue in normalized for cue in self.RETURN_CUES)

    def _recent_intent_pattern(
        self,
        *,
        normalized: str,
        recent_interactions: list[dict[str, object]],
    ) -> str:
        if any(cue in normalized for cue in self.FOLLOW_THROUGH_CUES):
            return "following_through"
        if any(cue in normalized for cue in self.HESITATION_CUES):
            return "hesitating"
        if any(cue in normalized for cue in self.CHALLENGE_CUES):
            return "disagreeing"
        if any(cue in normalized for cue in self.VALIDATION_CUES):
            return "agreeing"
        if "?" in normalized or normalized.startswith(("what", "why", "how", "can", "should", "do ")):
            return "asking_questions"
        for item in recent_interactions[:2]:
            mode = str(item.get("mode") or "").strip()
            kind = str(item.get("kind") or "").strip()
            summary = str(item.get("summary") or "").lower().strip()
            if mode == "clarification" or "question" in kind or "question" in summary:
                return "asking_questions"
        return "building"

    def _conversation_momentum(
        self,
        *,
        normalized: str,
        interaction_mode: str,
        idea_state: str,
        recent_intent_pattern: str,
    ) -> str:
        if interaction_mode == "social":
            return "chatting"
        if any(cue in normalized for cue in self.WRAP_UP_CUES) or idea_state in {"validated", "parked"}:
            return "wrapping_up"
        if recent_intent_pattern == "following_through":
            return "building"
        if recent_intent_pattern == "hesitating" or idea_state == "uncertain":
            return "doubting"
        if idea_state in {"exploring", "refining", "branching", "challenged"} or interaction_mode in {
            "analytical",
            "exploratory",
            "hybrid",
            "synthesis",
        }:
            return "building"
        return "steady"

    @staticmethod
    def _unresolved_thread(
        *,
        interaction_mode: str,
        idea_state: str,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> tuple[bool, str | None]:
        if interaction_mode == "social":
            return False, None
        if idea_state == "uncertain":
            return True, "A core assumption is still unresolved."
        if idea_state == "branching":
            return True, "Multiple branches are still open."
        if idea_state == "challenged":
            return True, "The main line is still under challenge."
        if recent_interactions:
            last_mode = str(recent_interactions[0].get("mode") or "").strip()
            if last_mode == "clarification":
                return True, "A recent clarification request is still unresolved."
        if active_thread is not None and idea_state not in {"validated", "parked"}:
            return True, "The current thread still has live work in it."
        return False, None

    @staticmethod
    def _branch_state(
        *,
        idea_state: str,
        return_requested: bool,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        if return_requested:
            return_target = None
            if active_thread is not None:
                return_target = str(
                    active_thread.get("thread_summary")
                    or active_thread.get("objective")
                    or active_thread.get("summary")
                    or ""
                ).strip() or None
            if return_target is None and recent_interactions:
                return_target = str(recent_interactions[0].get("summary") or "").strip() or None
            return "returning_to_main", return_target
        if idea_state != "branching":
            return None, None
        return_target = None
        if active_thread is not None:
            return_target = str(
                active_thread.get("thread_summary")
                or active_thread.get("objective")
                or active_thread.get("summary")
                or ""
            ).strip() or None
        if return_target is None and recent_interactions:
            return_target = str(recent_interactions[0].get("summary") or "").strip() or None
        return "side_branch_open", return_target

    @staticmethod
    def _live_unresolved_question(
        *,
        interaction_mode: str,
        recent_intent_pattern: str,
        unresolved_thread_open: bool,
        recent_interactions: list[dict[str, object]],
    ) -> str | None:
        if interaction_mode == "social" or not unresolved_thread_open:
            return None
        if recent_intent_pattern in {"agreeing", "disagreeing"}:
            return None
        for item in recent_interactions[:3]:
            mode = str(item.get("mode") or "").strip()
            if mode == "clarification":
                continue
            candidate = ConversationAwarenessLayer._extract_live_question(item)
            if candidate:
                return candidate
        return None

    @staticmethod
    def _extract_live_question(record: dict[str, object]) -> str | None:
        direct_questions = list(record.get("research_questions") or [])
        if direct_questions:
            candidate = str(direct_questions[0]).strip()
            if candidate:
                return candidate

        conversation_turn = record.get("conversation_turn")
        if not isinstance(conversation_turn, dict):
            response = record.get("response") or {}
            conversation_turn = response.get("conversation_turn") if isinstance(response, dict) else {}
        if isinstance(conversation_turn, dict):
            next_move = str(conversation_turn.get("next_move") or "").strip()
            if next_move:
                return next_move
            follow_ups = list(conversation_turn.get("follow_ups") or [])
            if follow_ups:
                candidate = str(follow_ups[0]).strip()
                if candidate:
                    return candidate

        response = record.get("response") or {}
        if isinstance(response, dict):
            response_questions = list(response.get("research_questions") or [])
            if response_questions:
                candidate = str(response_questions[0]).strip()
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _adaptive_posture(
        *,
        interaction_mode: str,
        idea_state: str,
        response_strategy: str,
        recent_intent_pattern: str,
        conversation_momentum: str,
        unresolved_thread_open: bool,
    ) -> str:
        if interaction_mode == "social":
            return "acknowledge"
        if recent_intent_pattern == "hesitating" or idea_state == "uncertain" or response_strategy == "ask_question":
            return "step_back"
        if conversation_momentum == "wrapping_up":
            return "acknowledge"
        if response_strategy == "challenge" or idea_state in {"challenged", "branching"}:
            return "push"
        if conversation_momentum == "building":
            return "push"
        if unresolved_thread_open and conversation_momentum == "building":
            return "push"
        return "acknowledge"
