from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.conversation_stamina_support import (
    ConversationStaminaState,
    ConversationStaminaSupport,
)


@dataclass(slots=True)
class ConversationBeatState:
    continuity_state: str
    conversation_depth: int
    topic_shift: str
    response_repetition_risk: str
    follow_up_offer_allowed: bool
    long_chat_stamina: dict[str, object]
    current_topic: str | None = None
    recent_social_posture: str | None = None
    memory_restraint: str = "recent_first"

    def to_dict(self) -> dict[str, object]:
        return {
            "continuity_state": self.continuity_state,
            "conversation_depth": self.conversation_depth,
            "topic_shift": self.topic_shift,
            "response_repetition_risk": self.response_repetition_risk,
            "follow_up_offer_allowed": self.follow_up_offer_allowed,
            "long_chat_stamina": dict(self.long_chat_stamina),
            "current_topic": self.current_topic,
            "recent_social_posture": self.recent_social_posture,
            "memory_restraint": self.memory_restraint,
        }


class ConversationBeatSupport:
    """Summarizes the local conversational beat for long-chat continuity."""

    CONTINUATION_CUES = {
        "yeah",
        "yep",
        "yes",
        "true",
        "right",
        "exactly",
        "that makes sense",
        "makes sense",
        "keep going",
        "go on",
        "continue",
        "tell me more",
        "what else",
        "go deeper",
    }
    HESITATION_CUES = ("hmm", "maybe", "not sure", "i don't know", "i dont know", "unsure")
    WRAP_UP_CUES = ("good for now", "that's enough", "thats enough", "we can stop", "done for now")
    RETURN_CUES = (
        "go back",
        "back to what we were saying",
        "return to what we were saying",
        "pick back up",
        "where were we",
    )
    FOLLOW_UP_OFFER_MARKERS = (
        "if you want",
        "we can keep",
        "we could keep",
        "we can stay",
        "we could stay",
        "happy to keep",
    )

    @classmethod
    def build(
        cls,
        *,
        prompt: str,
        response: dict[str, object],
        recent_interactions: list[dict[str, object]] | None,
        active_thread: dict[str, object] | None,
    ) -> ConversationBeatState:
        interactions = list(recent_interactions or [])
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        response_text = cls._response_text(response)
        recent_texts = cls._recent_texts(interactions)
        current_topic = cls._current_topic(
            prompt=normalized,
            response=response,
            recent_interactions=interactions,
            active_thread=active_thread,
        )
        stamina = ConversationStaminaSupport.build(
            recent_interactions=interactions,
            active_thread=active_thread,
            topic=current_topic,
        )
        repetition_risk = cls._repetition_risk(
            response_text=response_text,
            recent_texts=recent_texts,
            recent_interactions=interactions,
            stamina=stamina,
        )
        continuity_state = cls._continuity_state(
            normalized=normalized,
            recent_interactions=interactions,
            active_thread=active_thread,
        )
        follow_up_offer_allowed = cls._follow_up_offer_allowed(
            continuity_state=continuity_state,
            repetition_risk=repetition_risk,
            stamina=stamina,
        )
        return ConversationBeatState(
            continuity_state=continuity_state,
            conversation_depth=cls._conversation_depth(interactions),
            topic_shift=cls._topic_shift(
                normalized=normalized,
                current_topic=current_topic,
                recent_interactions=interactions,
                active_thread=active_thread,
                continuity_state=continuity_state,
            ),
            response_repetition_risk=repetition_risk,
            follow_up_offer_allowed=follow_up_offer_allowed,
            long_chat_stamina={
                "long_chat": stamina.long_chat,
                "reliable_long_chat": stamina.reliable_long_chat,
                "mixed_recent_modes": stamina.mixed_recent_modes,
                "continuation_offer_cooldown": stamina.continuation_offer_cooldown,
                "continuity_presence_cooldown": stamina.continuity_presence_cooldown,
            },
            current_topic=current_topic,
            recent_social_posture=cls._recent_social_posture(interactions),
        )

    @classmethod
    def _continuity_state(
        cls,
        *,
        normalized: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        if any(cue in normalized for cue in cls.RETURN_CUES):
            return "returning"
        if any(cue in normalized for cue in cls.WRAP_UP_CUES):
            return "wrapping_up"
        if any(cue in normalized for cue in cls.HESITATION_CUES):
            return "hesitating"
        if normalized in cls.CONTINUATION_CUES or any(normalized.startswith(f"{cue} ") for cue in cls.CONTINUATION_CUES):
            return "continuing"
        if cls._looks_like_social_opening(normalized) and not recent_interactions:
            return "social_opening"
        if active_thread is not None or recent_interactions:
            return "thread_holding"
        return "fresh"

    @classmethod
    def _topic_shift(
        cls,
        *,
        normalized: str,
        current_topic: str | None,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        continuity_state: str,
    ) -> str:
        if continuity_state == "returning":
            return "return_to_recent"
        if continuity_state in {"continuing", "hesitating", "wrapping_up"}:
            return "none"
        anchor = str(
            (active_thread or {}).get("normalized_topic")
            or (active_thread or {}).get("thread_summary")
            or current_topic
            or ""
        ).lower()
        if not anchor and recent_interactions:
            anchor = str(recent_interactions[0].get("summary") or recent_interactions[0].get("prompt") or "").lower()
        prompt_tokens = cls._topic_tokens(normalized)
        anchor_tokens = cls._topic_tokens(anchor)
        if not prompt_tokens or not anchor_tokens:
            return "none"
        overlap = prompt_tokens & anchor_tokens
        if overlap:
            return "none"
        return "soft"

    @classmethod
    def _repetition_risk(
        cls,
        *,
        response_text: str,
        recent_texts: list[str],
        recent_interactions: list[dict[str, object]],
        stamina: ConversationStaminaState,
    ) -> str:
        normalized_response = cls._normalize_text(response_text)
        if normalized_response and normalized_response in {cls._normalize_text(text) for text in recent_texts[:4]}:
            return "high"
        recent_offer_count = sum(cls._has_follow_up_offer(text) for text in recent_texts[:4])
        response_has_offer = cls._has_follow_up_offer(response_text)
        recent_kinds = [str(item.get("kind") or "").strip() for item in recent_interactions[:4]]
        same_kind_count = recent_kinds.count(str((recent_interactions[0] if recent_interactions else {}).get("kind") or ""))
        if stamina.continuation_offer_cooldown or (response_has_offer and recent_offer_count >= 2):
            return "high"
        if response_has_offer and recent_offer_count >= 1:
            return "medium"
        if same_kind_count >= 3:
            return "medium"
        return "low"

    @staticmethod
    def _follow_up_offer_allowed(
        *,
        continuity_state: str,
        repetition_risk: str,
        stamina: ConversationStaminaState,
    ) -> bool:
        if continuity_state == "wrapping_up":
            return False
        if repetition_risk == "high" or stamina.continuation_offer_cooldown:
            return False
        return True

    @staticmethod
    def _conversation_depth(recent_interactions: list[dict[str, object]]) -> int:
        conversational_count = sum(
            1
            for item in recent_interactions[:12]
            if str(item.get("mode") or "").strip() == "conversation"
        )
        return conversational_count + 1

    @staticmethod
    def _current_topic(
        *,
        prompt: str,
        response: dict[str, object],
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str | None:
        if active_thread:
            for key in ("normalized_topic", "objective", "thread_summary", "summary"):
                value = str(active_thread.get(key) or "").strip()
                if value:
                    return value
        turn = response.get("conversation_turn") if isinstance(response.get("conversation_turn"), dict) else {}
        for key in ("next_move", "lead"):
            value = str(turn.get(key) or "").strip()
            if value:
                return value[:120]
        for item in recent_interactions[:3]:
            value = str(item.get("prompt") or item.get("summary") or "").strip()
            if value:
                return value[:120]
        return prompt[:120] if prompt else None

    @classmethod
    def _recent_social_posture(cls, recent_interactions: list[dict[str, object]]) -> str | None:
        for item in recent_interactions[:3]:
            mode = str(item.get("mode") or "").strip()
            kind = str(item.get("kind") or "").strip()
            if mode == "conversation":
                if "check_in" in kind:
                    return "check_in"
                if "greeting" in kind:
                    return "greeting"
                if "self_overview" in kind:
                    return "self_referential"
                return "conversational"
        return None

    @classmethod
    def _recent_texts(cls, recent_interactions: list[dict[str, object]]) -> list[str]:
        texts: list[str] = []
        for item in recent_interactions[:8]:
            response = item.get("response") if isinstance(item.get("response"), dict) else {}
            text = str(response.get("reply") or response.get("summary") or item.get("summary") or "").strip()
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _response_text(response: dict[str, object]) -> str:
        return str(response.get("reply") or response.get("summary") or response.get("user_facing_answer") or "").strip()

    @classmethod
    def _has_follow_up_offer(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        return any(marker in normalized for marker in cls.FOLLOW_UP_OFFER_MARKERS)

    @staticmethod
    def _looks_like_social_opening(normalized: str) -> bool:
        return normalized in {"hi", "hello", "hey", "hey buddy", "hi buddy", "hey lumen", "good morning", "good evening"}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _topic_tokens(text: str) -> set[str]:
        stop = {
            "about",
            "again",
            "going",
            "have",
            "just",
            "keep",
            "like",
            "mean",
            "that",
            "this",
            "what",
            "when",
            "where",
            "with",
            "would",
            "you",
            "your",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(token) > 2 and token not in stop
        }
