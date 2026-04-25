from __future__ import annotations

from dataclasses import dataclass

from lumen.reasoning.response_variation import ResponseVariationLayer


@dataclass(slots=True)
class ConversationStaminaState:
    long_chat: bool
    reliable_long_chat: bool
    mixed_recent_modes: bool
    continuity_topic: str | None
    recent_surface_roles: tuple[str, ...]
    recent_follow_up_shapes: tuple[str, ...]
    recent_texts: tuple[str, ...]
    continuity_presence_cooldown: bool
    continuation_offer_cooldown: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "long_chat": self.long_chat,
            "reliable_long_chat": self.reliable_long_chat,
            "mixed_recent_modes": self.mixed_recent_modes,
            "continuity_topic": self.continuity_topic,
            "recent_surface_roles": list(self.recent_surface_roles),
            "recent_follow_up_shapes": list(self.recent_follow_up_shapes),
            "recent_texts": list(self.recent_texts),
            "continuity_presence_cooldown": self.continuity_presence_cooldown,
            "continuation_offer_cooldown": self.continuation_offer_cooldown,
        }


class ConversationStaminaSupport:
    """Tracks lightweight conversational surface history so long chats stay varied."""

    @staticmethod
    def build(
        *,
        recent_interactions: list[dict[str, object]] | None,
        active_thread: dict[str, object] | None,
        topic: str | None,
    ) -> ConversationStaminaState:
        interactions = list(recent_interactions or [])
        recent_modes = [str(item.get("mode") or "").strip() for item in interactions[:4] if str(item.get("mode") or "").strip()]
        conversational = [
            item
            for item in interactions[:12]
            if str(item.get("mode") or "").strip() == "conversation"
        ]
        recent_surface_roles: list[str] = []
        recent_follow_up_shapes: list[str] = []
        recent_texts: list[str] = []
        for item in conversational[:8]:
            kind = str(item.get("kind") or "").strip()
            if kind:
                recent_surface_roles.append(kind)
            text = str(item.get("summary") or item.get("reply") or "").strip()
            if text:
                recent_texts.append(text)
                lowered = text.lower()
                if "if you want" in lowered or "we can keep" in lowered or "we could keep" in lowered:
                    recent_follow_up_shapes.append("continuation_offer")
                elif lowered.endswith("?"):
                    recent_follow_up_shapes.append("question")
                else:
                    recent_follow_up_shapes.append("statement")
        continuity_topic = (
            str(active_thread.get("normalized_topic") or active_thread.get("thread_summary") or "").strip()
            if active_thread
            else ""
        ) or topic
        long_chat = len(conversational) >= 4
        mixed_recent_modes = bool(recent_modes) and any(mode != "conversation" for mode in recent_modes)
        topic_strength = ConversationStaminaSupport._topic_strength(
            topic=continuity_topic,
            recent_texts=recent_texts,
        )
        continuity_presence_cooldown = any(
            phrase in " ".join(recent_texts[:3]).lower()
            for phrase in (
                "i'm still with you on this",
                "we can keep pulling on this from here",
                "we're still on the same thread here",
                "let's stay with this for a second",
            )
        )
        continuation_offer_cooldown = recent_follow_up_shapes[:3].count("continuation_offer") >= 2
        return ConversationStaminaState(
            long_chat=long_chat,
            reliable_long_chat=bool(long_chat and not mixed_recent_modes and (topic_strength > 0 or active_thread is not None)),
            mixed_recent_modes=mixed_recent_modes,
            continuity_topic=continuity_topic or None,
            recent_surface_roles=tuple(recent_surface_roles),
            recent_follow_up_shapes=tuple(recent_follow_up_shapes),
            recent_texts=tuple(recent_texts),
            continuity_presence_cooldown=continuity_presence_cooldown,
            continuation_offer_cooldown=continuation_offer_cooldown,
        )

    @staticmethod
    def _topic_strength(*, topic: str | None, recent_texts: list[str]) -> int:
        cleaned = str(topic or "").strip().lower()
        if not cleaned:
            return 0
        tokens = [token for token in cleaned.split() if len(token) > 2]
        if not tokens:
            return 0
        score = 0
        haystacks = [text.lower() for text in recent_texts[:4]]
        for token in tokens:
            if any(token in haystack for haystack in haystacks):
                score += 1
        return score

    @staticmethod
    def continuity_presence(
        *,
        style: str,
        state: ConversationStaminaState,
    ) -> tuple[str, ...]:
        topic = str(state.continuity_topic or "this").strip()
        if style == "direct":
            return (
                "Let's stay with this.",
                f"Stay with {topic}.",
                "Keep going from here.",
            )
        if style == "collab":
            return (
                "I'm still with you on this.",
                f"Yeah, let's stay with {topic} for a second.",
                "We can keep pulling on this from here.",
                "Alright, we're still in it.",
            )
        return (
            "Let's stay with this for a second.",
            f"We can keep going on {topic} from here.",
            "We're still on the same thread here.",
        )

    @staticmethod
    def follow_up_pool(
        *,
        style: str,
        state: ConversationStaminaState,
        follow_up: str,
    ) -> tuple[str, ...]:
        if style == "direct":
            return (
                f"We can keep going on {follow_up}.",
                f"Next, stay with {follow_up}.",
                f"We can push a little further on {follow_up}.",
            )
        if style == "collab":
            return (
                f"We can keep tugging on {follow_up} from here if you want.",
                f"We could stay with {follow_up} a little longer.",
                f"Happy to keep pulling on {follow_up}.",
                f"We could go one click deeper on {follow_up}.",
                f"We can keep moving with {follow_up} if that feels right.",
            )
        return (
            f"We can keep going on {follow_up} from here if you want.",
            f"We can stay with {follow_up} a little longer.",
            f"We could go a bit further on {follow_up}.",
            f"We can keep pulling on {follow_up} if that helps.",
        )

    @staticmethod
    def choose_continuity_presence(
        *,
        style: str,
        state: ConversationStaminaState,
    ) -> str | None:
        if not state.reliable_long_chat or state.continuity_presence_cooldown:
            return None
        pool = ConversationStaminaSupport.continuity_presence(style=style, state=state)
        return ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style, state.continuity_topic or "", "continuity_presence"],
            recent_texts=list(state.recent_texts),
        )
