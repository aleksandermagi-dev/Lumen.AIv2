from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.reasoning.conversation_assembler import ConversationAssembler
from lumen.reasoning.conversation_stamina_support import (
    ConversationStaminaState,
    ConversationStaminaSupport,
)
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_variation import ResponseVariationLayer


@dataclass(slots=True)
class ConversationalReplyState:
    lane: str
    intent: str
    stance: str
    topic: str | None
    main_content: str
    memory_context: tuple[str, ...] = ()
    project_memory_hint: str | None = None
    pickup_bridge: str | None = None
    optional_follow_up: str | None = None
    optional_closer: str | None = None
    opener: str | None = None
    confidence: str | None = None
    continuation_type: str | None = None
    stamina_state: ConversationStaminaState | None = None
    conversation_beat: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "intent": self.intent,
            "stance": self.stance,
            "topic": self.topic,
            "main_content": self.main_content,
            "memory_context": list(self.memory_context),
            "project_memory_hint": self.project_memory_hint,
            "pickup_bridge": self.pickup_bridge,
            "optional_follow_up": self.optional_follow_up,
            "optional_closer": self.optional_closer,
            "opener": self.opener,
            "confidence": self.confidence,
            "continuation_type": self.continuation_type,
            "stamina_state": self.stamina_state.to_dict() if self.stamina_state is not None else None,
            "conversation_beat": dict(self.conversation_beat or {}),
        }

    @property
    def realized_voice(self) -> str:
        if self.lane != "conversational":
            return "calm_grounded"
        if self.intent.startswith("conversation.") and self.continuation_type in {"collaborate", "thread_hold"}:
            return "warm_partner"
        return "calm_grounded"


class ConversationalReplyRealizer:
    """Builds one clean user-facing utterance for conversational turns."""

    @classmethod
    def build_state(
        cls,
        *,
        response: dict[str, object],
        interaction_profile: Any,
        recent_interactions: list[dict[str, object]] | None = None,
        active_thread: dict[str, object] | None = None,
    ) -> ConversationalReplyState | None:
        if str(response.get("mode") or "").strip() != "conversation":
            return None
        turn = response.get("conversation_turn") if isinstance(response.get("conversation_turn"), dict) else {}
        kind = str(response.get("kind") or turn.get("kind") or "conversation.reply").strip()
        stance = str(
            (response.get("stance_consistency") or {}).get("category")
            or turn.get("stance_category")
            or "neutral_acknowledgment"
        ).strip()
        lane = "conversational"
        summary = str(response.get("reply") or response.get("summary") or "").strip()
        follow_up = cls._follow_up_from_turn(turn)
        topic = cls._topic_from_turn(turn)
        continuation_type = str(turn.get("kind") or "").strip() or None
        memory_context = cls._memory_context_from_response(response)
        memory_reply_hint = str(response.get("memory_reply_hint") or "").strip()
        project_memory_hint = str(response.get("project_memory_hint") or "").strip() or None
        pickup_bridge = str(turn.get("pickup_bridge") or "").strip() or None
        conversation_beat = (
            dict(response.get("conversation_beat"))
            if isinstance(response.get("conversation_beat"), dict)
            else None
        )
        stamina_state = ConversationStaminaSupport.build(
            recent_interactions=recent_interactions,
            active_thread=active_thread,
            topic=topic,
        )

        if memory_reply_hint:
            return ConversationalReplyState(
                lane=lane,
                intent=kind,
                stance=stance,
                topic=topic,
                main_content=memory_reply_hint,
                memory_context=memory_context,
                project_memory_hint=project_memory_hint,
                continuation_type=continuation_type,
                confidence=cls._confidence_from_response(response),
                pickup_bridge=pickup_bridge,
                stamina_state=stamina_state,
                conversation_beat=conversation_beat,
            )

        if summary and not turn:
            return ConversationalReplyState(
                lane=lane,
                intent=kind,
                stance=stance,
                topic=topic,
                main_content=summary,
                memory_context=memory_context,
                project_memory_hint=project_memory_hint,
                continuation_type=continuation_type,
                pickup_bridge=pickup_bridge,
                stamina_state=stamina_state,
                conversation_beat=conversation_beat,
            )

        main_content = cls._main_content_from_turn(turn=turn, fallback_text=summary)
        if not main_content:
            return None

        return ConversationalReplyState(
            lane=lane,
            intent=kind,
            stance=stance,
            topic=topic,
            main_content=main_content,
            memory_context=memory_context,
            project_memory_hint=project_memory_hint,
            pickup_bridge=pickup_bridge,
            optional_follow_up=follow_up,
            continuation_type=continuation_type,
            confidence=cls._confidence_from_response(response),
            stamina_state=stamina_state,
            conversation_beat=conversation_beat,
        )

    @classmethod
    def realize(
        cls,
        *,
        state: ConversationalReplyState,
        interaction_profile: Any,
        recent_interactions: list[dict[str, object]] | None = None,
    ) -> str:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = [str(item.get("summary") or item.get("reply") or "").strip() for item in (recent_interactions or [])]
        continuity_presence = cls._continuity_presence_fragment(
            style=style,
            state=state,
        )
        opener = cls._opener_fragment(
            style=style,
            state=state,
            recent_texts=recent_texts,
        )
        follow_up = cls._follow_up_sentence(
            style=style,
            state=state,
            recent_texts=recent_texts,
        )
        closer = cls._closer_sentence(
            style=style,
            state=state,
        )
        return ConversationAssembler.assemble(
            style=style,
            seed_parts=[
                state.lane,
                state.intent,
                state.stance,
                state.topic or "",
                state.continuation_type or "",
                state.main_content,
            ],
            recent_texts=recent_texts,
            opener=state.opener or state.pickup_bridge or state.project_memory_hint or continuity_presence or opener,
            content=state.main_content,
            closer=follow_up or closer,
        )

    @staticmethod
    def _continuity_presence_fragment(
        *,
        style: str,
        state: ConversationalReplyState,
    ) -> str | None:
        if state.pickup_bridge or state.project_memory_hint:
            return None
        if state.intent in {"conversation.greeting", "conversation.check_in", "conversation.micro_turn"}:
            return None
        beat = state.conversation_beat or {}
        if beat.get("response_repetition_risk") == "high":
            return None
        stamina_state = state.stamina_state
        if stamina_state is None:
            return None
        if not stamina_state.reliable_long_chat:
            return None
        return ConversationStaminaSupport.choose_continuity_presence(
            style=style,
            state=stamina_state,
        )

    @staticmethod
    def _opener_fragment(
        *,
        style: str,
        state: ConversationalReplyState,
        recent_texts: list[str],
    ) -> str | None:
        if style != "collab":
            return None
        if not state.continuation_type:
            return None
        if state.intent not in {"conversation.greeting", "conversation.check_in"}:
            return None
        pools = {
            "conversation.greeting": (
                "Hey there!",
                "Good to see you.",
                "Hey - glad you're here.",
            ),
            "conversation.check_in": (
                "Hey.",
                "Good to see you.",
                "I'm with you.",
            ),
        }
        opener = ResponseVariationLayer.select_from_pool(
            pools.get(state.intent, ()),
            seed_parts=[style, state.intent, state.main_content, "opener"],
            recent_texts=recent_texts,
        ) if pools.get(state.intent) else None
        if not opener:
            return None
        normalized_opener = " ".join(opener.lower().split())
        normalized_content = " ".join(str(state.main_content or "").lower().split())
        if normalized_content.startswith(normalized_opener):
            return None
        return opener

    @staticmethod
    def _main_content_from_turn(*, turn: dict[str, object], fallback_text: str) -> str:
        starter = str(turn.get("follow_through_starter") or "").strip()
        lead = str(turn.get("lead") or "").strip()
        if starter:
            if not lead:
                return starter
            if ConversationalReplyRealizer._is_generic_conversational_lead(lead):
                return starter
            normalized_starter = " ".join(starter.lower().split())
            normalized_lead = " ".join(lead.lower().split())
            if normalized_lead.startswith(normalized_starter):
                return lead
            return f"{starter} {lead}"
        if lead:
            return lead
        return fallback_text

    @staticmethod
    def _is_generic_conversational_lead(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        generic_fragments = (
            "here's my read",
            "here is my read",
            "here's the clearest read",
            "here is the clearest read",
            "here's the current read",
            "here is the current read",
            "here's where i'd start",
            "here is where i'd start",
            "that tracks. here's",
            "that makes sense. here's",
            "worth exploring",
            "there's something worth exploring here",
            "there is something worth exploring here",
            "here's the shape of it so far",
            "here is the shape of it so far",
        )
        return any(fragment in normalized for fragment in generic_fragments)

    @staticmethod
    def _topic_from_turn(turn: dict[str, object]) -> str | None:
        follow_ups = list(turn.get("follow_ups") or [])
        for item in follow_ups:
            text = str(item).strip()
            if text:
                return text
        next_move = str(turn.get("next_move") or "").strip()
        return next_move or None

    @staticmethod
    def _confidence_from_response(response: dict[str, object]) -> str | None:
        confidence = str(response.get("confidence_posture") or "").strip()
        return confidence or None

    @staticmethod
    def _memory_context_from_response(response: dict[str, object]) -> tuple[str, ...]:
        retrieval = response.get("memory_retrieval")
        if not isinstance(retrieval, dict):
            return ()
        selected = retrieval.get("selected")
        if not isinstance(selected, list):
            return ()
        contexts: list[str] = []
        for item in selected[:2]:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or "").strip()
            if summary:
                contexts.append(summary)
        return tuple(contexts)

    @classmethod
    def _follow_up_from_turn(cls, turn: dict[str, object]) -> str | None:
        lead = str(turn.get("lead") or "").strip()
        if lead.endswith("?"):
            return None
        next_move = str(turn.get("next_move") or "").strip()
        if next_move.endswith("?"):
            return next_move
        follow_ups = list(turn.get("follow_ups") or [])
        for item in follow_ups:
            text = str(item).strip()
            if text.endswith("?"):
                return text
        for item in follow_ups:
            text = str(item).strip()
            if text:
                return text
        return None

    @classmethod
    def _follow_up_sentence(
        cls,
        *,
        style: str,
        state: ConversationalReplyState,
        recent_texts: list[str],
    ) -> str | None:
        follow_up = str(state.optional_follow_up or "").strip()
        if not follow_up:
            return None
        if follow_up.endswith("?"):
            return follow_up
        beat = state.conversation_beat or {}
        if beat.get("follow_up_offer_allowed") is False:
            return None
        stamina_state = state.stamina_state
        if (
            stamina_state is not None
            and stamina_state.reliable_long_chat
            and not stamina_state.continuation_offer_cooldown
        ):
            pool = ConversationStaminaSupport.follow_up_pool(
                style=style,
                state=stamina_state,
                follow_up=follow_up,
            )
        elif style == "direct":
            pool = (
                f"We can go next on {follow_up}.",
                f"Next, we can hit {follow_up}.",
            )
        elif style == "collab":
            pool = (
                f"If you want, we can keep pulling on {follow_up}.",
                f"We can stay with {follow_up} a little longer if you want.",
                f"Happy to keep going on {follow_up} with you.",
                f"We could poke at {follow_up} next if that feels useful.",
                f"We could keep tugging on {follow_up} for a minute.",
                f"We can stay right there a little longer if you want.",
            )
        else:
            pool = (
                f"We can keep going on {follow_up} if you want.",
                f"If you want, we can go a bit further on {follow_up}.",
                f"We can dig into {follow_up} next if you want.",
                f"We can stay with {follow_up} a little longer if you want.",
                f"We can keep pulling on {follow_up} a bit if that helps.",
            )
        return ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style, state.intent, state.stance, follow_up, "follow_up"],
            recent_texts=recent_texts,
        )

    @staticmethod
    def _closer_sentence(*, style: str, state: ConversationalReplyState) -> str | None:
        closer = str(state.optional_closer or "").strip()
        return closer or None

