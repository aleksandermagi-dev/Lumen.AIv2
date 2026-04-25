from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse


@dataclass(slots=True)
class ThreadExplanationEvidence:
    topic: str | None
    summary: str | None
    unresolved: str | None
    aligned_with_recent: bool
    strong: bool


class ThreadExplanationSupport:
    """Builds brief, conversational explanations of the live thread without surfacing internals."""

    THREAD_STATUS_PROMPTS = {
        "what are we doing again",
        "what are we doing",
        "what thread are we on",
        "what thread are we in",
        "what are we working on again",
        "what are we on",
    }
    WHY_STYLE_PROMPTS = {
        "why are you answering like that",
        "why are you responding like that",
        "why are you taking it this way",
    }
    GETTING_AT_PROMPTS = {
        "what do you think i'm getting at",
        "what do you think i am getting at",
        "what do you think i'm pointing at",
    }

    @classmethod
    def is_thread_explanation_prompt(cls, prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        return (
            normalized in cls.THREAD_STATUS_PROMPTS
            or normalized in cls.WHY_STYLE_PROMPTS
            or normalized in cls.GETTING_AT_PROMPTS
        )

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        evidence = cls._evaluate_evidence(
            active_thread=active_thread,
            recent_interactions=recent_interactions,
        )
        last_prompt = str((recent_interactions[0] if recent_interactions else {}).get("prompt") or "").strip()

        if normalized in cls.WHY_STYLE_PROMPTS:
            reply = cls._why_reply(
                style=style,
                topic=evidence.topic,
                unresolved=evidence.unresolved,
                last_prompt=last_prompt,
                strong=evidence.strong,
            )
        elif normalized in cls.GETTING_AT_PROMPTS:
            reply = cls._getting_at_reply(
                style=style,
                topic=evidence.topic,
                unresolved=evidence.unresolved,
                strong=evidence.strong,
            )
        else:
            reply = cls._thread_status_reply(
                style=style,
                topic=evidence.topic,
                summary=evidence.summary,
                unresolved=evidence.unresolved,
                strong=evidence.strong,
            )

        return ConversationResponse(
            mode="conversation",
            kind="conversation.thread_explanation",
            summary=reply,
            reply=reply,
        ).to_dict() | {
            "interaction_mode": "social",
            "idea_state": "refining",
            "response_strategy": "answer",
            "reasoning_depth": "low",
            "tools_enabled": False,
            "lightweight_social": True,
        }

    @staticmethod
    def _topic(*, active_thread: dict[str, object] | None) -> str | None:
        if active_thread is None:
            return None
        topic = str(active_thread.get("normalized_topic") or "").strip()
        if topic:
            return topic
        prompt = str(active_thread.get("prompt") or "").strip()
        return prompt or None

    @staticmethod
    def _summary(*, active_thread: dict[str, object] | None) -> str | None:
        if active_thread is None:
            return None
        summary = str(
            active_thread.get("thread_summary")
            or active_thread.get("summary")
            or active_thread.get("objective")
            or ""
        ).strip()
        return summary or None

    @staticmethod
    def _unresolved_focus(
        *,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> str | None:
        if recent_interactions:
            latest = recent_interactions[0]
            question = str(((latest.get("conversation_turn") or {}) if isinstance(latest.get("conversation_turn"), dict) else {}).get("next_move") or "").strip()
            if question:
                return question
        if active_thread is None:
            return None
        objective = str(active_thread.get("objective") or "").strip()
        return objective or None

    @classmethod
    def _evaluate_evidence(
        cls,
        *,
        active_thread: dict[str, object] | None,
        recent_interactions: list[dict[str, object]],
    ) -> ThreadExplanationEvidence:
        topic = cls._topic(active_thread=active_thread)
        summary = cls._summary(active_thread=active_thread)
        unresolved = cls._unresolved_focus(active_thread=active_thread, recent_interactions=recent_interactions)
        if active_thread is None:
            return ThreadExplanationEvidence(
                topic=topic,
                summary=summary,
                unresolved=unresolved,
                aligned_with_recent=False,
                strong=False,
            )
        active_mode = str(active_thread.get("mode") or "").strip()
        aligned_with_recent = True
        if recent_interactions:
            latest = recent_interactions[0]
            latest_mode = str(latest.get("mode") or "").strip()
            latest_text = " ".join(
                str(latest.get(key) or "").strip().lower()
                for key in ("prompt", "summary", "reply")
            )
            if latest_mode and active_mode and latest_mode != active_mode and not cls._topic_overlap(topic, latest_text):
                aligned_with_recent = False
        strong = bool(
            topic
            and summary
            and len(summary) >= 12
            and aligned_with_recent
        )
        return ThreadExplanationEvidence(
            topic=topic,
            summary=summary,
            unresolved=unresolved,
            aligned_with_recent=aligned_with_recent,
            strong=strong,
        )

    @staticmethod
    def _topic_overlap(topic: str | None, text: str) -> bool:
        cleaned = str(topic or "").strip().lower()
        if not cleaned or not text:
            return False
        tokens = [token for token in cleaned.split() if len(token) > 2]
        return any(token in text for token in tokens)

    @staticmethod
    def _thread_status_reply(
        *,
        style: str,
        topic: str | None,
        summary: str | None,
        unresolved: str | None,
        strong: bool,
    ) -> str:
        if not topic and not summary:
            return (
                "We don't have a strong live thread right this second. We can start fresh if you want."
                if style != "direct"
                else "No strong live thread right now. We can start fresh."
            )
        if not strong:
            if style == "direct":
                return f"It might still be {topic or 'that thread'}, but I don't have a clean live read."
            if style == "collab":
                return f"I don't have a perfectly clean live thread locked right now, but it looks closest to {topic or 'that thread'}."
            return f"It might be that we're still on {topic or 'that thread'}, but the live thread is a little fuzzy."
        if style == "direct":
            if unresolved:
                return f"We're still on {topic or 'that thread'}. The live focus is {unresolved.rstrip('.') }."
            return f"We're on {topic or 'that thread'}. {summary or ''}".strip()
        if style == "collab":
            if unresolved:
                return f"We're still in {topic or 'that thread'}, and the live pull is {unresolved.rstrip('.') }."
            return f"We're still in {topic or 'that thread'}. {summary or ''}".strip()
        if unresolved:
            return f"We're still on {topic or 'that thread'}, mainly around {unresolved.rstrip('.') }."
        return f"We're still on {topic or 'that thread'}. {summary or ''}".strip()

    @staticmethod
    def _why_reply(
        *,
        style: str,
        topic: str | None,
        unresolved: str | None,
        last_prompt: str,
        strong: bool,
    ) -> str:
        if not strong:
            if style == "direct":
                return "Because it looked a bit more like a continuation than a clean reset."
            return "Because it read more like you were continuing a thread than clearly starting a new one."
        if style == "direct":
            if topic:
                return f"Because it looked like you were continuing {topic}, not starting fresh."
            return "Because it looked like a continuation, not a fresh start."
        if topic and unresolved:
            return f"Because it felt like you were still pulling on {topic}, especially the part about {unresolved.rstrip('.') }."
        if topic:
            return f"Because it looked like you were still working the {topic} thread rather than starting over."
        if last_prompt:
            return f"Because your last turn read like a continuation of {last_prompt.rstrip('.') }."
        return "Because it read like you were continuing the same thread rather than switching topics."

    @staticmethod
    def _getting_at_reply(
        *,
        style: str,
        topic: str | None,
        unresolved: str | None,
        strong: bool,
    ) -> str:
        if not strong:
            return (
                "It sounds like you might be reaching back toward the main thread again."
                if style != "direct"
                else "It sounds like you're probably pointing back at the main thread."
            )
        if topic and unresolved:
            return (
                f"It sounds like you're pointing back at {topic}, especially the part about {unresolved.rstrip('.') }."
            )
        if topic:
            return f"It sounds like you're pointing back at {topic}."
        return (
            "It sounds like you're reaching for the main thread again."
            if style != "direct"
            else "It sounds like you're pointing back at the main thread."
        )
