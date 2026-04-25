from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse


@dataclass(slots=True)
class WorkThreadContinuityDecision:
    intent: str
    confidence: float
    reason: str


class WorkThreadContinuitySupport:
    """Answers lightweight current-work follow-ups from the live active thread."""

    NEXT_STEP_PROMPTS = {
        "what next",
        "next",
        "next step",
        "next steps",
        "what should we do next",
        "where do we go from here",
        "what's next",
        "whats next",
    }
    SUMMARY_PROMPTS = {
        "summarize where we are",
        "summary of where we are",
        "where are we",
        "where are we at",
        "status",
        "recap",
        "catch me up",
    }
    CONTINUE_PROMPTS = {
        "keep going",
        "continue",
        "go on",
        "proceed",
        "carry on",
        "keep working",
    }
    DECISION_PROMPTS = {
        "what did we decide",
        "what have we decided",
        "decisions so far",
        "what are the decisions",
        "what's decided",
        "whats decided",
    }

    @classmethod
    def classify(
        cls,
        *,
        prompt: str,
        active_thread: dict[str, object] | None,
    ) -> WorkThreadContinuityDecision | None:
        if not isinstance(active_thread, dict) or not active_thread:
            return None
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if normalized in cls.NEXT_STEP_PROMPTS:
            return WorkThreadContinuityDecision(
                intent="next_step",
                confidence=0.92,
                reason="Exact current-work next-step follow-up matched an active thread.",
            )
        if normalized in cls.SUMMARY_PROMPTS:
            return WorkThreadContinuityDecision(
                intent="status_summary",
                confidence=0.9,
                reason="Exact current-work summary follow-up matched an active thread.",
            )
        if normalized in cls.CONTINUE_PROMPTS:
            return WorkThreadContinuityDecision(
                intent="continue_work",
                confidence=0.88,
                reason="Exact current-work continuation prompt matched an active thread.",
            )
        if normalized in cls.DECISION_PROMPTS:
            return WorkThreadContinuityDecision(
                intent="decision_recap",
                confidence=0.88,
                reason="Exact current-work decision recap prompt matched an active thread.",
            )
        return None

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        active_thread: dict[str, object],
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        decision = cls.classify(prompt=prompt, active_thread=active_thread)
        if decision is None:
            return None
        if decision.intent == "continue_work" and cls._has_live_unresolved_question(recent_interactions):
            return None
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        context = cls._context(active_thread=active_thread, recent_interactions=recent_interactions)
        reply = cls._reply(style=style, decision=decision, context=context)
        return ConversationResponse(
            mode="conversation",
            kind=f"conversation.work_thread_{decision.intent}",
            summary=reply,
            reply=reply,
        ).to_dict() | {
            "interaction_mode": "work_thread",
            "idea_state": "refining",
            "response_strategy": "answer",
            "reasoning_depth": "low",
            "tools_enabled": False,
            "lightweight_social": False,
            "work_thread_continuity": {
                "active": True,
                "intent": decision.intent,
                "source": "active_thread",
                "confidence": decision.confidence,
                "reason": decision.reason,
                "thread_mode": context["mode"],
                "thread_kind": context["kind"],
                "topic": context["topic"],
                "has_tool_continuity": bool(context["tool"]),
                "memory_policy": "recent_thread_first",
            },
            "conversation_turn": {
                "kind": "work_thread_hold",
                "lead": reply,
                "follow_ups": [],
            },
        }

    @staticmethod
    def _context(
        *,
        active_thread: dict[str, object],
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        topic = str(
            active_thread.get("normalized_topic")
            or active_thread.get("prompt")
            or active_thread.get("objective")
            or "the current thread"
        ).strip()
        objective = str(active_thread.get("objective") or "").strip()
        summary = str(
            active_thread.get("thread_summary")
            or active_thread.get("summary")
            or objective
            or topic
        ).strip()
        latest_prompt = ""
        latest_assistant = ""
        if recent_interactions:
            latest = recent_interactions[0]
            latest_prompt = str(latest.get("prompt") or "").strip()
            response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
            latest_assistant = str(
                response.get("reply")
                or response.get("user_facing_answer")
                or response.get("summary")
                or latest.get("summary")
                or ""
            ).strip()
        tool_context = active_thread.get("tool_context") if isinstance(active_thread.get("tool_context"), dict) else {}
        return {
            "mode": str(active_thread.get("mode") or "").strip() or None,
            "kind": str(active_thread.get("kind") or "").strip() or None,
            "topic": topic,
            "objective": objective,
            "summary": summary,
            "latest_prompt": latest_prompt,
            "latest_assistant": latest_assistant,
            "tool": dict(tool_context or {}),
        }

    @staticmethod
    def _has_live_unresolved_question(recent_interactions: list[dict[str, object]]) -> bool:
        for item in recent_interactions[:2]:
            direct_questions = item.get("research_questions")
            if isinstance(direct_questions, list) and any(str(question).strip() for question in direct_questions):
                return True
            response = item.get("response") if isinstance(item.get("response"), dict) else {}
            if isinstance(response.get("research_questions"), list) and any(
                str(question).strip() for question in response.get("research_questions") or []
            ):
                return True
            conversation_turn = response.get("conversation_turn") if isinstance(response.get("conversation_turn"), dict) else item.get("conversation_turn")
            if isinstance(conversation_turn, dict) and str(conversation_turn.get("next_move") or "").strip():
                return True
        return False

    @classmethod
    def _reply(
        cls,
        *,
        style: str,
        decision: WorkThreadContinuityDecision,
        context: dict[str, object],
    ) -> str:
        topic = cls._sentence_fragment(context.get("topic")) or "the current thread"
        objective = cls._sentence_fragment(context.get("objective"))
        summary = cls._sentence_fragment(context.get("summary"))
        latest_prompt = cls._sentence_fragment(context.get("latest_prompt"))
        tool = context.get("tool") if isinstance(context.get("tool"), dict) else {}
        tool_label = " / ".join(
            part for part in (str(tool.get("tool_id") or "").strip(), str(tool.get("capability") or "").strip()) if part
        )

        if decision.intent == "status_summary":
            base = summary or objective or topic
            if style == "direct":
                return f"We're on {topic}. Current state: {base}."
            if style == "collab":
                return f"We're still on {topic}. The clean read is: {base}."
            return f"We're on {topic}. The current thread is: {base}."

        if decision.intent == "decision_recap":
            base = objective or summary or topic
            if style == "direct":
                return f"Decision frame so far: keep the work anchored on {base}."
            if style == "collab":
                return f"The decision shape so far is to keep us anchored on {base}, then tighten the next move from there."
            return f"So far, the decision frame is to keep the work anchored on {base} and choose the next move from that."

        if decision.intent == "continue_work":
            anchor = latest_prompt or objective or summary or topic
            if style == "direct":
                return f"Continuing from {anchor}: tighten the next concrete step and avoid widening the thread."
            if style == "collab":
                return f"Let's keep carrying that forward from {anchor}: the useful move is to tighten the next concrete step instead of widening the thread."
            return f"Continuing from {anchor}, the useful move is to tighten the next concrete step and keep the thread narrow."

        next_focus = objective or summary or topic
        tool_clause = f" If we need execution, the live tool context is {tool_label}." if tool_label else ""
        if style == "direct":
            return f"Next: make {next_focus} concrete, then verify the smallest blocker.{tool_clause}"
        if style == "collab":
            return f"Next, I'd keep us close to {next_focus}: make the next step concrete, then check the smallest blocker before we expand.{tool_clause}"
        return f"Next, we should keep the work anchored on {next_focus}, make the next step concrete, and verify the smallest blocker before expanding.{tool_clause}"

    @staticmethod
    def _sentence_fragment(value: object) -> str:
        text = " ".join(str(value or "").strip().split())
        return text.rstrip(".")
