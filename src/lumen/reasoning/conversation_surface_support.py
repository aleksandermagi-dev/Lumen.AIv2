from __future__ import annotations

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.social_interaction_policy import SocialInteractionPolicy
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse


class ConversationSurfaceSupport:
    """Owns lightweight conversational response helpers used by InteractionService."""

    RETURN_TO_RECENT_PROMPTS = {
        "go back to what we were saying",
        "back to what we were saying",
        "return to what we were saying",
        "pick back up",
        "where were we",
        "go back",
    }

    @staticmethod
    def build_lightweight_social_response(
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> dict[str, object]:
        social = SocialInteractionPolicy.build_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )
        response = ConversationResponse(
            mode="conversation",
            kind=str(social["kind"]),
            summary=str(social["reply"]),
            reply=str(social["reply"]),
        ).to_dict() | {
            "state_control": ConversationSurfaceSupport.social_state_control(
                kind=str(social["kind"]),
                prompt=prompt,
            ),
            "dialogue_management": {
                "interaction_mode": social["interaction_mode"],
                "idea_state": social["idea_state"],
                "response_strategy": social["response_strategy"],
                "synthesis_checkpoint_due": False,
                "checkpoint_reason": None,
            },
            "thought_framing": {
                "response_kind_label": "lightweight_social",
                "conversation_activity": "opening or maintaining a lightweight social turn",
                "research_questions": [],
                "checkpoint_summary": None,
            },
            "interaction_mode": social["interaction_mode"],
            "idea_state": social["idea_state"],
            "response_strategy": social["response_strategy"],
            "reasoning_depth": social["reasoning_depth"],
            "tools_enabled": social["tools_enabled"],
            "lightweight_social": True,
        }
        if str(social.get("kind") or "").strip() == "conversation.thought_mode":
            for key in ("thought_seed", "thought_topic", "thought_explanation"):
                value = str(social.get(key) or "").strip()
                if value:
                    response[key] = value
        return response

    @staticmethod
    def build_thought_follow_up_response(
        *,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        latest = recent_interactions[0] if recent_interactions else {}
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        thought_explanation = str(response.get("thought_explanation") or latest.get("thought_explanation") or "").strip()
        thought_topic = str(response.get("thought_topic") or latest.get("thought_topic") or "").strip()
        latest_summary = str(latest.get("summary") or "").strip().lower()
        if thought_explanation:
            reply = (
                thought_explanation.split(". ")[0].rstrip(".") + "."
                if style == "direct"
                else thought_explanation
            )
        elif "small assumption" in latest_summary:
            reply = (
                "Because a tiny assumption can quietly tilt everything built on top of it."
                if style == "direct"
                else "Because one tiny assumption can quietly tilt everything built on top of it, and then the whole line of reasoning starts leaning with it."
            )
        elif "framing" in latest_summary or "shape of a question" in latest_summary:
            reply = (
                "Because the framing changes what answers even become visible."
                if style == "direct"
                else "Because the way you frame a question changes what answers even become visible, so the framing does more work than it first looks like."
            )
        else:
            reply = (
                "Because the interesting part is what that idea changes downstream."
                if style == "direct"
                else "Because the interesting part is usually what that thought changes downstream once you start following it for a minute."
            )
        return ConversationResponse(
            mode="conversation",
            kind="conversation.thought_follow_up",
            summary=reply,
            reply=reply,
        ).to_dict() | {
            "interaction_mode": "social",
            "idea_state": "refining",
            "response_strategy": "answer",
            "reasoning_depth": "low",
            "tools_enabled": False,
            "lightweight_social": True,
            "thought_topic": thought_topic or None,
        }

    @staticmethod
    def is_return_to_recent_prompt(prompt: str) -> bool:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        return normalized in ConversationSurfaceSupport.RETURN_TO_RECENT_PROMPTS

    @staticmethod
    def build_return_to_recent_response(
        *,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        latest_conversation = None
        for item in recent_interactions[:5]:
            if str(item.get("mode") or "").strip() == "conversation":
                latest_conversation = item
                break
        if latest_conversation is None:
            return None
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        topic = str(latest_conversation.get("prompt") or latest_conversation.get("summary") or "that thread").strip()
        if style == "direct":
            reply = f"Back to that: {topic}."
        elif style == "collab":
            reply = f"Yeah, let's pick that thread back up: {topic}."
        else:
            reply = f"Let's go back to that thread: {topic}."
        return ConversationResponse(
            mode="conversation",
            kind="conversation.return_to_recent",
            summary=reply,
            reply=reply,
        ).to_dict() | {
            "interaction_mode": "social",
            "idea_state": "refining",
            "response_strategy": "answer",
            "reasoning_depth": "low",
            "tools_enabled": False,
            "lightweight_social": True,
            "conversation_turn": {
                "kind": "thread_hold",
                "lead": reply,
                "follow_ups": [],
            },
        }

    @staticmethod
    def is_thought_follow_up_prompt(
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
    ) -> bool:
        if not recent_interactions:
            return False
        latest = recent_interactions[0]
        if str(latest.get("kind") or "").strip() != "conversation.thought_mode":
            return False
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        return normalized in {
            "why",
            "why is that",
            "what do you mean",
            "what do you mean by that",
            "how so",
            "go on",
            "tell me more",
            "what else",
            "keep going",
            "really",
            "wait what",
            "what are you thinking",
            "what's on your mind",
            "whats on your mind",
        }

    @staticmethod
    def social_state_control(*, kind: str, prompt: str) -> dict[str, object]:
        normalized = " ".join(str(prompt).lower().split())
        if kind in {"conversation.thought_mode", "conversation.topic_suggestion"}:
            return {
                "core_state": "curiosity",
                "trigger": "confusion",
                "anti_spiral_active": False,
                "anti_spiral_reason": None,
                "response_bias": "explore",
                "humor_allowed": False,
            }
        if kind in {"conversation.acknowledgment", "conversation.transition"}:
            return {
                "core_state": "momentum",
                "trigger": "breakthrough",
                "anti_spiral_active": False,
                "anti_spiral_reason": None,
                "response_bias": "advance",
                "humor_allowed": False,
            }
        if kind == "conversation.farewell":
            return {
                "core_state": "focus",
                "trigger": "clarity_achieved",
                "anti_spiral_active": False,
                "anti_spiral_reason": None,
                "response_bias": "stabilize",
                "humor_allowed": False,
            }
        humor_allowed = any(token in normalized for token in ("lol", "haha", "lmao"))
        return {
            "core_state": "focus",
            "trigger": "clarity_achieved",
            "anti_spiral_active": False,
            "anti_spiral_reason": None,
            "response_bias": "clarify",
            "humor_allowed": humor_allowed,
        }
