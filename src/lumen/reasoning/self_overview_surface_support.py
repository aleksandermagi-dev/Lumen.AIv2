from __future__ import annotations

from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse
from lumen.reasoning.response_variation import ResponseVariationLayer


class SelfOverviewSurfaceSupport:
    """Owns self-overview and bounded memory-overview surfaces."""

    IDENTITY_PROMPTS = {
        "who are you",
        "tell me about yourself",
        "tell me more about yourself",
        "explain yourself",
        "what are you like",
        "what do you do here",
        "how would you describe yourself",
        "what kind of assistant are you",
        "what about you",
        "how about you",
    }
    CAPABILITY_PROMPTS = {
        "what can you do",
        "what all can you do",
        "what do you do",
        "tell me what you can do",
        "what are you capable of",
        "what can you help with",
        "what all can you help with",
    }
    KNOWLEDGE_PROMPTS = {
        "what all do you know",
        "what do you know",
        "tell me what you know",
    }
    MEMORY_PROMPTS = {
        "what do you remember",
        "what all do you remember",
        "what can you remember",
    }
    SOCIAL_FOLLOW_UP_PROMPTS = {
        "and you",
        "what about you",
        "how about you",
    }

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        tool_map: dict[str, list[str]],
        knowledge_service: KnowledgeService | None,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        prompt_classification = cls.classify_prompt(
            prompt=prompt,
            recent_interactions=recent_interactions,
        )
        if prompt_classification is None:
            return None
        category = prompt_classification["category"]
        source = prompt_classification["source"]
        normalized = str(prompt_classification["normalized"] or "")
        if category == "memory":
            return cls._memory_overview(
                interaction_profile=interaction_profile,
                recent_interactions=recent_interactions,
            )

        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        bundle_labels = {
            "math": "math",
            "system": "system analysis",
            "knowledge": "knowledge checks",
            "workspace": "workspace inspection",
            "report": "reports",
            "memory": "memory review",
        }
        available = [
            bundle_labels[bundle_id]
            for bundle_id in tool_map.keys()
            if bundle_id in bundle_labels
        ] or ["general reasoning"]
        capabilities_text = cls._join_human_list(available)
        overview = knowledge_service.overview() if knowledge_service is not None else {}
        category_names = [
            str(item.get("category") or "").strip().replace("_", " ")
            for item in overview.get("categories", [])
            if isinstance(item, dict) and str(item.get("category") or "").strip()
        ][:7]
        if "math" not in category_names:
            all_category_names = [
                str(item.get("category") or "").strip().replace("_", " ")
                for item in overview.get("categories", [])
                if isinstance(item, dict) and str(item.get("category") or "").strip()
            ]
            if "math" in all_category_names:
                category_names = [*category_names[:6], "math"]
        knowledge_text = cls._join_human_list(category_names) if category_names else "a small local knowledge base"

        if category == "identity":
            if style == "direct":
                parts = [
                    ("lead", (
                        "I'm Lumen. I aim to be clear, grounded, and useful.",
                        "I'm Lumen. I try to be clear, calm, and useful.",
                        "I'm Lumen. I try to be practical, grounded, and easy to work with.",
                    )),
                    ("close", (
                        f"I can help with {capabilities_text}, and my local knowledge is strongest around {knowledge_text}.",
                        f"I can help across {capabilities_text}, with the strongest local coverage around {knowledge_text}.",
                        f"I work across {capabilities_text}, with the strongest local footing around {knowledge_text}.",
                    )),
                ]
            elif style == "collab":
                parts = [
                    ("lead", (
                        "I'm Lumen. I try to show up as a thoughtful, grounded partner in the conversation.",
                        "I'm Lumen. I aim to be clear, steady, and easy to work through things with.",
                        "I'm Lumen. I try to be present, grounded, and genuinely useful while we work through things together.",
                    )),
                    ("close", (
                        f"I can help with things like {capabilities_text}, and my local knowledge is strongest around {knowledge_text}.",
                        f"I can work across {capabilities_text}, and my strongest local knowledge coverage is around {knowledge_text}.",
                        f"I can help with a mix of {capabilities_text}, with the strongest local grounding around {knowledge_text}.",
                    )),
                ]
            else:
                parts = [
                    ("lead", (
                        "I'm Lumen. I try to be clear, grounded, and helpful.",
                        "I'm Lumen. I aim to be calm, practical, and useful.",
                        "I'm Lumen. I try to be steady, clear, and helpful.",
                    )),
                    ("close", (
                        f"I can help with {capabilities_text}, and my local knowledge is strongest around {knowledge_text}.",
                        f"I work across {capabilities_text}, with the strongest local coverage around {knowledge_text}.",
                        f"I can help across {capabilities_text}, with the strongest local grounding around {knowledge_text}.",
                    )),
                ]
        elif category == "knowledge":
            if style == "direct":
                parts = [
                    ("lead", (
                        f"I can help with {capabilities_text}.",
                        f"I work across {capabilities_text}.",
                        f"I handle {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge is strongest around {knowledge_text}.",
                        f"Locally, I know the most about {knowledge_text}.",
                        f"My strongest local coverage is {knowledge_text}.",
                    )),
                ]
            elif style == "collab":
                parts = [
                    ("lead", (
                        f"I can help with things like {capabilities_text}.",
                        f"I can work across {capabilities_text}.",
                        f"I can help with a mix of {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge is strongest around {knowledge_text}, so I can explain, compare, and pressure-test ideas there with you.",
                        f"My local knowledge is strongest around {knowledge_text}, which gives us a good base for explanation and comparison.",
                        f"My strongest local knowledge coverage is {knowledge_text}, so we can use that to work through questions together.",
                    )),
                ]
            else:
                parts = [
                    ("lead", (
                        f"I can help with {capabilities_text}.",
                        f"I can work across {capabilities_text}.",
                        f"I can help with tasks like {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge is strongest around {knowledge_text}.",
                        f"My strongest local knowledge coverage is {knowledge_text}.",
                        f"My local knowledge is strongest around {knowledge_text}.",
                    )),
                    ("close", (
                        "That gives me a good base for explanation, comparison, and general reasoning.",
                        "That makes me most useful for explaining and comparing ideas in those areas.",
                        "That gives me a solid base for explaining concepts and working through questions there.",
                    )),
                ]
        else:
            if style == "direct":
                parts = [
                    ("lead", (
                        f"I can help with {capabilities_text}.",
                        f"I handle {capabilities_text}.",
                        f"I can work with {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge covers {knowledge_text}.",
                        f"I also know about {knowledge_text}.",
                        f"My local knowledge is around {knowledge_text}.",
                    )),
                ]
            elif style == "collab":
                parts = [
                    ("lead", (
                        f"I can help with things like {capabilities_text}.",
                        f"I can help across {capabilities_text}.",
                        f"I can work with a mix of {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge is strongest around {knowledge_text}, so I can explain, compare, or check ideas with you there.",
                        f"My local knowledge is strongest around {knowledge_text}, which gives us a strong base for working through those topics together.",
                        f"My strongest local knowledge coverage is {knowledge_text}, so we can explore those areas in a grounded way.",
                    )),
                ]
            else:
                parts = [
                    ("lead", (
                        f"I can help with {capabilities_text}.",
                        f"I can work across {capabilities_text}.",
                        f"I can help with tasks like {capabilities_text}.",
                    )),
                    ("knowledge", (
                        f"My local knowledge covers {knowledge_text}.",
                        f"My strongest local knowledge is around {knowledge_text}.",
                        f"My local knowledge is strongest around {knowledge_text}.",
                    )),
                    ("close", (
                        "That lets me explain concepts, compare related ideas, and work through questions in those areas.",
                        "That gives me a useful base for explanation, comparison, and structured reasoning there.",
                        "That makes me most useful for grounded explanation and comparison in those areas.",
                    )),
                ]

        reply = ResponseVariationLayer.realize(
            parts=parts,
            seed_parts=[normalized, style, capabilities_text, knowledge_text, "self_overview"],
            recent_texts=recent_texts,
        )
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.self_overview",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["self_overview_focus"] = category
        response["self_overview_source"] = source
        return response

    @classmethod
    def _memory_overview(
        cls,
        *,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object]:
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        if style == "direct":
            pool = (
                "I can keep track of saved details and recent research when it's relevant, but I do not expose personal memory unless you ask directly.",
                "I can remember saved details and recent research, but I keep that bounded and specific.",
                "I can keep saved details and recent research in view, but I do not assume personal memory exposure.",
            )
        elif style == "collab":
            pool = (
                "I can keep track of saved details and recent research when it's helpful, but I keep that bounded. If you want to know what I remember about you or this thread, ask directly and I'll stay specific.",
                "I can hold onto saved details and recent research when it helps, but I try to keep that contained. If you want the exact memory picture, ask directly and I'll answer specifically.",
                "I can remember saved details and recent research when it's useful, but I keep it bounded. If you want to check what I remember about you or this thread, ask directly and I'll keep it precise.",
            )
        else:
            pool = (
                "I can keep track of saved details and recent research when it's relevant, but I keep that bounded. If you want to know what I remember about you or the current thread, ask directly and I'll answer specifically.",
                "I can remember saved details and recent research when it's useful, but I keep that scoped. If you want to inspect what I remember, ask directly and I'll stay specific.",
                "I can keep saved details and recent research in view when it helps, but I keep that bounded. If you want the exact memory picture, ask directly and I'll answer clearly.",
            )
        reply = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style, "memory_overview"],
            recent_texts=recent_texts,
        )
        return ConversationResponse(
            mode="conversation",
            kind="conversation.memory_overview",
            summary=reply,
            reply=reply,
        ).to_dict()

    @classmethod
    def classify_prompt(
        cls,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]] | None = None,
    ) -> dict[str, str] | None:
        normalized = PromptSurfaceBuilder.build(prompt).lookup_ready_text
        if normalized in cls.MEMORY_PROMPTS:
            return {"category": "memory", "source": "direct", "normalized": normalized}
        if normalized in cls.CAPABILITY_PROMPTS:
            return {"category": "capabilities", "source": "direct", "normalized": normalized}
        if normalized in cls.KNOWLEDGE_PROMPTS:
            return {"category": "knowledge", "source": "direct", "normalized": normalized}
        if normalized in cls.IDENTITY_PROMPTS:
            source = "social_follow_up" if normalized in cls.SOCIAL_FOLLOW_UP_PROMPTS else "direct"
            return {"category": "identity", "source": source, "normalized": normalized}
        if (
            normalized in cls.SOCIAL_FOLLOW_UP_PROMPTS
            and cls.has_recent_conversational_context(recent_interactions or [])
        ):
            return {"category": "identity", "source": "social_follow_up", "normalized": normalized}
        return None

    @classmethod
    def looks_like_self_referential_prompt(
        cls,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]] | None = None,
    ) -> bool:
        classification = cls.classify_prompt(
            prompt=prompt,
            recent_interactions=recent_interactions,
        )
        return bool(classification and classification.get("category") in {"identity", "capabilities", "knowledge"})

    @staticmethod
    def has_recent_conversational_context(recent_interactions: list[dict[str, object]]) -> bool:
        for item in recent_interactions[:3]:
            mode = str(item.get("mode") or "").strip().lower()
            kind = str(item.get("kind") or "").strip().lower()
            response_payload = item.get("response") if isinstance(item.get("response"), dict) else {}
            interaction_mode = str(
                response_payload.get("interaction_mode")
                or item.get("interaction_mode")
                or ""
            ).strip().lower()
            if mode == "conversation" or kind.startswith("conversation.") or interaction_mode == "social":
                return True
        return False

    @staticmethod
    def _join_human_list(items: list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1] + [f"and {items[-1]}"])
