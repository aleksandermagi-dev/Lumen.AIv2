from __future__ import annotations

from typing import Any

from lumen.reasoning.interaction_style_policy import InteractionStylePolicy


class ResponseVariationLayer:
    """Lightweight surface-realization helper with small pools and anti-repetition."""

    ACKNOWLEDGMENT_POOL = ("yeah", "gotcha", "right", "okay", "makes sense", "I see")
    TRANSITION_POOL = ("so", "alright", "now", "then", "wait", "hold on")
    THINKING_SIGNAL_POOL = ("hmm", "interesting", "let's see", "I think", "could be")
    REACTION_POOL = ("oh", "ah", "wait", "okay yeah", "that's nice", "fair")
    SOFTENER_POOL = ("kinda", "a bit", "maybe", "probably", "sort of")
    DIRECT_GOODBYE_POOL = (
        "Okay.",
        "Later.",
        "Talk later.",
        "See you.",
        "Alright. Later.",
        "Alright.",
        "Take care.",
        "Sounds good.",
        "We'll pick it up later.",
    )
    DEFAULT_GOODBYE_POOL = (
        "See you.",
        "Talk soon.",
        "Take care.",
        "Alright, talk soon.",
        "Sounds good, talk soon.",
        "Alright, we'll pick this up later.",
        "Let me know when you're back.",
        "See you later.",
        "Sounds good. I'll be here.",
        "Take care. We can pick it up later.",
        "Alright. Come back when you're ready.",
    )
    COLLAB_GOODBYE_POOL = (
        "See you later. I got you when you're back.",
        "Talk soon. I'll be here.",
        "Take care. We'll pick this up later.",
        "Alright, I'll be here.",
        "Cool, I'll be here when you're ready.",
        "Let me know when you're back.",
        "Sounds good, talk soon.",
        "Take care. We can pick this back up whenever you want.",
        "Alright. I'll be here when you want to jump back in.",
        "See you later. We'll pick it right back up when you want.",
        "Take care. I'll hold the thread for you.",
        "Sounds good. I'll be right here when you want back in.",
    )
    DIRECT_TERMINAL_GOODBYE_POOL = (
        "Sounds good. See you later.",
        "Talk soon.",
        "Later.",
        "Alright. Talk later.",
    )
    DEFAULT_TERMINAL_GOODBYE_POOL = (
        "Sounds good. See you later.",
        "Thanks. Talk soon.",
        "Good work today. See you later.",
        "Take care. Talk soon.",
    )
    COLLAB_TERMINAL_GOODBYE_POOL = (
        "Sounds good. Talk soon.",
        "Absolutely. See you later.",
        "Glad we got somewhere today. Talk soon.",
        "Good work today. We'll pick it back up when you want.",
        "Take care. We'll keep going when you're back.",
    )
    DIRECT_GRATITUDE_POOL = (
        "You're welcome.",
        "Any time.",
        "Yep.",
        "Glad it helped.",
        "No problem.",
        "Of course.",
    )
    DEFAULT_GRATITUDE_POOL = (
        "You're welcome.",
        "Of course.",
        "Glad to help.",
        "Any time.",
        "Happy to help.",
        "No problem.",
        "Glad that helped.",
        "You're welcome. Happy to help.",
    )
    COLLAB_GRATITUDE_POOL = (
        "You're welcome.",
        "Of course. Glad to help.",
        "Any time. Happy to be in it with you.",
        "Glad that helped.",
        "Absolutely. I'm glad that was useful.",
        "Any time. We can pick it back up whenever you want.",
        "You're welcome. Glad we got somewhere with it.",
        "Of course. I'm glad that landed.",
        "Any time. Happy to stay with it with you.",
    )
    SHAPE_VERB_POOLS = {
        "neutral": ("explain", "outline", "clarify", "summarize"),
        "technical": ("analyze", "specify", "structure", "refine"),
        "creative": ("explore", "develop", "shape", "extend"),
        "light": ("walk through", "talk through", "unpack", "frame"),
    }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @classmethod
    def recent_surface_texts(cls, recent_interactions: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for item in recent_interactions[:5]:
            cls._collect_texts(texts, item)
            response = item.get("response")
            if isinstance(response, dict):
                cls._collect_texts(texts, response)
        return texts

    @classmethod
    def select_from_pool(
        cls,
        pool: tuple[str, ...],
        *,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        if len(pool) == 1:
            return pool[0]
        recent_norm = {cls._normalize(text) for text in recent_texts or [] if str(text).strip()}
        available = [item for item in pool if cls._normalize(item) not in recent_norm]
        options = available or list(pool)
        seed = "|".join(seed_parts)
        index = sum(ord(char) for char in seed) % len(options)
        return options[index]

    @classmethod
    def realize(
        cls,
        *,
        parts: list[tuple[str, tuple[str, ...]]],
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
        separator: str = " ",
    ) -> str:
        realized: list[str] = []
        local_recent = list(recent_texts or [])
        for label, pool in parts:
            choice = cls.select_from_pool(
                pool,
                seed_parts=[*seed_parts, label],
                recent_texts=local_recent,
            )
            if choice:
                realized.append(choice)
                local_recent.append(choice)
        candidate = separator.join(part.strip() for part in realized if part.strip())
        if not candidate:
            return ""
        recent_norm = {cls._normalize(text) for text in recent_texts or [] if str(text).strip()}
        if cls._normalize(candidate) not in recent_norm:
            return candidate
        for index in range(len(parts) - 1, -1, -1):
            label, pool = parts[index]
            current = realized[index]
            alternatives = [
                item
                for item in pool
                if item != current and cls._normalize(item) not in recent_norm
            ]
            if not alternatives:
                continue
            fallback_seed = "|".join([*seed_parts, label, "fallback"])
            replacement = alternatives[sum(ord(char) for char in fallback_seed) % len(alternatives)]
            realized[index] = replacement
            updated = separator.join(part.strip() for part in realized if part.strip())
            if cls._normalize(updated) not in recent_norm:
                return updated
        return candidate

    @classmethod
    def _collect_texts(cls, target: list[str], item: dict[str, Any]) -> None:
        for key in (
            "reply",
            "summary",
            "response_intro",
            "response_opening",
            "clarification_question",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                target.append(value.strip())
        for key in ("steps", "findings"):
            value = item.get(key)
            if isinstance(value, list):
                for entry in value[:2]:
                    if isinstance(entry, str) and entry.strip():
                        target.append(entry.strip())

    @classmethod
    def conversational_token(
        cls,
        *,
        token_type: str,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        mode_profile = InteractionStylePolicy.mode_profile({"interaction_style": normalized_style})
        word_pool_usage = str(mode_profile.get("word_pool_usage") or "medium")
        if normalized_style == "direct" and token_type in {"softener", "reaction"}:
            return ""
        if word_pool_usage == "low" and token_type in {"acknowledgment", "transition", "reaction", "softener"}:
            return ""
        if word_pool_usage == "medium" and token_type in {"reaction", "softener"}:
            return ""

        pool = {
            "acknowledgment": cls.ACKNOWLEDGMENT_POOL,
            "transition": cls.TRANSITION_POOL,
            "thinking": cls.THINKING_SIGNAL_POOL,
            "reaction": cls.REACTION_POOL,
            "softener": cls.SOFTENER_POOL,
        }.get(token_type)
        if not pool:
            return ""

        return cls.select_from_pool(
            pool,
            seed_parts=[normalized_style, token_type, *seed_parts],
            recent_texts=recent_texts,
        )

    @classmethod
    def style_bridge(
        cls,
        *,
        style: str,
        bridge_type: str,
        base: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        mode_profile = InteractionStylePolicy.mode_profile({"interaction_style": normalized_style})
        trimmed = str(base or "").strip()
        if not trimmed:
            return ""

        if mode_profile.get("word_pool_usage") == "low":
            if bridge_type == "thinking":
                token = cls.conversational_token(
                    token_type="thinking",
                    style=normalized_style,
                    seed_parts=seed_parts,
                    recent_texts=recent_texts,
                )
                return f"{token}, {trimmed}" if token else trimmed
            return trimmed

        if mode_profile.get("word_pool_usage") == "medium":
            token_type = "transition" if bridge_type in {"transition", "flow"} else "thinking"
            token = cls.conversational_token(
                token_type=token_type,
                style=normalized_style,
                seed_parts=seed_parts,
                recent_texts=recent_texts,
            )
            return f"{token}, {trimmed}" if token else trimmed

        if bridge_type == "softened":
            reaction = cls.conversational_token(
                token_type="reaction",
                style=normalized_style,
                seed_parts=[*seed_parts, "reaction"],
                recent_texts=recent_texts,
            )
            softener = cls.conversational_token(
                token_type="softener",
                style=normalized_style,
                seed_parts=[*seed_parts, "softener"],
                recent_texts=recent_texts,
            )
            if reaction and softener:
                return f"{reaction}, {trimmed} {softener}".strip()
            if reaction:
                return f"{reaction}, {trimmed}"
            if softener:
                return f"{trimmed} {softener}".strip()
            return trimmed

        token_type = "acknowledgment" if bridge_type == "acknowledgment" else "transition"
        token = cls.conversational_token(
            token_type=token_type,
            style=normalized_style,
            seed_parts=seed_parts,
            recent_texts=recent_texts,
        )
        return f"{token}, {trimmed}" if token else trimmed

    @classmethod
    def response_shape(
        cls,
        *,
        prompt: str,
        interaction_style: str,
        route_mode: str,
    ) -> str:
        normalized_prompt = cls._normalize(prompt)
        style = InteractionStylePolicy.normalize_style(interaction_style)
        if " like " in normalized_prompt or normalized_prompt.startswith("like "):
            return "analogy"
        if any(token in normalized_prompt for token in ("break down", "step by step", "walk me through")):
            return "breakdown"
        if normalized_prompt.startswith(("summarize ", "summary of ", "in short")):
            return "summary"
        if style == "direct":
            return "direct_explanation"
        if route_mode == "planning":
            return "breakdown"
        if style == "collab":
            return "breakdown"
        return "direct_explanation"

    @classmethod
    def choose_shape_verb(
        cls,
        *,
        prompt: str,
        interaction_style: str,
        route_mode: str,
        serious_topic: bool = False,
        recent_texts: list[str] | None = None,
    ) -> str:
        style = InteractionStylePolicy.normalize_style(interaction_style)
        if serious_topic:
            pool_key = "neutral"
        elif route_mode == "planning":
            pool_key = "technical"
        elif style == "collab":
            pool_key = "light"
        elif style == "direct":
            pool_key = "neutral"
        else:
            pool_key = "technical"
        return cls.select_from_pool(
            cls.SHAPE_VERB_POOLS[pool_key],
            seed_parts=[prompt, style, route_mode, pool_key, "verb_control"],
            recent_texts=recent_texts,
        )

    @classmethod
    def goodbye_phrase(
        cls,
        *,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        if normalized_style == "direct":
            pool = cls.DIRECT_GOODBYE_POOL
        elif normalized_style == "collab":
            pool = cls.COLLAB_GOODBYE_POOL
        else:
            pool = cls.DEFAULT_GOODBYE_POOL
        return cls.select_from_pool(
            pool,
            seed_parts=[normalized_style, "goodbye", *seed_parts],
            recent_texts=recent_texts,
        )

    @classmethod
    def terminal_goodbye_phrase(
        cls,
        *,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        if normalized_style == "direct":
            pool = cls.DIRECT_TERMINAL_GOODBYE_POOL
        elif normalized_style == "collab":
            pool = cls.COLLAB_TERMINAL_GOODBYE_POOL
        else:
            pool = cls.DEFAULT_TERMINAL_GOODBYE_POOL
        return cls.select_from_pool(
            pool,
            seed_parts=[normalized_style, "terminal_goodbye", *seed_parts],
            recent_texts=recent_texts,
        )

    @classmethod
    def gratitude_phrase(
        cls,
        *,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        if normalized_style == "direct":
            pool = cls.DIRECT_GRATITUDE_POOL
        elif normalized_style == "collab":
            pool = cls.COLLAB_GRATITUDE_POOL
        else:
            pool = cls.DEFAULT_GRATITUDE_POOL
        return cls.select_from_pool(
            pool,
            seed_parts=[normalized_style, "gratitude", *seed_parts],
            recent_texts=recent_texts,
        )

