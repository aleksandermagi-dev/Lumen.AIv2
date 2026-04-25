from __future__ import annotations

from typing import Any


class InteractionStylePolicy:
    """Owns presentation-layer style decisions without changing truthfulness rules."""

    MODE_PROFILES = {
        "default": {
            "structure": "balanced",
            "word_pool_usage": "medium",
            "looseness": "moderate",
        },
        "collab": {
            "structure": "loose",
            "word_pool_usage": "high",
            "looseness": "high",
        },
        "direct": {
            "structure": "tight",
            "word_pool_usage": "low",
            "looseness": "low",
        },
    }
    VOICE_PROFILES = {
        "default": {
            "voice_profile": "calm_grounded",
            "tone_signature": "clear_grounded_warm",
            "presence": "steady",
        },
        "collab": {
            "voice_profile": "warm_partner",
            "tone_signature": "present_warm_partnered",
            "presence": "high",
        },
        "direct": {
            "voice_profile": "crisp_focused",
            "tone_signature": "concise_confident_clean",
            "presence": "low_friction",
        },
    }

    @staticmethod
    def interaction_style(profile: Any) -> str:
        if hasattr(profile, "interaction_style"):
            raw = str(getattr(profile, "interaction_style", None) or "conversational")
            return InteractionStylePolicy.normalize_style(raw)
        if isinstance(profile, dict):
            raw = str(profile.get("interaction_style") or "conversational")
            return InteractionStylePolicy.normalize_style(raw)
        return "collab"

    @staticmethod
    def normalize_style(style: str) -> str:
        normalized = str(style or "default").strip().lower()
        if normalized == "conversational":
            return "collab"
        if normalized in {"default", "collab", "direct"}:
            return normalized
        return "default"

    @staticmethod
    def reasoning_depth(profile: Any) -> str:
        if hasattr(profile, "reasoning_depth"):
            return str(getattr(profile, "reasoning_depth", None) or "normal")
        if isinstance(profile, dict):
            return str(profile.get("reasoning_depth") or "normal")
        return "normal"

    @classmethod
    def is_direct(cls, profile: Any) -> bool:
        return cls.interaction_style(profile) == "direct"

    @classmethod
    def is_conversational(cls, profile: Any) -> bool:
        return cls.interaction_style(profile) == "collab"

    @classmethod
    def is_default(cls, profile: Any) -> bool:
        return cls.interaction_style(profile) == "default"

    @classmethod
    def is_deep(cls, profile: Any) -> bool:
        return cls.reasoning_depth(profile) == "deep"

    @classmethod
    def is_light(cls, profile: Any) -> bool:
        return cls.reasoning_depth(profile) == "light"

    @classmethod
    def package_type(cls, *, mode: str, profile: Any) -> str:
        if mode == "tool":
            return "brief"
        if mode in {"planning", "research"}:
            if cls.is_light(profile):
                return "quick"
            if cls.is_deep(profile):
                return "deep"
            return "structured"
        if cls.is_light(profile):
            return "quick"
        if cls.is_deep(profile):
            return "deep"
        if cls.is_direct(profile):
            return "brief"
        return "quick"

    @classmethod
    def validation_advice_limit(cls, profile: Any) -> int:
        if cls.is_deep(profile):
            return 3
        if cls.is_direct(profile):
            return 1
        return 2

    @classmethod
    def follow_up_limit(cls, profile: Any) -> int:
        if cls.is_direct(profile):
            return 1
        return 2

    @classmethod
    def mode_profile(cls, profile: Any) -> dict[str, str]:
        style = cls.interaction_style(profile)
        return dict(cls.MODE_PROFILES.get(style, cls.MODE_PROFILES["default"]))

    @classmethod
    def voice_profile(cls, profile: Any) -> dict[str, str]:
        style = cls.interaction_style(profile)
        return dict(cls.VOICE_PROFILES.get(style, cls.VOICE_PROFILES["default"]))
