from __future__ import annotations

from lumen.nlu.prompt_nlu import PromptNLU


class DesignSurfaceSupport:
    """Owns lightweight design-intake normalization and planning carryover."""

    _prompt_nlu = PromptNLU()

    DIRECT_STARTERS = (
        "design me an ",
        "design me a ",
        "design me ",
        "design an ",
        "design a ",
        "design ",
        "build me an ",
        "build me a ",
        "build me ",
        "build an ",
        "build a ",
        "build ",
        "come up with a design for ",
        "come up with an idea for ",
        "sketch a concept for ",
        "prototype a ",
        "prototype an ",
        "invent a ",
        "invent an ",
        "invent ",
        "concept for ",
    )

    DESIGN_NOUNS = (
        "engine",
        "propulsion",
        "thruster",
        "motor",
        "system",
        "reactor",
        "device",
        "mechanism",
        "prototype",
        "concept",
        "architecture",
        "machine",
        "vehicle",
        "rocket",
        "api",
        "service",
        "workflow",
        "pipeline",
        "agent",
        "assistant",
        "dashboard",
        "platform",
        "app",
        "application",
        "cli",
    )
    SOFTWARE_DESIGN_NOUNS = (
        "api",
        "service",
        "workflow",
        "pipeline",
        "agent",
        "assistant",
        "dashboard",
        "platform",
        "app",
        "application",
        "cli",
        "architecture",
        "system",
    )

    FOLLOW_UP_PROMPTS = {
        "go on",
        "tell me more",
        "what else",
        "why",
        "what do you mean",
        "how so",
    }

    @classmethod
    def rewrite_prompt(
        cls,
        *,
        prompt: str,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str:
        normalized = cls._lookup_ready_text(prompt)
        if not normalized:
            return prompt

        if normalized in cls.FOLLOW_UP_PROMPTS:
            carryover = cls._follow_up_prompt(
                recent_interactions=recent_interactions,
                active_thread=active_thread,
            )
            if carryover:
                return carryover

        if any(normalized.startswith(prefix) for prefix in cls.DIRECT_STARTERS):
            subject = cls.extract_subject(normalized)
            if subject:
                return f"create a {subject} design"
        if normalized.startswith("improve this design"):
            return "refine the current design and identify the main tradeoffs"
        if normalized.startswith("suggest failure modes for this design"):
            return "analyze this design for likely failure modes"
        if normalized.startswith("how would you structure this system"):
            return "design the architecture for this system"
        return prompt

    @classmethod
    def extract_subject(cls, prompt: str) -> str:
        normalized = cls._lookup_ready_text(prompt)
        for prefix in cls.DIRECT_STARTERS:
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip(" .!?")
        return normalized.strip(" .!?")

    @classmethod
    def _follow_up_prompt(
        cls,
        *,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
    ) -> str | None:
        if active_thread is not None and str(active_thread.get("mode") or "").strip() == "planning":
            topic = str(active_thread.get("prompt") or active_thread.get("thread_summary") or "").strip()
            if topic and cls._is_designish(topic):
                return cls._expand_design_prompt(topic)

        if not recent_interactions:
            return None
        latest = recent_interactions[0]
        latest_mode = str(latest.get("mode") or "").strip()
        latest_kind = str(latest.get("kind") or "").strip()
        if latest_mode != "planning" and not latest_kind.startswith("planning."):
            return None
        latest_prompt = str(latest.get("prompt") or "").strip()
        latest_summary = str(latest.get("summary") or "").strip()
        if latest_prompt and cls._is_designish(latest_prompt):
            return cls._expand_design_prompt(latest_prompt)
        if latest_summary and cls._is_designish(latest_summary):
            return f"continue this design direction: {latest_summary}"
        return None

    @classmethod
    def _expand_design_prompt(cls, topic: str) -> str:
        normalized = cls._lookup_ready_text(topic)
        subject = cls.extract_subject(normalized)
        if subject and any(token in subject for token in cls.DESIGN_NOUNS):
            return f"create a {subject} design with another concrete layer"
        if any(normalized.startswith(prefix) for prefix in cls.DIRECT_STARTERS):
            return f"{normalized} and add another concrete layer"
        return f"design {normalized} and add another concrete layer"

    @classmethod
    def _is_designish(cls, text: str) -> bool:
        normalized = cls._lookup_ready_text(text)
        return any(token in normalized for token in cls.DESIGN_NOUNS) or any(
            normalized.startswith(prefix) for prefix in cls.DIRECT_STARTERS
        )

    @classmethod
    def should_use_design_tool(cls, *, prompt: str, route_kind: str) -> bool:
        normalized = cls._lookup_ready_text(prompt)
        if str(route_kind or "").strip() != "planning.architecture":
            return False
        if not normalized:
            return False
        if not any(token in normalized for token in cls.SOFTWARE_DESIGN_NOUNS):
            return False
        if any(normalized.startswith(prefix) for prefix in cls.DIRECT_STARTERS):
            return True
        return any(token in normalized for token in ("architecture", "system", "workflow", "pipeline", "api", "service"))

    @classmethod
    def build_design_tool_params(
        cls,
        *,
        prompt: str,
        interaction_style: str,
    ) -> dict[str, object]:
        normalized = cls._lookup_ready_text(prompt)
        return {
            "brief": normalized or prompt,
            "interaction_style": interaction_style,
        }

    @classmethod
    def _lookup_ready_text(cls, prompt: str) -> str:
        return cls._prompt_nlu.analyze(prompt).surface_views.lookup_ready_text
