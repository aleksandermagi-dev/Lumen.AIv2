from __future__ import annotations


class ExplanatoryIntentPolicy:
    """Shared prompt-shape helpers for explanatory and subject-style requests."""

    EXPLANATORY_ENTITY_LABELS = {
        "person",
        "place",
        "event",
        "concept",
        "object",
        "formula",
        "process",
        "system",
    }

    TOPIC_ONLY_BLOCKED_PREFIXES = (
        "create ",
        "build ",
        "run ",
        "analyze ",
        "compare ",
        "summarize ",
        "summarise ",
        "report ",
        "inspect ",
        "review ",
        "fix ",
        "implement ",
        "expand ",
        "continue ",
    )

    TOPIC_ONLY_BLOCKED_TOKENS = {
        "hello",
        "hey",
        "hi",
        "yo",
        "thanks",
        "thank",
        "goodbye",
        "bye",
        "later",
        "architecture",
        "direction",
        "routing",
        "constraints",
        "migration",
        "plan",
        "roadmap",
        "expand",
        "continue",
        "further",
        "that",
    }

    BROAD_EXPLANATORY_PREFIXES = (
        "tell me about ",
        "what is ",
        "what are ",
        "what's ",
        "whats ",
        "what was ",
        "who is ",
        "who was ",
        "where is ",
        "where was ",
        "when was ",
        "when did ",
        "explain ",
        "describe ",
        "define ",
    )

    BROAD_EXPLANATORY_BLOCKED_TOKENS = {
        "hello",
        "hey",
        "hi",
        "yo",
        "thanks",
        "thank",
        "goodbye",
        "bye",
        "later",
        "archive",
        "migration",
        "routing",
        "plan",
        "summary",
        "summarize",
        "structure",
        "retrieval",
        "thread",
        "session",
        "current",
        "local",
        "indexed",
        "compare",
        "design",
        "build",
        "implement",
        "debug",
        "validate",
        "workspace",
        "inspect",
        "report",
    }

    BROAD_EXPLANATORY_MARKERS = {
        "formula",
        "equation",
        "law",
        "system",
        "loop",
        "process",
        "cycle",
        "voltage",
    }

    KNOWLEDGE_BLOCKED_PREFIXES = (
        "summarize ",
        "summary of ",
        "review ",
        "analyze ",
        "compare ",
        "contrast ",
        "report ",
        "inspect ",
        "give me a brief direct answer about ",
    )

    SOCIAL_PREFIXES = ("hey ", "hi ", "hello ", "yo ")

    @classmethod
    def looks_like_topic_only_query(cls, normalized_prompt: str) -> bool:
        if not normalized_prompt:
            return False
        if not normalized_prompt or " " not in normalized_prompt:
            return bool(normalized_prompt) and normalized_prompt.isalpha()
        if any(normalized_prompt.startswith(prefix) for prefix in cls.TOPIC_ONLY_BLOCKED_PREFIXES):
            return False
        tokens = normalized_prompt.split()
        if any(token in cls.TOPIC_ONLY_BLOCKED_TOKENS for token in tokens):
            return False
        return len(tokens) <= 4 and normalized_prompt.replace(" ", "").isalpha()

    @classmethod
    def looks_like_explanatory_subject_query(cls, normalized_prompt: str) -> bool:
        if cls.looks_like_topic_only_query(normalized_prompt):
            return True
        return cls.looks_like_broad_explanatory_prompt(normalized_prompt)

    @classmethod
    def looks_like_broad_explanatory_prompt(cls, normalized_prompt: str) -> bool:
        if not normalized_prompt:
            return False
        if normalized_prompt.startswith("prove "):
            return False
        if any(normalized_prompt.startswith(prefix) for prefix in cls.BROAD_EXPLANATORY_PREFIXES):
            return True
        trimmed = normalized_prompt
        for prefix in cls.SOCIAL_PREFIXES:
            if trimmed.startswith(prefix):
                trimmed = trimmed[len(prefix) :].strip()
                break
        if any(trimmed.startswith(prefix) for prefix in ("what is ", "what's ", "whats ")):
            return True
        if cls.is_blocked_knowledge_explanatory_prompt(normalized_prompt):
            return False
        tokens = [token for token in normalized_prompt.replace("?", " ").replace("!", " ").split() if token]
        if not tokens or any(token in cls.BROAD_EXPLANATORY_BLOCKED_TOKENS for token in tokens):
            return False
        if any(token in cls.BROAD_EXPLANATORY_MARKERS for token in tokens):
            return len(tokens) <= 5
        compact = normalized_prompt.replace(" ", "").replace("'", "").replace("-", "")
        return 1 <= len(tokens) <= 5 and compact.isalpha()

    @classmethod
    def is_blocked_knowledge_explanatory_prompt(cls, normalized_prompt: str) -> bool:
        return any(normalized_prompt.startswith(prefix) for prefix in cls.KNOWLEDGE_BLOCKED_PREFIXES)

    @classmethod
    def has_explanatory_entities(
        cls,
        entities: tuple[dict[str, object], ...] | list[dict[str, object]] | None,
    ) -> bool:
        entity_list = list(entities or [])
        return any(
            isinstance(entity, dict)
            and str(entity.get("label") or "").strip().lower() in cls.EXPLANATORY_ENTITY_LABELS
            for entity in entity_list
        )
