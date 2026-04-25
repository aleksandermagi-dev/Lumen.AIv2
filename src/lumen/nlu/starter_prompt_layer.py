from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StarterPromptCategory:
    key: str
    label: str
    prompts: tuple[str, ...]
    word_pool: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StarterPromptAnalysis:
    category_scores: dict[str, int]
    matched_categories: tuple[str, ...]


class StarterPromptLayer:
    """Semantic starter-prompt helper for early guidance and clustered intent cues."""

    CATEGORIES: tuple[StarterPromptCategory, ...] = (
        StarterPromptCategory(
            key="reasoning",
            label="Reasoning / Problem-Solving",
            prompts=(
                "Help me think through this problem step by step",
                "What's the best way to approach this under these constraints?",
                "Poke holes in this idea before I commit to it",
            ),
            word_pool=("think", "think through", "approach", "solve", "break down", "figure out", "work through"),
        ),
        StarterPromptCategory(
            key="science",
            label="Science / Explanation",
            prompts=(
                "Explain black holes simply and clearly",
                "What is entropy, simply but correctly?",
                "Compare a black hole and a neutron star clearly",
                "Explain this concept in relation to another one",
            ),
            word_pool=("explain", "concept", "theory", "compare", "understand", "walk through"),
        ),
        StarterPromptCategory(
            key="builder",
            label="Inventor / Builder Mode",
            prompts=(
                "Design a propulsion concept under these constraints",
                "Suggest failure modes for this design",
                "Help me refine this invention idea into something testable",
            ),
            word_pool=("design", "build", "create", "refine", "prototype", "improve"),
        ),
        StarterPromptCategory(
            key="system",
            label="System / Strategy",
            prompts=(
                "Analyze this system and tell me what breaks first",
                "What's the weak point in this setup?",
                "Help me structure this better without overengineering it",
            ),
            word_pool=("analyze", "structure", "optimize", "weak point", "failure", "system"),
        ),
        StarterPromptCategory(
            key="tools",
            label="Files / Tools",
            prompts=(
                "Analyze this attached file and tell me what it is",
                "Run ANH on this spectrum file",
                "Generate content ideas on this topic",
            ),
            word_pool=("file", "folder", "zip", "attached", "run anh", "generate content", "analyze this file"),
        ),
        StarterPromptCategory(
            key="exploration",
            label="Soft Entry",
            prompts=(
                "I have an idea but I'm not sure if it makes sense yet",
                "Help me turn this rough thought into something structured",
                "I don't know where to start. Can you guide me through this?",
            ),
            word_pool=("idea", "not sure", "confused", "help me", "start", "direction"),
        ),
    )

    @classmethod
    def starter_prompts(cls) -> tuple[StarterPromptCategory, ...]:
        return cls.CATEGORIES

    @classmethod
    def analyze(cls, text: str) -> StarterPromptAnalysis:
        normalized = cls._normalize(text)
        scores: dict[str, int] = {}
        for category in cls.CATEGORIES:
            score = 0
            for phrase in category.word_pool:
                if phrase in normalized:
                    score += 1 if " " not in phrase else 2
            scores[category.key] = score
        matched = tuple(
            category.key
            for category in sorted(
                cls.CATEGORIES,
                key=lambda item: scores.get(item.key, 0),
                reverse=True,
            )
            if scores.get(category.key, 0) > 0
        )
        return StarterPromptAnalysis(
            category_scores=scores,
            matched_categories=matched,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())
