from __future__ import annotations

import re

from lumen.nlu.models import NormalizedTopic


class TopicNormalizer:
    """Builds a lightweight normalized topic phrase from a prompt."""

    LEADING_PATTERNS = (
        r"^create\s+",
        r"^build\s+",
        r"^make\s+",
        r"^draft\s+",
        r"^design\s+",
        r"^propose\s+",
        r"^summarize\s+",
        r"^summary of\s+",
        r"^explain\s+",
        r"^compare\s+",
        r"^what about\s+",
        r"^how about\s+",
        r"^expand\s+",
        r"^continue with\s+",
        r"^continue\s+",
        r"^review\s+",
    )
    STOPWORDS = {
        "the",
        "a",
        "an",
        "for",
        "of",
        "and",
        "to",
        "with",
        "that",
        "this",
        "it",
        "we",
        "i",
    }

    def normalize(self, text: str) -> NormalizedTopic:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return NormalizedTopic(value=None, tokens=())

        topic = normalized
        for pattern in self.LEADING_PATTERNS:
            topic = re.sub(pattern, "", topic)
        topic = re.sub(r"^(a|an|the)\s+", "", topic)
        topic = re.sub(r"\s+", " ", topic).strip(" ?!.,")
        if not topic:
            return NormalizedTopic(value=None, tokens=())

        tokens = tuple(
            token
            for token in re.findall(r"[a-z0-9_]+", topic)
            if token not in self.STOPWORDS and len(token) > 2
        )
        return NormalizedTopic(
            value=topic or None,
            tokens=tokens,
        )
