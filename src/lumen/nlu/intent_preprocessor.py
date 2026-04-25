from __future__ import annotations


class IntentPreprocessor:
    """Applies intent-facing shaping after surface normalization."""

    SOFT_PREFIXES = (
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "quickly ",
        "yo ",
    )

    @classmethod
    def prepare(cls, normalized_text: str) -> str:
        stripped = " ".join(str(normalized_text or "").strip().lower().split())
        changed = True
        while changed:
            changed = False
            for prefix in cls.SOFT_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
                    break
        return stripped
