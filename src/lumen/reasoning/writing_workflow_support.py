from __future__ import annotations

import re


class WritingWorkflowSupport:
    """Detects bounded writing/editing workflows that can use hosted inference when available."""

    _TRANSLATE_RE = re.compile(r"^(translate)\b", flags=re.IGNORECASE)
    _REWRITE_RE = re.compile(r"^(rewrite|rephrase|reword)\b", flags=re.IGNORECASE)
    _PARAPHRASE_RE = re.compile(r"^(paraphrase)\b", flags=re.IGNORECASE)
    _CLEANUP_RE = re.compile(r"^(proofread|clean up|cleanup|autocorrect|fix grammar)\b", flags=re.IGNORECASE)
    _DRAFT_RE = re.compile(r"^(draft|write)\b", flags=re.IGNORECASE)

    @classmethod
    def classify(cls, prompt: str) -> dict[str, object] | None:
        text = " ".join(str(prompt or "").strip().split())
        lowered = text.lower()
        if not lowered:
            return None
        if cls._TRANSLATE_RE.match(text):
            target = cls._extract_translation_target(lowered)
            return {
                "workflow": "translation",
                "label": "translation",
                "target_language": target,
            }
        if cls._REWRITE_RE.match(text):
            tone = cls._extract_tone(lowered)
            return {
                "workflow": "rewrite",
                "label": "rewrite",
                "tone_target": tone,
                "academic_genre": cls._extract_genre(lowered),
            }
        if cls._PARAPHRASE_RE.match(text):
            return {
                "workflow": "paraphrase",
                "label": "paraphrase",
                "academic_genre": cls._extract_genre(lowered),
            }
        if cls._CLEANUP_RE.match(text):
            return {
                "workflow": "cleanup",
                "label": "cleanup",
                "academic_genre": cls._extract_genre(lowered),
            }
        if cls._DRAFT_RE.match(text) and any(
            token in lowered for token in ("email", "essay", "note", "script", "message", "paragraph", "bio")
        ):
            return {
                "workflow": "draft",
                "label": "draft",
                "academic_genre": cls._extract_genre(lowered),
            }
        return None

    @staticmethod
    def hosted_instructions(*, workflow: dict[str, object], interaction_style: str, reasoning_depth: str) -> str:
        style = str(interaction_style or "default").strip().lower()
        depth = str(reasoning_depth or "normal").strip().lower()
        workflow_name = str(workflow.get("label") or "writing task")
        genre = str(workflow.get("academic_genre") or "").strip()
        base = (
            "You are Lumen performing a bounded writing/editing task. "
            "Return only the user-facing result. "
            "Do not claim hidden tools, internal states, or policy scaffolding. "
            "Keep uncertainty explicit if the source text is ambiguous. "
            "Do not fabricate citations or unsupported factual claims."
        )
        if style == "direct":
            style_line = "Keep the wording concise and practical."
        elif style == "collab":
            style_line = "Keep the tone warm and natural while staying useful."
        else:
            style_line = "Keep the tone balanced and grounded."
        if depth == "deep":
            depth_line = "Be thorough only when the writing task clearly benefits from it."
        else:
            depth_line = "Prefer a clean final output over extra explanation."
        workflow_line = f"The requested workflow is {workflow_name}."
        genre_line = f"Use the conventions of {genre}." if genre else ""
        return f"{base} {workflow_line} {genre_line} {style_line} {depth_line}".strip()

    @staticmethod
    def provider_gated_message(*, workflow: dict[str, object], provider_reason: str) -> str:
        label = str(workflow.get("label") or "writing/editing").strip()
        if label == "translation":
            return (
                "Translation-style writing help is available only when a hosted provider is configured. "
                f"Current runtime status: {provider_reason}"
            )
        if label == "cleanup":
            return (
                "Proofread, cleanup, and autocorrect workflows are available only when a hosted provider is configured. "
                f"Current runtime status: {provider_reason}"
            )
        if label == "rewrite":
            return (
                "Rewrite and tone-refinement workflows are available only when a hosted provider is configured. "
                f"Current runtime status: {provider_reason}"
            )
        if label == "paraphrase":
            return (
                "Paraphrase workflows are available only when a hosted provider is configured, and they should still be reviewed for originality and citation integrity. "
                f"Current runtime status: {provider_reason}"
            )
        return (
            "Drafting support for this writing task is available only when a hosted provider is configured. "
            f"Current runtime status: {provider_reason}"
        )

    @staticmethod
    def _extract_translation_target(prompt: str) -> str | None:
        match = re.search(r"\bto\s+([a-z][a-z ]+)$", prompt)
        if match is None:
            return None
        return match.group(1).strip(" .!?")

    @staticmethod
    def _extract_tone(prompt: str) -> str | None:
        match = re.search(r"\b(in|into)\s+(a\s+)?([a-z -]+)\s+tone\b", prompt)
        if match is None:
            return None
        return match.group(3).strip(" .!?")

    @staticmethod
    def _extract_genre(prompt: str) -> str | None:
        for genre in (
            "analytical essay",
            "persuasive essay",
            "literature analysis",
            "research summary",
            "research report",
            "lab report",
        ):
            if genre in prompt:
                return genre
        return None
