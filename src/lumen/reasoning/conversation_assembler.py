from __future__ import annotations

from typing import Any

from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_variation import ResponseVariationLayer


class ConversationAssembler:
    """Lightweight composition layer for conversational surface responses."""

    @classmethod
    def assemble(
        cls,
        *,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None = None,
        opener: str | tuple[str, ...] | None = None,
        stance: str | tuple[str, ...] | None = None,
        content: str | tuple[str, ...] | None = None,
        closer: str | tuple[str, ...] | None = None,
    ) -> str:
        normalized_style = InteractionStylePolicy.normalize_style(style)
        fragments = [
            cls._resolve_fragment(
                fragment=fragment,
                label=label,
                style=normalized_style,
                seed_parts=seed_parts,
                recent_texts=recent_texts,
            )
            for label, fragment in (
                ("opener", opener),
                ("stance", stance),
                ("content", content),
                ("closer", closer),
            )
        ]
        pieces = [piece for piece in fragments if piece]
        return cls._join_parts(pieces)

    @staticmethod
    def _resolve_fragment(
        *,
        fragment: str | tuple[str, ...] | None,
        label: str,
        style: str,
        seed_parts: list[str],
        recent_texts: list[str] | None,
    ) -> str:
        if fragment is None:
            return ""
        if isinstance(fragment, tuple):
            return ResponseVariationLayer.select_from_pool(
                fragment,
                seed_parts=[style, label, *seed_parts],
                recent_texts=recent_texts,
            ).strip()
        return str(fragment).strip()

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        if not parts:
            return ""
        assembled = parts[0]
        for part in parts[1:]:
            if not part:
                continue
            if assembled.endswith(("—", "-", ":", ";")):
                assembled = f"{assembled} {part}"
            elif part.startswith((",", ".", "?", "!", ";", ":")):
                assembled = f"{assembled}{part}"
            else:
                assembled = f"{assembled} {part}"
        return " ".join(assembled.split()).strip()
