from __future__ import annotations

import re

from lumen.nlu.intent_preprocessor import IntentPreprocessor
from lumen.nlu.models import PromptSurfaceViews
from lumen.nlu.text_normalizer import TextNormalizer


class PromptSurfaceBuilder:
    _LEADING_ADDRESS_RE = re.compile(
        r"^(?:(?:hey|hi|hello|yo)\b[\s,]+lumen\b[\s,]*|lumen\b[\s,]*)",
        flags=re.IGNORECASE,
    )

    @classmethod
    def build(cls, text: str) -> PromptSurfaceViews:
        raw_text = str(text or "").strip()
        normalized_text = TextNormalizer.normalize(raw_text)
        intent_ready_text = IntentPreprocessor.prepare(normalized_text)
        route_ready_text = cls._strip_leading_address(intent_ready_text).strip()
        lookup_ready_text = cls._strip_leading_address(normalized_text).rstrip("?.!").strip()
        tool_ready_text = cls._strip_leading_address(intent_ready_text).strip()
        tool_source_text = cls._strip_leading_address(raw_text).strip()
        return PromptSurfaceViews(
            raw_text=raw_text,
            normalized_text=normalized_text,
            intent_ready_text=intent_ready_text,
            reconstructed_text=intent_ready_text,
            route_ready_text=route_ready_text,
            lookup_ready_text=lookup_ready_text,
            tool_ready_text=tool_ready_text,
            tool_source_text=tool_source_text,
        )

    @classmethod
    def _strip_leading_address(cls, text: str) -> str:
        stripped = cls._LEADING_ADDRESS_RE.sub("", str(text or "").strip(), count=1)
        return stripped.strip()
