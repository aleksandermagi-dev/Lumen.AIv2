from __future__ import annotations

from lumen.nlu.models import DetectedLanguage


class LanguageDetector:
    """Very small heuristic detector for prompt-level language hints."""

    LANGUAGE_HINTS = {
        "en": {"the", "what", "how", "why", "create", "plan", "summary", "compare"},
        "es": {"el", "la", "que", "como", "por", "crear", "plan", "resumen", "comparar"},
        "fr": {"le", "la", "que", "comment", "pourquoi", "creer", "plan", "resume", "comparer"},
        "de": {"der", "die", "das", "wie", "warum", "plan", "zusammenfassung", "vergleichen"},
    }

    def detect(self, text: str) -> DetectedLanguage:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return DetectedLanguage(code="en", confidence=0.5)

        tokens = set(normalized.split())
        best_code = "en"
        best_score = 0
        for code, hints in self.LANGUAGE_HINTS.items():
            score = len(tokens & hints)
            if score > best_score:
                best_code = code
                best_score = score

        if best_score <= 0:
            return DetectedLanguage(code="en", confidence=0.55)
        confidence = min(0.95, 0.55 + (best_score * 0.12))
        return DetectedLanguage(code=best_code, confidence=round(confidence, 2))
