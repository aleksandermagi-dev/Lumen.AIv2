from __future__ import annotations

from lumen.nlu.text_normalizer import TextNormalizer
from lumen.reasoning.pipeline_models import VibeCatcherResult


class VibeCatcher:
    """Translates messy human input into directional signals before heavier reasoning commits."""

    DISAGREEMENT_CUES = (
        "no way",
        "ain't",
        "is not",
        "nah",
        "nope",
        "not like that",
        "not really",
        "wrong",
        "that is not right",
    )
    FRUSTRATION_CUES = (
        "wtf",
        "what the fuck",
        "stuck",
        "broken",
        "frustrated",
        "annoying",
        "messed up",
        "jacked up",
        "scuffed",
    )
    HESITATION_CUES = ("maybe", "i guess", "not sure", "do not know", "kind of", "sort of")
    EMPHASIS_CUES = ("really", "seriously", "literally", "definitely")
    URGENCY_CUES = ("right now", "as soon as possible", "need this", "need it")

    def catch(self, prompt: str) -> VibeCatcherResult:
        normalized_prompt = TextNormalizer.normalize(prompt)
        lowered = " ".join(str(prompt).strip().lower().split())
        directional_signals: list[str] = []
        if any(cue in normalized_prompt for cue in self.DISAGREEMENT_CUES):
            directional_signals.append("disagreement")
        if any(cue in normalized_prompt for cue in self.FRUSTRATION_CUES):
            directional_signals.append("frustration")
        if any(cue in normalized_prompt for cue in self.HESITATION_CUES):
            directional_signals.append("hesitation")
        if any(cue in normalized_prompt for cue in self.EMPHASIS_CUES):
            directional_signals.append("emphasis")
        if any(cue in normalized_prompt for cue in self.URGENCY_CUES):
            directional_signals.append("urgency")
        if normalized_prompt != lowered:
            directional_signals.append("normalization_applied")

        raw_token_count = len(prompt.split())
        normalized_token_count = len(normalized_prompt.split())
        collapsed_token = raw_token_count <= 1 and len(prompt.strip()) >= 18
        heavy_normalization = normalized_prompt != lowered and normalized_token_count >= raw_token_count + 2
        directional_only = bool(directional_signals) and normalized_token_count <= 6
        hesitant_but_loose = (
            "hesitation" in directional_signals
            and heavy_normalization
            and normalized_token_count <= 12
        )
        low_confidence = collapsed_token or (directional_only and heavy_normalization) or hesitant_but_loose
        interpretation_confidence = 0.38 if low_confidence else (0.64 if heavy_normalization else 0.9)
        recovery_hint = None
        if low_confidence:
            if {"disagreement", "frustration"} & set(directional_signals):
                recovery_hint = (
                    "I can tell you are reacting against something, but the target is still too compressed to route cleanly."
                )
            elif "hesitation" in directional_signals:
                recovery_hint = (
                    "I can tell there is a real concern here, but the prompt is still too loose to commit to one interpretation."
                )
            else:
                recovery_hint = (
                    "The input carries direction, but it is too compressed to route confidently without clarification."
                )
        return VibeCatcherResult(
            normalized_prompt=normalized_prompt,
            directional_signals=directional_signals,
            interpretation_confidence=interpretation_confidence,
            low_confidence=low_confidence,
            recovery_hint=recovery_hint,
        )
