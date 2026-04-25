from __future__ import annotations

from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.pipeline_models import (
    ConversationAwarenessResult,
    EmpathyModelResult,
    HumanLanguageLayerResult,
    StateControlResult,
)


class HumanLanguageLayer:
    """Shapes lightweight human-facing language posture on top of core reasoning."""

    EXPLORATORY_CUES = ("what if", "could", "i wonder", "might", "how would we test", "what would explain")
    ASSERTIVE_CUES = ("this is how it works", "this is what it is", "this is the issue", "it definitely", "it clearly")
    UNSURE_CUES = ("not sure", "unsure", "i think", "probably", "maybe", "sort of")
    CORRECTION_CUES = (
        "no not that",
        "no that's not what i meant",
        "no that is not what i meant",
        "no thats not what i meant",
        "that's not what i meant",
        "thats not what i meant",
        "wait i meant",
        "wait no",
        "redo it but",
        "redo that but",
        "not like that",
        "i meant this",
        "i meant the other",
    )
    FRUSTRATED_CUES = (
        "broken",
        "annoying",
        "wtf",
        "this shit",
        "damn",
        "frustrating",
        "didn't work",
        "didnt work",
        "not working",
        "broke",
    )

    def assess(
        self,
        *,
        prompt: str,
        conversation_awareness: ConversationAwarenessResult,
        empathy_model: EmpathyModelResult | None,
        state_control: StateControlResult | None,
        interaction_profile,
        active_thread: dict[str, object] | None,
    ) -> HumanLanguageLayerResult:
        normalized = " ".join(str(prompt).strip().lower().split())
        user_energy = self._user_energy(prompt=prompt, normalized=normalized, empathy_model=empathy_model)
        correction_detected = any(cue in normalized for cue in self.CORRECTION_CUES)
        epistemic_stance, stance_confidence = self._epistemic_stance(normalized=normalized)
        emotional_alignment = self._emotional_alignment(
            user_energy=user_energy,
            empathy_model=empathy_model,
        )
        mode_profile = InteractionStylePolicy.mode_profile(interaction_profile)
        response_brevity = self._response_brevity(
            interaction_profile=interaction_profile,
            user_energy=user_energy,
        )
        humor_allowed = bool(state_control.humor_allowed) if state_control is not None else False
        context_continuity = active_thread is not None or conversation_awareness.unresolved_thread_open
        return HumanLanguageLayerResult(
            flow_style=str(mode_profile.get("structure") or "balanced"),
            sentence_variation=True,
            allow_imperfection=mode_profile.get("looseness") != "low",
            context_continuity=context_continuity,
            emotional_alignment=emotional_alignment,
            user_energy=user_energy,
            correction_detected=correction_detected,
            epistemic_stance=epistemic_stance,
            stance_confidence=stance_confidence,
            humor_allowed=humor_allowed,
            curiosity_signal=epistemic_stance == "exploratory",
            reaction_tokens_enabled=user_energy in {"casual", "excited", "frustrated"},
            response_brevity=response_brevity,
        )

    def _user_energy(
        self,
        *,
        prompt: str,
        normalized: str,
        empathy_model: EmpathyModelResult | None,
    ) -> str:
        if any(cue in normalized for cue in self.FRUSTRATED_CUES):
            return "frustrated"
        if prompt.isupper() and len(prompt.strip()) >= 6:
            return "excited"
        if prompt.count("!") >= 2:
            return "excited"
        if empathy_model is not None and empathy_model.response_sensitivity in {"stabilizing", "gentle"}:
            return "frustrated"
        if len(normalized.split()) <= 4:
            return "focused"
        return "casual"

    def _epistemic_stance(self, *, normalized: str) -> tuple[str, str]:
        assertive_hits = sum(1 for cue in self.ASSERTIVE_CUES if cue in normalized)
        unsure_hits = sum(1 for cue in self.UNSURE_CUES if cue in normalized)
        exploratory_hits = sum(1 for cue in self.EXPLORATORY_CUES if cue in normalized)

        if assertive_hits > 0:
            return "assertive", "high"
        if unsure_hits > exploratory_hits and unsure_hits > 0:
            return "unsure", "low"
        if exploratory_hits > 0:
            return "exploratory", "low"
        return "exploratory", "medium"

    @staticmethod
    def _emotional_alignment(
        *,
        user_energy: str,
        empathy_model: EmpathyModelResult | None,
    ) -> str:
        if user_energy == "frustrated":
            return "calm_supportive"
        if user_energy == "excited":
            return "engaged"
        if empathy_model is not None and empathy_model.response_sensitivity == "careful":
            return "reflective"
        return "steady"

    @staticmethod
    def _response_brevity(*, interaction_profile, user_energy: str) -> str:
        mode_profile = InteractionStylePolicy.mode_profile(interaction_profile)
        if mode_profile.get("structure") == "tight" or user_energy == "focused":
            return "light"
        if mode_profile.get("looseness") == "high":
            return "expanded"
        return "balanced"
