from __future__ import annotations

import re

from lumen.memory.memory_models import MemoryClassification


class MemoryClassifier:
    """Classifies interactions into conservative memory-save categories."""

    RESEARCH_ACTION_HINTS = {
        "build",
        "compare",
        "create",
        "debug",
        "design",
        "document",
        "implement",
        "inspect",
        "migrate",
        "plan",
        "refactor",
        "review",
        "route",
        "summarize",
        "test",
        "verify",
    }

    RESEARCH_HINTS = {
        "api",
        "app",
        "artifact",
        "bundle",
        "capability",
        "cli",
        "config",
        "contract",
        "controller",
        "code",
        "dataset",
        "deploy",
        "deployment",
        "diagnostic",
        "docs",
        "research",
        "plan",
        "roadmap",
        "architecture",
        "migration",
        "design",
        "experiment",
        "hypothesis",
        "finding",
        "decision",
        "milestone",
        "analysis",
        "compare",
        "implement",
        "debug",
        "workspace",
        "archive",
        "report",
        "routing",
        "pipeline",
        "module",
        "schema",
        "session",
        "service",
        "system",
        "tool",
        "validation",
        "workflow",
    }

    PERSONAL_HINTS = {
        "about me",
        "for me personally",
        "my anxiety",
        "my health",
        "my family",
        "my relationship",
        "my finances",
        "my trauma",
        "my diagnosis",
        "i feel",
        "i am anxious",
        "i'm anxious",
        "i am stressed",
        "i'm stressed",
        "remember this about me",
        "save this about me",
        "my personal",
        "my private",
        "my life",
    }

    AMBIGUOUS_CONVERSATIONAL_HINTS = {
        "can you help",
        "could you help",
        "help me think",
        "what do you think",
        "maybe",
        "not sure",
        "i wonder",
        "thoughts",
    }

    THREAD_CONTINUATION_HINTS = {
        "keep going",
        "go on",
        "continue",
        "continue on",
        "what else",
        "and then",
        "go deeper",
        "tell me more",
        "more",
    }

    EXPLICIT_SAVE_HINTS = {
        "remember this",
        "save this",
        "remember that",
        "save that",
        "please remember",
        "please save",
    }

    PERSONAL_PREFERENCE_HINTS = {
        "remember that i prefer",
        "save that i prefer",
        "remember my preference",
        "save my preference",
        "remember i like",
        "save that i like",
    }

    DURABLE_PREFERENCE_HINTS = {
        "from now on",
        "going forward",
        "for future reference",
        "in the future",
        "moving forward",
    }

    CONVERSATIONAL_PREFERENCE_CUES = {
        "i prefer",
        "be more direct",
        "be more concise",
        "keep it brief",
        "keep it short",
        "direct",
        "less verbose",
        "more conversational",
        "call me ",
    }

    def classify(
        self,
        *,
        prompt: str,
        resolved_prompt: str | None,
        mode: str,
        dominant_intent: str | None,
        summary: str,
    ) -> MemoryClassification:
        prompt_text = " ".join(
            part.strip().lower()
            for part in (prompt, resolved_prompt or "")
            if str(part).strip()
        )
        original_prompt = re.sub(r"\s+", " ", str(prompt or "").strip().lower()).strip()
        normalized_resolved_prompt = re.sub(r"\s+", " ", str(resolved_prompt or "").strip().lower()).strip()
        summary_text = re.sub(r"\s+", " ", str(summary or "").strip().lower()).strip()
        normalized_prompt = re.sub(r"\s+", " ", prompt_text).strip()
        normalized = " ".join(part for part in (normalized_prompt, summary_text) if part).strip()
        prompt_tokens = set(re.findall(r"[a-z0-9_]+", normalized_prompt))
        summary_tokens = set(re.findall(r"[a-z0-9_]+", summary_text))

        if self._looks_durable_preference(normalized_prompt):
            return MemoryClassification.personal_candidate(
                confidence=0.93,
                reason="The interaction states a durable user preference that should only be stored as personal context.",
                explicit_save_requested=True,
            )

        if self._looks_personal(normalized) or self._looks_personal_preference(normalized_prompt):
            explicit_save = any(hint in normalized for hint in self.EXPLICIT_SAVE_HINTS)
            if explicit_save:
                return MemoryClassification.personal_candidate(
                    confidence=0.96,
                    reason="The interaction looks personal or private, so it requires explicit user-directed saving.",
                    explicit_save_requested=True,
                )
            return MemoryClassification.personal_candidate(
                confidence=0.88,
                reason="The interaction looks personal or sensitive, so it is not eligible for automatic saving.",
            )

        if self._looks_thread_continuation(
            original_prompt=original_prompt,
            resolved_prompt=normalized_resolved_prompt,
        ):
            return MemoryClassification.ephemeral(
                confidence=0.24,
                reason="The interaction looks like a lightweight continuation turn, so it remains ephemeral unless it resolves into a more explicit thread action.",
            )

        research_score = self._research_score(
            normalized_prompt=normalized_prompt,
            prompt_tokens=prompt_tokens,
            summary_tokens=summary_tokens,
            mode=mode,
            dominant_intent=dominant_intent,
        )
        if research_score >= 4:
            return MemoryClassification.research_candidate(
                confidence=min(0.95, 0.55 + (research_score * 0.08)),
                reason="The interaction appears clearly project- or research-oriented and is eligible for conservative research-note saving.",
            )

        if dominant_intent == "unknown" or not normalized:
            return MemoryClassification.ephemeral(
                confidence=0.18 if normalized else 0.0,
                reason="The interaction is too uncertain to classify safely, so it remains unsaved by default.",
            )

        if self._looks_ambiguous_or_low_signal(normalized=normalized_prompt, research_score=research_score):
            return MemoryClassification.ephemeral(
                confidence=0.28,
                reason="The interaction is too open-ended or weakly signaled to save safely, so it remains ephemeral by default.",
            )

        return MemoryClassification.ephemeral(
            confidence=0.42,
            reason="The interaction looks conversational but not clearly research-related, so it remains ephemeral.",
        )

    def _looks_personal(self, normalized: str) -> bool:
        return any(hint in normalized for hint in self.PERSONAL_HINTS)

    def _looks_personal_preference(self, normalized_prompt: str) -> bool:
        return any(hint in normalized_prompt for hint in self.PERSONAL_PREFERENCE_HINTS)

    def _looks_durable_preference(self, normalized_prompt: str) -> bool:
        has_duration = any(hint in normalized_prompt for hint in self.DURABLE_PREFERENCE_HINTS)
        has_preference = any(cue in normalized_prompt for cue in self.CONVERSATIONAL_PREFERENCE_CUES)
        return has_duration and has_preference

    def _looks_thread_continuation(
        self,
        *,
        original_prompt: str,
        resolved_prompt: str,
    ) -> bool:
        if not original_prompt:
            return False
        if original_prompt not in self.THREAD_CONTINUATION_HINTS:
            return False
        if not resolved_prompt:
            return True
        if resolved_prompt == original_prompt:
            return True
        expanded_actions = ("compare ", "expand ", "continue with ", "explain more about ")
        return not any(resolved_prompt.startswith(prefix) for prefix in expanded_actions)

    def _research_score(
        self,
        *,
        normalized_prompt: str,
        prompt_tokens: set[str],
        summary_tokens: set[str],
        mode: str,
        dominant_intent: str | None,
    ) -> int:
        score = 0
        prompt_research_hint_matches = len(prompt_tokens & self.RESEARCH_HINTS)
        prompt_action_hint_matches = len(prompt_tokens & self.RESEARCH_ACTION_HINTS)
        summary_research_hint_matches = len(summary_tokens & self.RESEARCH_HINTS)
        summary_action_hint_matches = len(summary_tokens & self.RESEARCH_ACTION_HINTS)
        if mode in {"planning", "research"}:
            score += 1
        if dominant_intent in {"planning", "research"}:
            score += 1
        score += min(prompt_research_hint_matches, 3)
        score += min(prompt_action_hint_matches, 2)
        score += min(summary_research_hint_matches, 1)
        score += min(summary_action_hint_matches, 1)
        if "workspace" in prompt_tokens or "archive" in prompt_tokens or "routing" in prompt_tokens or "pipeline" in prompt_tokens:
            score += 1
        if len(prompt_tokens) >= 6 and prompt_research_hint_matches >= 1 and prompt_action_hint_matches >= 1:
            score += 1
        if self._looks_ambiguous_or_low_signal(normalized=normalized_prompt, research_score=score):
            score -= 2
        return max(score, 0)

    def _looks_ambiguous_or_low_signal(self, *, normalized: str, research_score: int) -> bool:
        if any(hint in normalized for hint in self.AMBIGUOUS_CONVERSATIONAL_HINTS):
            return True
        return research_score < 3 and len(normalized.split()) <= 6
