from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.routing.anchor_registry import (
    COMPARE_SHORTHAND_PREFIXES,
    THREAD_FOLLOW_UP_EXPANSIONS,
    TOOL_CONTEXT_SHORTHAND,
    TOOL_REPEAT_SHORTHAND,
    detect_follow_up_anchor,
)
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder
from lumen.routing.capability_manager import CapabilityManager


@dataclass(slots=True)
class PromptResolution:
    original_prompt: str
    resolved_prompt: str
    strategy: str = "none"
    reason: str = "No prompt rewrite applied"

    @property
    def changed(self) -> bool:
        return self.original_prompt != self.resolved_prompt


class PromptResolver:
    """Applies lightweight session-aware prompt rewrites for shorthand follow-ups."""

    SUBJECT_PREFIXES = (
        "Plan work for: ",
        "Research topic: ",
        "Execute tool task: ",
    )

    LEGACY_RESPONSE_PREFIXES = (
        "Planning response for: ",
        "Research response for: ",
    )

    def __init__(self, capability_manager: CapabilityManager | None = None) -> None:
        self.capability_manager = capability_manager
        self.prompt_nlu = PromptNLU()

    def resolve(
        self,
        prompt: str,
        *,
        active_thread: dict[str, Any] | None,
    ) -> PromptResolution:
        inferred_tool = self._infer_tool_alias(prompt)
        if inferred_tool is not None:
            return inferred_tool

        normalized = self.prompt_nlu.analyze(prompt).surface_views.route_ready_text
        rewritten_explanation = self._rewrite_standalone_explanation_prompt(normalized, prompt)
        if rewritten_explanation is not None:
            return rewritten_explanation

        if not active_thread:
            return PromptResolution(original_prompt=prompt, resolved_prompt=prompt)
        if not self._supports_subject_rewrite(active_thread):
            return PromptResolution(original_prompt=prompt, resolved_prompt=prompt)

        subject = self._build_subject(active_thread)
        if not subject:
            return PromptResolution(original_prompt=prompt, resolved_prompt=prompt)
        noun_phrase = self._to_noun_phrase(subject)
        compare_target = self._explicit_compare_target(normalized)
        if compare_target:
            return PromptResolution(
                original_prompt=prompt,
                resolved_prompt=f"compare {noun_phrase} and {compare_target}",
                strategy="compare_shorthand",
                reason="Expanded comparison shorthand using the active thread subject and the explicit comparison target",
            )

        if self._is_compare_shorthand(normalized):
            return PromptResolution(
                original_prompt=prompt,
                resolved_prompt=f"compare {noun_phrase}",
                strategy="compare_shorthand",
                reason="Expanded comparison shorthand using the active thread subject",
            )

        if self._is_anh_tool_shorthand(normalized, active_thread):
            return PromptResolution(
                original_prompt=prompt,
                resolved_prompt=self._resolve_previous_tool_prompt(active_thread) or "run anh",
                strategy="anh_tool_shorthand",
                reason="Mapped ANH shorthand to the active ANH tool command and active GA tool command from thread context",
            )

        if self._is_tool_repeat_shorthand(normalized, active_thread):
            previous_prompt = self._resolve_previous_tool_prompt(active_thread)
            if previous_prompt:
                return PromptResolution(
                    original_prompt=prompt,
                    resolved_prompt=previous_prompt,
                    strategy="tool_repeat_shorthand",
                    reason="Reused the active tool prompt for a repeat-style follow-up",
                )

        thread_follow_up = self._expand_thread_follow_up(normalized, prompt, noun_phrase)
        if thread_follow_up is not None:
            return thread_follow_up

        if self._is_reference_follow_up(normalized):
            return PromptResolution(
                original_prompt=prompt,
                resolved_prompt=self._build_reference_prompt(normalized, noun_phrase),
                strategy="reference_follow_up",
                reason="Expanded reference-style follow-up using the active thread subject",
            )

        return PromptResolution(original_prompt=prompt, resolved_prompt=prompt)

    def _infer_tool_alias(self, prompt: str) -> PromptResolution | None:
        if self.capability_manager is None:
            return None
        inferred = self.capability_manager.infer_command_alias(prompt)
        if inferred is not None:
            _, alias = inferred
            normalized_prompt = self.prompt_nlu.analyze(prompt).surface_views.route_ready_text
            if normalized_prompt == alias:
                return None
            return PromptResolution(
                original_prompt=prompt,
                resolved_prompt=alias,
                strategy="tool_hint_alias",
                reason="Resolved the prompt to the closest manifest-declared tool alias using NLU hints",
            )
        signal_match = self.capability_manager.infer_by_signals(prompt)
        if signal_match is None or signal_match.confidence < 0.82:
            return None
        capability = self.capability_manager.get(signal_match.capability_key)
        aliases = capability.command_aliases or []
        if not aliases:
            return None
        canonical_alias = aliases[0]
        normalized_prompt = self.prompt_nlu.analyze(prompt).surface_views.route_ready_text
        if normalized_prompt == canonical_alias:
            return None
        return PromptResolution(
            original_prompt=prompt,
            resolved_prompt=canonical_alias,
            strategy="tool_signal_alias",
            reason="Resolved the prompt to a manifest-declared tool alias using hybrid tool-routing signals",
        )

    @staticmethod
    def _supports_subject_rewrite(active_thread: dict[str, Any]) -> bool:
        mode = str(active_thread.get("mode") or "").strip().lower()
        kind = str(active_thread.get("kind") or "").strip().lower()
        if mode in {"planning", "research", "tool"}:
            return True
        if kind.startswith(("planning.", "research.", "tool.")):
            return True
        legacy_fields = (
            str(active_thread.get("objective") or "").strip(),
            str(active_thread.get("thread_summary") or "").strip(),
            str(active_thread.get("summary") or "").strip(),
        )
        known_prefixes = (
            *PromptResolver.SUBJECT_PREFIXES,
            *PromptResolver.LEGACY_RESPONSE_PREFIXES,
        )
        if any(field.startswith(prefix) for field in legacy_fields for prefix in known_prefixes):
            return True
        thread_summary = str(active_thread.get("thread_summary") or "").strip()
        if thread_summary and PromptResolver._looks_like_task_subject(thread_summary):
            return True
        active_prompt = str(active_thread.get("prompt") or "").strip().lower()
        return active_prompt == "analyze ga" or bool(active_thread.get("tool_context"))

    @staticmethod
    def _looks_like_task_subject(subject: str) -> bool:
        normalized = " ".join(str(subject or "").strip().lower().split())
        if not normalized:
            return False
        conversational_starts = (
            "here's ",
            "heres ",
            "here is ",
            "i'm ",
            "im ",
            "you're ",
            "youre ",
            "let's ",
            "lets ",
            "sounds good",
            "alright",
            "okay",
            "good to see you",
            "how are you",
        )
        if normalized.startswith(conversational_starts):
            return False
        task_markers = (
            "create ",
            "build ",
            "design ",
            "summarize ",
            "summary of ",
            "review ",
            "analyze ",
            "compare ",
            "explain ",
            "plan ",
        )
        return any(normalized.startswith(marker) for marker in task_markers) or any(
            token in normalized
            for token in ("migration", "roadmap", "archive", "routing", "architecture")
        )

    @staticmethod
    def _is_compare_shorthand(normalized_prompt: str) -> bool:
        anchor = detect_follow_up_anchor(normalized_prompt)
        return bool(anchor is not None and anchor.kind == "reference" and anchor.action == "compare") or any(
            normalized_prompt.startswith(prefix) for prefix in COMPARE_SHORTHAND_PREFIXES
        )

    @staticmethod
    def _is_anh_tool_shorthand(
        normalized_prompt: str,
        active_thread: dict[str, Any],
    ) -> bool:
        if normalized_prompt not in TOOL_CONTEXT_SHORTHAND:
            return False
        tool_context = active_thread.get("tool_context") or {}
        if str(tool_context.get("tool_id") or "").strip().lower() == "anh":
            return True
        active_text = " ".join(
            str(active_thread.get(field, "")).lower()
            for field in ("prompt", "objective", "thread_summary", "summary")
        )
        return any(token in active_text for token in ("anh", "si iv", "spectral", "absorption", "dip"))

    @staticmethod
    def _is_tool_repeat_shorthand(
        normalized_prompt: str,
        active_thread: dict[str, Any],
    ) -> bool:
        has_tool_context = bool(active_thread.get("tool_context"))
        if str(active_thread.get("mode") or "").strip() != "tool" and not has_tool_context:
            return False
        return normalized_prompt in TOOL_REPEAT_SHORTHAND

    def _expand_thread_follow_up(
        self,
        normalized_prompt: str,
        original_prompt: str,
        noun_phrase: str,
    ) -> PromptResolution | None:
        anchor = detect_follow_up_anchor(normalized_prompt)
        if normalized_prompt in THREAD_FOLLOW_UP_EXPANSIONS and normalized_prompt.startswith("expand"):
            return PromptResolution(
                original_prompt=original_prompt,
                resolved_prompt=f"expand {noun_phrase}",
                strategy="thread_follow_up",
                reason="Expanded an 'expand' follow-up using the active thread subject",
            )
        if normalized_prompt in THREAD_FOLLOW_UP_EXPANSIONS and normalized_prompt.startswith("continue"):
            return PromptResolution(
                original_prompt=original_prompt,
                resolved_prompt=f"continue with {noun_phrase}",
                strategy="thread_follow_up",
                reason="Expanded a 'continue' follow-up using the active thread subject",
            )
        if anchor is not None and anchor.explanation_mode == "deeper":
            return PromptResolution(
                original_prompt=original_prompt,
                resolved_prompt=f"explain more about {noun_phrase}",
                strategy="thread_follow_up",
                reason="Expanded a deeper-explanation follow-up using the active thread subject",
            )
        if anchor is not None and anchor.explanation_mode == "break_down":
            return PromptResolution(
                original_prompt=original_prompt,
                resolved_prompt=f"break {noun_phrase} down simply",
                strategy="thread_follow_up",
                reason="Expanded a simplification follow-up using the active thread subject",
            )
        return None

    @staticmethod
    def _is_reference_follow_up(normalized_prompt: str) -> bool:
        anchor = detect_follow_up_anchor(normalized_prompt)
        return bool(anchor is not None and anchor.kind == "reference")

    @staticmethod
    def _resolve_previous_tool_prompt(active_thread: dict[str, Any]) -> str:
        previous_prompt = str(active_thread.get("prompt") or "").strip()
        normalized_previous = " ".join(previous_prompt.lower().split())
        if previous_prompt and normalized_previous not in TOOL_REPEAT_SHORTHAND:
            return previous_prompt

        tool_context = active_thread.get("tool_context") or {}
        tool_id = str(tool_context.get("tool_id") or "").strip().lower()
        capability = str(tool_context.get("capability") or "").strip().lower()
        if tool_id == "anh" and capability == "spectral_dip_scan":
            return "run anh"
        return previous_prompt

    @staticmethod
    def _build_subject(active_thread: dict[str, Any]) -> str:
        reasoning_state = active_thread.get("reasoning_state") or {}
        if isinstance(reasoning_state, dict):
            for key in ("canonical_subject", "continuation_target", "resolved_prompt", "current_task"):
                raw = str(reasoning_state.get(key) or "").strip()
                cleaned = PromptResolver._clean_subject(raw)
                if cleaned:
                    return cleaned
        candidates = (
            ("prompt", active_thread.get("prompt")),
            ("objective", active_thread.get("objective")),
            ("thread_summary", active_thread.get("thread_summary")),
            ("summary", active_thread.get("summary")),
        )
        for source, candidate in candidates:
            raw = str(candidate or "").strip()
            cleaned = PromptResolver._clean_subject(raw)
            if cleaned:
                if source in {"thread_summary", "summary"} and not PromptResolver._usable_summary_subject(
                    raw_subject=raw,
                    cleaned_subject=cleaned,
                ):
                    continue
                return cleaned
        return ""

    @staticmethod
    def _clean_subject(subject: str) -> str:
        prefixes = (
            *PromptResolver.SUBJECT_PREFIXES,
            *PromptResolver.LEGACY_RESPONSE_PREFIXES,
        )
        cleaned = subject
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break
        if " | latest: " in cleaned:
            cleaned = cleaned.split(" | latest: ", maxsplit=1)[0].strip()
        return cleaned

    @staticmethod
    def _usable_summary_subject(*, raw_subject: str, cleaned_subject: str) -> bool:
        normalized_raw = " ".join(str(raw_subject or "").strip().lower().split())
        if not normalized_raw or not cleaned_subject:
            return False
        if any(normalized_raw.startswith(prefix.lower()) for prefix in PromptResolver.LEGACY_RESPONSE_PREFIXES):
            return True
        if any(normalized_raw.startswith(prefix.lower()) for prefix in PromptResolver.SUBJECT_PREFIXES):
            return True
        conversational_summary_starts = (
            "here's ",
            "heres ",
            "here is ",
            "i'm ",
            "im ",
            "you're ",
            "youre ",
            "let's ",
            "lets ",
            "sounds good",
            "alright",
            "okay",
        )
        if normalized_raw.startswith(conversational_summary_starts):
            return False
        return True

    @staticmethod
    def _to_noun_phrase(subject: str) -> str:
        lowered = subject.lower()
        if lowered.startswith("compare "):
            compared = subject[len("compare ") :].strip()
            if compared.startswith(("the ", "a ", "an ")):
                return compared
            return "the " + compared if compared else subject
        if lowered.startswith("contrast "):
            contrasted = subject[len("contrast ") :].strip()
            if contrasted.startswith(("the ", "a ", "an ")):
                return contrasted
            return "the " + contrasted if contrasted else subject
        if lowered.startswith("create a migration plan for "):
            return "the migration plan for " + subject[len("create a migration plan for ") :].strip()
        if lowered.startswith("create the migration plan for "):
            return "the migration plan for " + subject[len("create the migration plan for ") :].strip()
        if lowered.startswith("create a roadmap for "):
            return "the roadmap for " + subject[len("create a roadmap for ") :].strip()
        if lowered.startswith("build a roadmap for "):
            return "the roadmap for " + subject[len("build a roadmap for ") :].strip()
        if lowered.startswith("summarize "):
            tail = subject[len("summarize ") :].strip()
            if tail.startswith(("the ", "a ", "an ")):
                return tail
            return "the " + tail
        if lowered.startswith("summary of "):
            return "the " + subject
        if lowered.startswith("review "):
            tail = subject[len("review ") :].strip()
            if tail.startswith(("the ", "a ", "an ")):
                return tail
            return "the " + tail
        if lowered.startswith("analyze "):
            return subject
        return subject

    @staticmethod
    def _build_reference_prompt(normalized_prompt: str, noun_phrase: str) -> str:
        if normalized_prompt.startswith("what about"):
            return f"what about {noun_phrase}"
        if normalized_prompt.startswith("how about"):
            return f"how about {noun_phrase}"
        return f"{normalized_prompt} regarding {noun_phrase}"

    @staticmethod
    def _explicit_compare_target(normalized_prompt: str) -> str:
        patterns = (
            "compare that to ",
            "compare this to ",
            "compare it to ",
            "compare that with ",
            "compare this with ",
            "compare it with ",
        )
        for prefix in patterns:
            if normalized_prompt.startswith(prefix):
                return normalized_prompt[len(prefix) :].strip()
        return ""

    @staticmethod
    def _rewrite_standalone_explanation_prompt(
        normalized_prompt: str,
        original_prompt: str,
    ) -> PromptResolution | None:
        if normalized_prompt.startswith("break down ") and normalized_prompt.endswith(" step by step"):
            subject = normalized_prompt[len("break down ") : -len(" step by step")].strip()
            if subject:
                return PromptResolution(
                    original_prompt=original_prompt,
                    resolved_prompt=f"explain {subject} step by step",
                    strategy="standalone_explanation_rewrite",
                    reason="Rewrote a standalone break-down prompt into an explanation request.",
                )
        if normalized_prompt.startswith("break down ") and normalized_prompt.endswith(" simply"):
            subject = normalized_prompt[len("break down ") : -len(" simply")].strip()
            if subject:
                return PromptResolution(
                    original_prompt=original_prompt,
                    resolved_prompt=f"explain {subject} simply",
                    strategy="standalone_explanation_rewrite",
                    reason="Rewrote a standalone break-down prompt into a simple explanation request.",
                )
        return None
