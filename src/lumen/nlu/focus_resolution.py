from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder


@dataclass(frozen=True, slots=True)
class FocusResolution:
    normalized_prompt: str
    focus: str
    wrappers_removed: tuple[str, ...] = ()

    def diagnostics(self) -> dict[str, object]:
        return {
            "normalized_prompt": self.normalized_prompt,
            "focus": self.focus,
            "wrappers_removed": list(self.wrappers_removed),
            "reason": self.reason(),
        }

    def reason(self) -> str:
        if not self.focus:
            return "no_subject_resolution"
        if self.focus == self.normalized_prompt.strip():
            return "direct_focus"
        return "wrapper_stripped"


class FocusResolutionSupport:
    _SUBJECT_PREFIXES: tuple[str, ...] = (
        "what is the meaning of ",
        "tell me about ",
        "can you explain ",
        "could you explain ",
        "help me understand ",
        "what is a ",
        "what is an ",
        "what is ",
        "what are ",
        "what were ",
        "what's a ",
        "what's an ",
        "what's ",
        "whats a ",
        "whats an ",
        "whats ",
        "what was ",
        "who is ",
        "who was ",
        "explain ",
        "describe ",
        "define ",
    )
    _SUBJECT_CLEANUP_PREFIXES: tuple[str, ...] = (
        "the concept of ",
        "the idea of ",
        "the topic of ",
    )
    _TRAILING_NOISE: tuple[str, ...] = (" lol", " haha", " lmao", " hehe")
    _RECALL_CUES: tuple[str, ...] = (
        "what do you remember about",
        "what do you remember",
        "what do we remember about",
        "what do we remember",
        "what do you have on",
        "remember about",
    )
    _PROJECT_RETURN_CUES: tuple[str, ...] = (
        "back to ",
        "where were we on ",
        "continue the ",
        "continue with ",
        "what was our last take on ",
        "pick back up on ",
        "pick back up ",
        "return to ",
        "resume the ",
    )

    @classmethod
    def subject_focus(cls, prompt: str) -> FocusResolution:
        normalized_prompt = PromptSurfaceBuilder.build(prompt).lookup_ready_text.strip()
        subject = normalized_prompt.rstrip("?.!")
        removed: list[str] = []
        for prefix in ("hey ", "hi ", "hello ", "yo "):
            if subject.startswith(prefix):
                subject = subject[len(prefix) :].strip()
                removed.append(prefix.strip())
                break
        for prefix in cls._SUBJECT_PREFIXES:
            if subject.startswith(prefix):
                subject = subject[len(prefix) :].strip()
                removed.append(prefix.strip())
                break
        suffixes = (" mean", " means")
        for template in ("what do ", "what does "):
            if subject.startswith(template):
                candidate = subject[len(template) :].strip()
                removed.append(template.strip())
                for suffix in suffixes:
                    if candidate.endswith(suffix):
                        subject = candidate[: -len(suffix)].strip()
                        removed.append(suffix.strip())
                        break
                else:
                    subject = candidate
                break
        for prefix in cls._SUBJECT_CLEANUP_PREFIXES:
            if subject.startswith(prefix):
                subject = subject[len(prefix) :].strip()
                removed.append(prefix.strip())
                break
        if subject.startswith("a "):
            subject = subject[2:].strip()
            removed.append("a")
        elif subject.startswith("an "):
            subject = subject[3:].strip()
            removed.append("an")
        for suffix in cls._TRAILING_NOISE:
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)].strip()
                removed.append(suffix.strip())
        return FocusResolution(
            normalized_prompt=normalized_prompt,
            focus=subject,
            wrappers_removed=tuple(removed),
        )

    @classmethod
    def recall_focus(cls, prompt: str) -> FocusResolution:
        normalized_prompt = PromptSurfaceBuilder.build(prompt).lookup_ready_text.strip()
        focus = normalized_prompt
        removed: list[str] = []
        for cue in cls._RECALL_CUES:
            if focus.startswith(cue):
                focus = focus[len(cue) :].strip()
                removed.append(cue)
                break
        for prefix in ("my ", "the ", "our "):
            if focus.startswith(prefix):
                focus = focus[len(prefix) :].strip()
                removed.append(prefix.strip())
        focus = focus.strip(" ?!.")
        if focus.endswith("preferences"):
            focus = f"{focus[:-11]}preference".strip()
            removed.append("preferences")
        return FocusResolution(
            normalized_prompt=normalized_prompt,
            focus=focus,
            wrappers_removed=tuple(removed),
        )

    @classmethod
    def project_return_focus(cls, prompt: str) -> FocusResolution:
        normalized_prompt = PromptSurfaceBuilder.build(prompt).lookup_ready_text.strip()
        focus = normalized_prompt
        removed: list[str] = []
        for cue in cls._PROJECT_RETURN_CUES:
            if focus.startswith(cue):
                focus = focus[len(cue) :].strip()
                removed.append(cue.strip())
                break
        for suffix in (" project", " thread", " thing"):
            if focus.endswith(suffix):
                focus = focus[: -len(suffix)].strip()
                removed.append(suffix.strip())
        focus = focus.strip(" ?!.") or normalized_prompt
        return FocusResolution(
            normalized_prompt=normalized_prompt,
            focus=focus,
            wrappers_removed=tuple(removed),
        )
