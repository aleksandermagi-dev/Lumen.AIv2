from __future__ import annotations

from dataclasses import dataclass

from lumen.nlu.prompt_surface_builder import PromptSurfaceBuilder


DOMAIN_ANCHORS: dict[str, tuple[str, ...]] = {
    "astronomy": (
        "galaxy",
        "star",
        "planet",
        "comet",
        "black hole",
        "orbit",
        "nebula",
        "universe",
        "solar system",
        "milky way",
    ),
    "physics": (
        "gravity",
        "force",
        "energy",
        "mass",
        "acceleration",
        "watt",
        "watts",
        "ohm",
        "ohms",
        "voltage",
        "current",
        "momentum",
        "resistance",
    ),
    "math": (
        "solve",
        "equation",
        "algebra",
        "derivative",
        "integral",
        "sum",
        "multiply",
        "divide",
        "quadratic",
        "x =",
    ),
    "engineering": (
        "build",
        "design",
        "prototype",
        "system",
        "blueprint",
        "mechanism",
        "structure",
        "material",
        "ship",
        "engine",
    ),
    "history": (
        "ancient",
        "empire",
        "dynasty",
        "civilization",
        "pharaoh",
        "war",
        "timeline",
        "rome",
        "egypt",
        "mesopotamia",
        "history",
    ),
    "biology": (
        "cell",
        "dna",
        "organism",
        "species",
        "gene",
        "evolution",
        "biology",
        "ecosystem",
    ),
    "ai_computing": (
        "ai",
        "algorithm",
        "model",
        "operating system",
        "computer",
        "computing",
        "data structure",
        "neural network",
        "machine learning",
        "debug",
    ),
    "philosophy": (
        "philosophy",
        "ethics",
        "epistemology",
        "metaphysics",
        "meaning",
        "logic",
        "free will",
    ),
    "social": (
        "hello",
        "hi",
        "thanks",
        "thank you",
        "how are you",
        "good to see you",
    ),
    "planning": (
        "plan",
        "roadmap",
        "operations",
        "business",
        "strategy",
        "milestone",
        "next step",
        "brainstorm",
    ),
}

ACTION_ANCHORS: dict[str, tuple[str, ...]] = {
    "define": ("what is", "what are", "what does", "who is", "who was", "define"),
    "explain": ("tell me about", "explain", "describe", "what is", "what are", "what does"),
    "compare": ("compare", "difference between", "vs", "versus", "contrast"),
    "solve": ("solve", "equation", "find x"),
    "calculate": ("calculate", "compute"),
    "analyze": ("analyze", "review", "inspect"),
    "break_down": (
        "break it down",
        "break that down",
        "simplify that",
        "simplify it",
        "explain simply",
        "make that easier",
        "make it easier",
        "plain english",
        "in plain english",
    ),
    "go_deeper": (
        "go deeper",
        "explain more",
        "tell me more",
        "expand",
        "more detail",
    ),
    "step_by_step": (
        "step by step",
        "walk me through it",
        "walk me through that",
        "explain step by step",
    ),
    "plan": (
        "create a migration plan",
        "create migration plan",
        "create a roadmap",
        "build a roadmap",
        "how do i build",
        "how do we build",
        "what should we do",
        "design",
        "plan",
        "blueprint",
        "roadmap",
    ),
    "brainstorm": ("brainstorm", "ideas for", "options for"),
    "debug": ("debug", "fix", "troubleshoot"),
    "summarize": ("summarize", "summary of"),
    "continue": ("continue", "go on", "what else", "and then"),
}

CAPABILITY_ANCHORS: dict[str, tuple[str, ...]] = {
    "knowledge": ("define", "explain", "compare", "summarize"),
    "math_solver": ("solve", "calculate"),
    "explanation_transform": ("break_down", "go_deeper", "step_by_step"),
    "planning": ("plan", "brainstorm", "design"),
    "research": ("analyze", "explain", "summarize", "compare"),
    "memory": ("continue",),
    "conversation": ("social",),
    "safety": (),
}

FOLLOW_UP_CONFIRMATIONS: tuple[str, ...] = (
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "go on",
    "continue",
    "more",
    "do it",
    "go ahead",
)
FOLLOW_UP_DECLINES: tuple[str, ...] = ("no", "nope", "stop", "never mind", "not now", "nah", "maybe later")
REFERENCE_FOLLOW_UP_PREFIXES: tuple[str, ...] = (
    "what about",
    "how about",
    "expand that",
    "expand it",
    "expand this",
    "continue that",
    "continue it",
    "continue this",
    "continue with that",
    "continue with it",
    "compare that",
    "compare it",
    "now compare that",
    "now compare it",
    "do that",
    "run that",
    "use anh for that",
    "do that with anh",
    "run that with anh",
    "go deeper on that",
    "go deeper on it",
    "explain that",
    "explain it",
    "simplify that",
    "simplify it",
)
GENERAL_FOLLOW_UP_HINTS: tuple[str, ...] = (
    "continue",
    "expand",
    "go deeper",
    "tell me more",
    "what about",
    "how about",
    "and ",
    "also ",
    "now ",
    "break it down",
    "break that down",
    "step by step",
    "walk me through it",
    "walk me through that",
    "explain more",
)
COMPARE_SHORTHAND_PREFIXES: tuple[str, ...] = (
    "compare that",
    "compare it",
    "now compare that",
    "now compare it",
)
THREAD_FOLLOW_UP_EXPANSIONS: tuple[str, ...] = (
    "expand that",
    "expand that further",
    "expand it",
    "expand it further",
    "continue that",
    "continue it",
    "continue with that",
    "continue with it",
    "explain that",
    "explain it",
    "go deeper on that",
    "go deeper on it",
    "break that down",
    "break it down",
)
TOOL_REPEAT_SHORTHAND: tuple[str, ...] = (
    "do that",
    "run that",
    "do it",
    "run it",
    "do the same analysis",
    "run the same analysis",
    "do that again",
    "run that again",
    "do the same analysis again",
    "run the same analysis again",
    "rerun that",
)
TOOL_CONTEXT_SHORTHAND: tuple[str, ...] = (
    "use anh for that",
    "do that with anh",
    "run that with anh",
)

EXPLANATION_MODE_ANCHORS: dict[str, tuple[str, ...]] = {
    "standard": tuple(),
    "deeper": (
        "go deeper",
        "go deeper on that",
        "go deeper on it",
        "explain more deeply",
        "explain more",
        "tell me more",
    ),
    "break_down": (
        "break it down",
        "break that down",
        "simplify that",
        "simplify it",
        "explain simply",
        "make that easier",
        "make it easier",
        "simpler version",
        "in plain english",
        "explain it in plain english",
    ),
    "step_by_step": (
        "step by step",
        "explain step by step",
        "walk me through it",
        "walk me through that",
        "walk me through this",
    ),
    "analogy": tuple(),
}


@dataclass(slots=True)
class FollowUpAnchor:
    kind: str
    normalized_prompt: str
    action: str | None = None
    explanation_mode: str | None = None
    requires_context: bool = False


@dataclass(slots=True)
class AnchorResolution:
    domains: tuple[str, ...]
    primary_domain: str | None
    actions: tuple[str, ...]
    primary_action: str | None
    capability_hint: str | None
    follow_up_kind: str | None
    explanation_mode: str | None
    topic_anchor: str | None
    requires_context: bool
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "domains": list(self.domains),
            "primary_domain": self.primary_domain,
            "actions": list(self.actions),
            "primary_action": self.primary_action,
            "capability_hint": self.capability_hint,
            "follow_up_kind": self.follow_up_kind,
            "explanation_mode": self.explanation_mode,
            "topic_anchor": self.topic_anchor,
            "requires_context": self.requires_context,
            "confidence": self.confidence,
        }


def normalize_anchor_prompt(prompt: str) -> str:
    return PromptSurfaceBuilder.build(prompt).lookup_ready_text


def detect_domain_anchors(prompt: str) -> tuple[str, ...]:
    normalized = normalize_anchor_prompt(prompt)
    return tuple(
        domain
        for domain, aliases in DOMAIN_ANCHORS.items()
        if any(_contains_anchor(normalized, alias) for alias in aliases)
    )


def detect_action_anchors(prompt: str) -> tuple[str, ...]:
    normalized = normalize_anchor_prompt(prompt)
    return tuple(
        action
        for action, aliases in ACTION_ANCHORS.items()
        if any(_matches_action_anchor(action, normalized, alias) for alias in aliases)
    )


def detect_explanation_mode(prompt: str) -> str | None:
    normalized = normalize_anchor_prompt(prompt)
    for mode, aliases in EXPLANATION_MODE_ANCHORS.items():
        if aliases and any(_contains_anchor(normalized, alias) for alias in aliases):
            return mode
    if normalized.startswith("go deeper on ") or normalized.startswith("go deeper into "):
        return "deeper"
    if normalized.startswith("break ") and normalized.endswith(" down simply"):
        return "break_down"
    if normalized.startswith("walk me through ") and normalized.endswith(" step by step"):
        return "step_by_step"
    return None


def detect_follow_up_anchor(prompt: str) -> FollowUpAnchor | None:
    normalized = normalize_anchor_prompt(prompt)
    if not normalized:
        return None
    if normalized in FOLLOW_UP_CONFIRMATIONS:
        return FollowUpAnchor(kind="confirmation", normalized_prompt=normalized, requires_context=True)
    if normalized in FOLLOW_UP_DECLINES:
        return FollowUpAnchor(kind="decline", normalized_prompt=normalized, requires_context=True)
    explanation_mode = detect_explanation_mode(normalized)
    if explanation_mode is not None:
        return FollowUpAnchor(
            kind="explanation",
            normalized_prompt=normalized,
            action=explanation_mode,
            explanation_mode=explanation_mode,
            requires_context=True,
        )
    if any(normalized.startswith(prefix) for prefix in COMPARE_SHORTHAND_PREFIXES):
        return FollowUpAnchor(kind="reference", normalized_prompt=normalized, action="compare", requires_context=True)
    if any(normalized.startswith(prefix) for prefix in REFERENCE_FOLLOW_UP_PREFIXES):
        return FollowUpAnchor(kind="reference", normalized_prompt=normalized, requires_context=True)
    if any(normalized == value for value in TOOL_REPEAT_SHORTHAND):
        return FollowUpAnchor(kind="tool_repeat", normalized_prompt=normalized, requires_context=True)
    if any(normalized == value for value in TOOL_CONTEXT_SHORTHAND):
        return FollowUpAnchor(kind="tool_context", normalized_prompt=normalized, requires_context=True)
    if looks_like_general_follow_up(normalized):
        actions = detect_action_anchors(normalized)
        return FollowUpAnchor(
            kind="general",
            normalized_prompt=normalized,
            action=actions[0] if actions else None,
            explanation_mode=detect_explanation_mode(normalized),
            requires_context=True,
        )
    return None


def resolve_anchor_context(
    prompt: str,
    recent_interactions: list[dict[str, object]] | None = None,
    active_thread: dict[str, object] | None = None,
    continuation_offer: dict[str, object] | None = None,
) -> AnchorResolution:
    normalized = normalize_anchor_prompt(prompt)
    domains = list(detect_domain_anchors(normalized))
    actions = list(detect_action_anchors(normalized))
    follow_up = detect_follow_up_anchor(normalized)
    topic_anchor = _topic_from_context(
        recent_interactions=recent_interactions or [],
        active_thread=active_thread,
        continuation_offer=continuation_offer,
    )
    if not domains and topic_anchor:
        domains = list(detect_domain_anchors(topic_anchor))
    if follow_up is not None and follow_up.action and follow_up.action not in actions:
        actions.insert(0, follow_up.action)
    primary_domain = domains[0] if domains else None
    primary_action = actions[0] if actions else None
    capability_hint = _capability_hint(primary_domain=primary_domain, primary_action=primary_action)
    requires_context = bool(follow_up is not None and follow_up.requires_context)
    confidence = 0.35
    if primary_domain:
        confidence += 0.2
    if primary_action:
        confidence += 0.2
    if follow_up is not None:
        confidence += 0.15
    if topic_anchor:
        confidence += 0.1
    return AnchorResolution(
        domains=tuple(domains),
        primary_domain=primary_domain,
        actions=tuple(actions),
        primary_action=primary_action,
        capability_hint=capability_hint,
        follow_up_kind=follow_up.kind if follow_up is not None else None,
        explanation_mode=(
            follow_up.explanation_mode
            if follow_up is not None and follow_up.explanation_mode
            else detect_explanation_mode(normalized)
        ),
        topic_anchor=topic_anchor,
        requires_context=requires_context,
        confidence=min(confidence, 0.95),
    )


def looks_like_general_follow_up(normalized_prompt: str) -> bool:
    normalized = normalize_anchor_prompt(normalized_prompt)
    if not normalized:
        return False
    return any(
        normalized.startswith(hint) or f" {hint}" in normalized
        for hint in GENERAL_FOLLOW_UP_HINTS
    )


def looks_like_reference_follow_up(normalized_prompt: str) -> bool:
    anchor = detect_follow_up_anchor(normalized_prompt)
    return bool(anchor is not None and anchor.kind == "reference")


def _topic_from_context(
    *,
    recent_interactions: list[dict[str, object]],
    active_thread: dict[str, object] | None,
    continuation_offer: dict[str, object] | None,
) -> str | None:
    if isinstance(continuation_offer, dict):
        topic = str(continuation_offer.get("topic") or "").strip()
        if topic:
            return topic
    if recent_interactions:
        latest = recent_interactions[0]
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        topic = str(surface.get("topic") or "").strip()
        if topic:
            return topic
        for source in (response, latest):
            topic = str(source.get("normalized_topic") or source.get("resolved_prompt") or source.get("prompt") or "").strip()
            if topic:
                return topic
    if isinstance(active_thread, dict):
        for field in ("normalized_topic", "prompt", "objective", "thread_summary"):
            topic = str(active_thread.get(field) or "").strip()
            if topic:
                return topic
    return None


def _capability_hint(*, primary_domain: str | None, primary_action: str | None) -> str | None:
    if primary_action in {"break_down", "go_deeper", "deeper", "step_by_step"}:
        return "explanation_transform"
    if primary_action in {"solve", "calculate"} or primary_domain == "math":
        return "math_solver"
    if primary_action in {"plan", "brainstorm"} or primary_domain in {"engineering", "planning"}:
        return "planning"
    if primary_domain == "social":
        return "conversation"
    if primary_domain is not None or primary_action in {"define", "explain", "compare", "summarize", "analyze"}:
        return "knowledge"
    return None


def _contains_anchor(normalized_prompt: str, alias: str) -> bool:
    alias_text = str(alias or "").strip().lower()
    if not alias_text:
        return False
    if normalized_prompt == alias_text:
        return True
    if normalized_prompt.startswith(f"{alias_text} "):
        return True
    if f" {alias_text} " in f" {normalized_prompt} ":
        return True
    return normalized_prompt.endswith(f" {alias_text}")


def _matches_action_anchor(action: str, normalized_prompt: str, alias: str) -> bool:
    alias_text = str(alias or "").strip().lower()
    if not alias_text:
        return False
    if action in {"define", "explain", "solve", "calculate", "analyze", "plan", "brainstorm", "debug", "summarize", "continue"}:
        if normalized_prompt == alias_text or normalized_prompt.startswith(f"{alias_text} "):
            return True
        if action == "compare" and alias_text in {"vs", "versus"}:
            return _contains_anchor(normalized_prompt, alias_text)
        if action == "plan" and alias_text in {"roadmap", "blueprint", "design"}:
            return normalized_prompt == alias_text or normalized_prompt.startswith(f"{alias_text} ")
        return False
    if action == "compare":
        return _contains_anchor(normalized_prompt, alias_text)
    return _contains_anchor(normalized_prompt, alias_text)
