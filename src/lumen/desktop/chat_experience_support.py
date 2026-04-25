from __future__ import annotations

import re
from typing import Any


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _response_text(response: dict[str, object]) -> str:
    for key in ("user_facing_answer", "reply", "summary"):
        value = str(response.get(key) or "").strip()
        if value:
            return value
    return ""


def _tool_execution(response: dict[str, object]) -> dict[str, object]:
    execution = response.get("tool_execution")
    return execution if isinstance(execution, dict) else {}


def _has_math_shape(text: str) -> bool:
    normalized = str(text or "")
    return bool(
        re.search(r"\b\d+[a-zA-Z]\b", normalized)
        or re.search(r"\b[a-zA-Z]\s*[\^²³]\s*\d*", normalized)
        or ("=" in normalized and re.search(r"[xyzXYZ]", normalized))
    )


def _task_label_from_tool(tool_id: str, capability: str) -> str:
    normalized_tool = _normalize(tool_id)
    normalized_capability = _normalize(capability)
    if normalized_tool == "math":
        return "Math Solve"
    if normalized_tool in {"data", "viz"}:
        return "Analysis"
    if normalized_tool == "paper":
        return "Research Review"
    if normalized_tool == "simulate":
        return "Theory"
    if normalized_tool in {"experiment", "invent"}:
        return "Invention"
    if normalized_tool == "system":
        if "docs" in normalized_capability:
            return "System Docs"
        return "System Analysis"
    if normalized_tool == "knowledge":
        return "Knowledge Check"
    if normalized_tool == "workspace":
        return "Workspace Review"
    if normalized_tool == "memory":
        return "Memory Review"
    if normalized_tool == "report":
        return "Report Review"
    return "Open Reasoning"


def infer_task_label(
    *,
    mode_label: str,
    response: dict[str, object] | None = None,
    prompt: str = "",
) -> str:
    payload = response or {}
    mode = _normalize(str(payload.get("mode") or ""))
    kind = _normalize(str(payload.get("kind") or ""))
    execution = _tool_execution(payload)
    if execution and not bool(payload.get("tool_execution_skipped")):
        return _task_label_from_tool(
            str(execution.get("tool_id") or ""),
            str(execution.get("capability") or ""),
        )
    if mode == "planning":
        if any(token in kind for token in ("design", "architecture")):
            return "Design Planning"
        return "Planning"
    if mode == "research":
        if "academic" in kind or "citation" in kind:
            return "Academic Support"
        if "literature" in kind:
            return "Research Review"
        if "dataset_readiness" in kind:
            return "Analysis"
        return "General Reasoning"
    if mode == "clarification":
        return "Clarification"
    if mode == "conversation":
        return "Conversation"
    if mode == "safety":
        return "Safety Boundary"

    normalized_prompt = _normalize(prompt)
    if _has_math_shape(prompt) or any(word in normalized_prompt for word in ("solve", "equation", "integrate", "matrix", "optimize")):
        return "Math Solve"
    if any(word in normalized_prompt for word in ("architecture", "refactor", "codebase", "module", "service", "system structure")):
        return "System Analysis"
    if any(word in normalized_prompt for word in ("simulate", "orbit", "diffusion", "population")):
        return "Theory"
    if any(word in normalized_prompt for word in ("invent", "design", "concept", "constraints", "materials", "failure modes")):
        return "Invention"
    if any(word in normalized_prompt for word in ("paper", "study", "research", "methods", "literature")):
        return "Research Review"
    if any(word in normalized_prompt for word in ("data", "graph", "chart", "plot", "correlate", "regression", "cluster")):
        return "Analysis"
    if any(word in normalized_prompt for word in ("relationship", "relate", "link", "contradiction", "cluster", "inconsistencies")):
        return "Knowledge Check"
    if any(word in normalized_prompt for word in ("design planning",)):
        return "Design Planning"
    if any(word in normalized_prompt for word in ("what", "why", "how", "explain", "tell me")):
        return "General Reasoning"
    return "Open Reasoning"


def build_context_bar(
    *,
    mode_label: str,
    response: dict[str, object] | None = None,
    prompt: str = "",
) -> str:
    return f"Mode: {mode_label} • Task: {infer_task_label(mode_label=mode_label, response=response, prompt=prompt)}"


def build_pending_label(*, mode_label: str, prompt: str = "") -> str:
    task_label = infer_task_label(mode_label=mode_label, prompt=prompt)
    if task_label == "Math Solve":
        return "Solving with math tools"
    if task_label in {"System Analysis", "System Docs"}:
        return "Analyzing..."
    if task_label in {"Design Planning", "Planning", "Invention"}:
        return "Designing..."
    if task_label == "Theory":
        return "Hypothesizing..."
    if task_label in {"Research Review", "Analysis"}:
        return "Analyzing..."
    if task_label == "Knowledge Check":
        normalized_prompt = _normalize(prompt)
        if any(token in normalized_prompt for token in ("contradiction", "inconsistencies", "check")):
            return "Checking..."
        return "Checking..."
    if task_label == "General Reasoning":
        return "Searching..."
    return "Thinking..."


def build_tool_transparency_line(response: dict[str, object]) -> str | None:
    if _normalize(str(response.get("mode") or "")) != "tool":
        return None
    if bool(response.get("tool_execution_skipped")):
        return None
    runtime = response.get("tool_runtime_status")
    if isinstance(runtime, dict) and str(runtime.get("failure_class") or "").strip() not in {"", "success"}:
        return None
    execution = _tool_execution(response)
    tool_id = _normalize(str(execution.get("tool_id") or ""))
    capability = _normalize(str(execution.get("capability") or ""))
    if not tool_id:
        return None
    if tool_id == "math":
        return "Using math tools..."
    if tool_id == "system":
        return "Preparing docs..." if "docs" in capability else "Analyzing system structure..."
    if tool_id == "knowledge":
        return "Checking knowledge links..."
    if tool_id == "workspace":
        return "Inspecting workspace..."
    if tool_id == "memory":
        return "Reviewing memory..."
    if tool_id == "report":
        return "Preparing a report..."
    if tool_id in {"data", "viz"}:
        return "Analyzing supplied data..."
    if tool_id == "paper":
        return "Reviewing paper context..."
    if tool_id == "content":
        return "Preparing writing support..."
    if tool_id in {"invent", "design", "experiment"}:
        return "Exploring a bounded design path..."
    return None


def build_capability_transparency_line(response: dict[str, object]) -> str | None:
    payload = response.get("capability_status")
    if not isinstance(payload, dict):
        return None
    status = _normalize(str(payload.get("status") or ""))
    details = str(payload.get("details") or "").strip()
    if status == "supported":
        return None
    labels = {
        "bounded": "Capability status: bounded.",
        "provider_gated": "Capability status: provider-gated.",
        "not_promised": "Capability status: not promised.",
    }
    lead = labels.get(status)
    if not lead:
        return None
    if details:
        return f"{lead} {details}"
    return lead


_MOMENTUM_POOLS: dict[str, tuple[str, ...]] = {
    "collab": (
        "Want to go deeper into this?",
        "We can test this next if you want.",
        "I can compare alternatives if that helps.",
        "Want me to break that down more simply?",
    ),
    "default": (
        "Want to go deeper into this?",
        "I can compare alternatives if that helps.",
        "I can break that down more simply if you want.",
        "We can test this next if that helps.",
    ),
    "direct": (
        "Want to go deeper?",
        "I can compare options next.",
        "I can break that down more simply.",
        "We can test this next.",
    ),
}


def maybe_attach_momentum_prompt(
    response: dict[str, object],
    *,
    style: str,
    recent_assistant_texts: list[str] | None = None,
) -> str | None:
    continuation_offer = response.get("continuation_offer")
    if isinstance(continuation_offer, dict):
        label = str(continuation_offer.get("label") or "").strip()
        if label:
            return label
    normalized_mode = _normalize(str(response.get("mode") or ""))
    normalized_kind = _normalize(str(response.get("kind") or ""))
    if normalized_mode in {"safety", "clarification", "conversation"}:
        return None
    if bool(response.get("tool_execution_skipped")):
        return None
    if "error" in normalized_kind or "refusal" in normalized_kind:
        return None

    base_text = _response_text(response)
    if len(base_text) < 80 or len(base_text.split()) < 12:
        return None

    recent_texts = [str(item).strip() for item in recent_assistant_texts or [] if str(item).strip()]
    normalized_recent = {_normalize(item) for item in recent_texts}
    all_prompts = {prompt for pool in _MOMENTUM_POOLS.values() for prompt in pool}
    if any(_normalize(prompt) in text for prompt in all_prompts for text in normalized_recent):
        return None

    normalized_style = _normalize(style)
    if normalized_style not in _MOMENTUM_POOLS:
        normalized_style = "default"
    pool = _MOMENTUM_POOLS[normalized_style]
    seed = f"{normalized_style}|{normalized_mode}|{normalized_kind}|{base_text[:80]}"
    choice = pool[sum(ord(char) for char in seed) % len(pool)]
    if _normalize(choice) in normalized_recent:
        return None
    return choice
