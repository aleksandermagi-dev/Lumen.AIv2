from __future__ import annotations

import re


class DesignSpecBuilder:
    """Builds a bounded system-spec payload for software and systems design prompts."""

    _SOFTWARE_MARKERS = (
        "api",
        "service",
        "workflow",
        "pipeline",
        "agent",
        "assistant",
        "app",
        "application",
        "dashboard",
        "platform",
        "system",
        "architecture",
        "tool",
        "engine",
        "cli",
    )

    _COMPONENT_MAP = {
        "api": "API boundary and request contract",
        "service": "Service layer for business logic",
        "workflow": "Workflow coordinator for multi-step execution",
        "pipeline": "Pipeline stages with explicit handoff contracts",
        "agent": "Agent orchestration layer with bounded authority",
        "assistant": "Interaction layer and response orchestration",
        "app": "Application shell and runtime entrypoints",
        "application": "Application shell and runtime entrypoints",
        "dashboard": "Presentation layer and state synchronization",
        "platform": "Shared platform services and deployment boundary",
        "system": "System boundary and core orchestration",
        "architecture": "Architecture boundary and module layout",
        "tool": "Tool adapter boundary and structured execution contract",
        "engine": "Domain engine with typed inputs and outputs",
        "cli": "CLI surface and command contract",
    }

    @classmethod
    def build(cls, *, brief: str, interaction_style: str = "default") -> dict[str, object]:
        normalized = " ".join(str(brief or "").strip().split())
        subject = cls._subject(normalized)
        subject_label = subject or "the system"
        design_domain = cls._domain(subject_label)
        constraints = cls._constraints(subject_label)
        tradeoffs = cls._tradeoffs(subject_label)
        failure_points = cls._failure_points(subject_label)
        components = cls._components(subject_label)
        resources = cls._resources(subject_label)
        summary = cls._summary(
            subject=subject_label,
            interaction_style=interaction_style,
            domain=design_domain,
        )
        next_steps = cls._next_steps(subject_label)
        return {
            "subject": subject_label,
            "design_domain": design_domain,
            "summary": summary,
            "system_overview": cls._system_overview(subject_label, design_domain),
            "components": components,
            "resources": resources,
            "constraints": constraints,
            "tradeoffs": tradeoffs,
            "failure_points": failure_points,
            "next_steps": next_steps,
            "assumptions": [
                f"Assume {subject_label} should be buildable as a bounded v1 rather than a maximal first release.",
                "Assume interfaces between layers should stay explicit so the design can evolve without large rewrites.",
            ],
        }

    @classmethod
    def _subject(cls, brief: str) -> str:
        normalized = str(brief or "").strip().lower()
        prefixes = (
            "design a ",
            "design an ",
            "design ",
            "build a ",
            "build an ",
            "build ",
            "create a ",
            "create an ",
            "create ",
            "plan a ",
            "plan an ",
            "plan ",
            "generate system spec for ",
            "generate a system spec for ",
        )
        subject = normalized
        for prefix in prefixes:
            if subject.startswith(prefix):
                subject = subject[len(prefix) :].strip()
                break
        subject = re.sub(r"\b(with|that|which)\b.+$", "", subject).strip(" .!?")
        return subject or normalized.strip(" .!?")

    @classmethod
    def _domain(cls, subject: str) -> str:
        normalized = subject.lower()
        if any(marker in normalized for marker in cls._SOFTWARE_MARKERS):
            return "software_system"
        return "general_system"

    @classmethod
    def _system_overview(cls, subject: str, design_domain: str) -> str:
        if design_domain == "software_system":
            return (
                f"Treat {subject} as a layered software system with a clear input surface, a core orchestration layer, "
                "typed domain logic, and bounded output adapters."
            )
        return (
            f"Treat {subject} as a modular system with an input stage, a coordinating core, support mechanisms, and a bounded output surface."
        )

    @classmethod
    def _components(cls, subject: str) -> list[str]:
        normalized = subject.lower()
        matched = [
            component
            for marker, component in cls._COMPONENT_MAP.items()
            if marker in normalized
        ]
        base = [
            "Entry surface for user or upstream inputs",
            "Decision layer that selects the next action without hiding authority",
            "Core execution layer with typed contracts",
            "Persistence or state boundary for continuity",
            "Validation and safety checks before final output",
        ]
        return matched + [item for item in base if item not in matched]

    @staticmethod
    def _resources(subject: str) -> list[str]:
        return [
            f"A concise requirements brief for {subject}",
            "Typed schemas or lightweight contracts for inputs and outputs",
            "A small validation suite for critical paths and regressions",
            "Runtime observability for failures, weak routes, and degraded outputs",
        ]

    @staticmethod
    def _constraints(subject: str) -> list[str]:
        return [
            f"Keep {subject} modular enough that one subsystem can change without forcing a full rewrite.",
            "Prefer deterministic behavior for routing, formatting, and tool selection.",
            "Keep safety, truthfulness, and uncertainty discipline above style.",
            "Optimize for a buildable v1 path instead of breadth-first feature sprawl.",
        ]

    @staticmethod
    def _tradeoffs(subject: str) -> list[str]:
        return [
            f"A richer {subject} surface improves usefulness but raises orchestration and testing complexity.",
            "More automation reduces manual work but makes failure recovery and observability more important.",
            "Tighter structure improves reliability but can feel rigid if the interaction layer does not soften the presentation.",
        ]

    @staticmethod
    def _failure_points(subject: str) -> list[str]:
        return [
            f"{subject} can become vague if responsibilities are not split cleanly between routing, execution, and presentation.",
            "Weak contracts between layers can leak scaffolding or produce incomplete response bodies.",
            "If continuity state is trusted too much, follow-up prompts can inherit the wrong goal or detail level.",
        ]

    @staticmethod
    def _next_steps(subject: str) -> list[str]:
        return [
            f"Lock the success criteria and primary user flow for {subject}.",
            "Define the minimum interface contracts between the main layers.",
            "Build the thinnest end-to-end slice first, then add failure handling and refinements.",
        ]

    @staticmethod
    def _summary(*, subject: str, interaction_style: str, domain: str) -> str:
        if interaction_style == "direct":
            return f"Build {subject} as a layered, buildable {domain.replace('_', ' ')} with clear contracts and a small first slice."
        if interaction_style == "collab":
            return (
                f"Here is a buildable first-pass spec for {subject}: keep the layers explicit, keep the contracts tight, and size the first version for clean iteration."
            )
        return (
            f"Here is a structured first-pass system spec for {subject}, aimed at a buildable v1 with clear layers, constraints, and tradeoffs."
        )
