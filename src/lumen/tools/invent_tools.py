from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_invent_params(*, input_path: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    # Placeholder for future structured design-brief ingestion.
    return dict(params)


def infer_brief(params: dict[str, Any]) -> str:
    brief = str(params.get("brief") or params.get("topic") or params.get("objective") or "").strip()
    return brief or "the design brief"


def infer_constraints(params: dict[str, Any]) -> list[str]:
    constraints = _normalize_list(params.get("constraints"))
    if constraints:
        return constraints
    return [
        "Keep the design bounded and testable.",
        "Prefer accessible materials and simple subsystems first.",
        "Make tradeoffs explicit instead of hiding them.",
    ]


def generate_concepts_payload(params: dict[str, Any]) -> dict[str, Any]:
    brief = infer_brief(params)
    constraints = infer_constraints(params)
    concepts = [
        {
            "name": "Baseline modular concept",
            "description": f"A simple modular approach for {brief} that keeps subsystems easy to swap and test.",
            "strengths": ["easier iteration", "clear failure isolation", "lower integration risk"],
            "tradeoffs": ["less optimized for peak performance", "may require more interface definition"],
        },
        {
            "name": "Efficiency-focused concept",
            "description": f"A tighter integrated concept for {brief} that prioritizes performance under the stated constraints.",
            "strengths": ["better peak efficiency", "fewer redundant components", "clear optimization target"],
            "tradeoffs": ["harder to prototype quickly", "more sensitive to assumption errors"],
        },
        {
            "name": "Resilience-first concept",
            "description": f"A conservative concept for {brief} that favors graceful degradation and maintainability.",
            "strengths": ["higher fault tolerance", "clear maintenance path", "more robust in uncertain conditions"],
            "tradeoffs": ["added mass or complexity", "can sacrifice performance headroom"],
        },
    ]
    return {
        "status": "ok",
        "invent_type": "generate_concepts",
        "brief": brief,
        "constraints": constraints,
        "concepts": concepts,
        "recommended_next_step": "Choose one concept direction and stress-test it against the strongest constraint.",
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def constraint_check_payload(params: dict[str, Any]) -> dict[str, Any]:
    brief = infer_brief(params)
    constraints = infer_constraints(params)
    concept = str(params.get("concept") or brief).strip()
    assessments = [
        {
            "constraint": item,
            "assessment": "partially satisfied" if index == 0 else "needs validation",
            "note": (
                f"The concept should be checked directly against '{item}'."
                if index == 0
                else f"Current prompt detail is not enough to prove compliance with '{item}'."
            ),
        }
        for index, item in enumerate(constraints)
    ]
    return {
        "status": "ok",
        "invent_type": "constraint_check",
        "brief": brief,
        "concept": concept,
        "constraints": constraints,
        "assessments": assessments,
        "overall_read": "The concept is directionally plausible, but the constraints still need explicit validation.",
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def material_suggestions_payload(params: dict[str, Any]) -> dict[str, Any]:
    brief = infer_brief(params)
    constraints = infer_constraints(params)
    materials = [
        {
            "material_class": "Aluminum alloys",
            "why_it_fits": "Good general-purpose balance of weight, machinability, and availability.",
            "watchouts": ["fatigue life", "heat limits in harsher environments"],
        },
        {
            "material_class": "Carbon-fiber composites",
            "why_it_fits": "Useful when stiffness-to-weight matters more than ease of repair.",
            "watchouts": ["repair complexity", "anisotropic behavior", "cost"],
        },
        {
            "material_class": "Engineering polymers",
            "why_it_fits": "Helpful for lightweight housings, insulation, and lower-load components.",
            "watchouts": ["creep", "temperature limits", "chemical compatibility"],
        },
    ]
    return {
        "status": "ok",
        "invent_type": "material_suggestions",
        "brief": brief,
        "constraints": constraints,
        "materials": materials,
        "selection_note": "Match the material choice to the most restrictive environment and load constraint first.",
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def failure_modes_payload(params: dict[str, Any]) -> dict[str, Any]:
    brief = infer_brief(params)
    concept = str(params.get("concept") or brief).strip()
    failure_modes = [
        {
            "mode": "interface mismatch",
            "severity": "medium",
            "note": "Subsystem assumptions can drift apart if interfaces are not pinned down early.",
        },
        {
            "mode": "material or component overstress",
            "severity": "high",
            "note": "Loads, heat, or duty cycles can exceed what the chosen concept can tolerate.",
        },
        {
            "mode": "maintenance complexity",
            "severity": "medium",
            "note": "A concept that looks efficient on paper can become fragile if service access is poor.",
        },
    ]
    return {
        "status": "ok",
        "invent_type": "failure_modes",
        "brief": brief,
        "concept": concept,
        "failure_modes": failure_modes,
        "mitigation_direction": "Prototype the riskiest interface or load-bearing assumption first.",
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def extract_brief_from_prompt(text: str) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return ""
    lowered = normalized.lower()
    for prefix in (
        "generate concept",
        "generate concepts",
        "create concept",
        "create concepts",
        "check concept constraints",
        "check constraints",
        "suggest materials",
        "analyze failure modes",
    ):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip(" :,-")
            lowered = normalized.lower()
            break
    for marker in ("for ", "about ", "around "):
        index = lowered.find(marker)
        if index != -1:
            candidate = normalized[index + len(marker) :].strip(" :,-")
            if candidate:
                return candidate
    return normalized


def extract_constraints_from_prompt(text: str) -> list[str]:
    normalized = " ".join(str(text or "").strip().split())
    lowered = normalized.lower()
    for marker in ("under these constraints", "within these constraints", "constraints:"):
        index = lowered.find(marker)
        if index != -1:
            fragment = normalized[index + len(marker) :].strip(" :,-")
            return _normalize_list(fragment)
    return []


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    return []
