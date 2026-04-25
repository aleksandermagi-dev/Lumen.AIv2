from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolThresholdDecision:
    should_use_tool: bool
    rationale: str
    expected_confidence_gain: float
    selected_tool: str | None = None
    selected_bundle: str | None = None
    internal_reasoning_sufficient: bool = False
    tool_necessary: bool = False
    tool_higher_confidence: bool = False
    material_outcome_improvement: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "should_use_tool": self.should_use_tool,
            "rationale": self.rationale,
            "expected_confidence_gain": round(float(self.expected_confidence_gain), 4),
            "selected_tool": self.selected_tool,
            "selected_bundle": self.selected_bundle,
            "internal_reasoning_sufficient": self.internal_reasoning_sufficient,
            "tool_necessary": self.tool_necessary,
            "tool_higher_confidence": self.tool_higher_confidence,
            "material_outcome_improvement": self.material_outcome_improvement,
        }


class ToolThresholdGate:
    """Gate tool execution so routing intent does not automatically become execution."""

    _EXPLICIT_TOOL_CUES = (
        "run ",
        "use ",
        "inspect ",
        "scan ",
        "analyze ",
        "generate ",
        "execute ",
    )

    def decide(
        self,
        *,
        prompt: str,
        route_mode: str,
        route_kind: str,
        route_confidence: float,
        tool_id: str | None,
        capability: str | None,
        input_path: Path | None,
        params: dict[str, Any] | None,
    ) -> ToolThresholdDecision:
        normalized_route_mode = str(route_mode or "").strip()
        has_structured_input = input_path is not None or bool(params)
        embedded_tool_support = (
            bool(tool_id and capability)
            and normalized_route_mode in {"planning", "research"}
            and has_structured_input
        )
        if normalized_route_mode != "tool" and not embedded_tool_support:
            return ToolThresholdDecision(
                should_use_tool=False,
                rationale="Current route does not require live tool execution.",
                expected_confidence_gain=0.0,
                selected_tool=tool_id,
                selected_bundle=tool_id,
                internal_reasoning_sufficient=True,
            )

        normalized_prompt = " ".join(str(prompt or "").lower().split())
        explicit_tool_request = any(normalized_prompt.startswith(prefix) for prefix in self._EXPLICIT_TOOL_CUES)
        internal_reasoning_sufficient = (
            not has_structured_input
            and not explicit_tool_request
            and float(route_confidence or 0.0) < 0.72
        )
        tool_necessary = bool(tool_id and capability) and (
            has_structured_input
            or explicit_tool_request
            or float(route_confidence or 0.0) >= 0.78
            or "tool." in str(route_kind or "")
        )
        expected_confidence_gain = (
            0.34 if has_structured_input else
            0.24 if explicit_tool_request else
            0.18 if float(route_confidence or 0.0) >= 0.82 else
            0.08
        )
        tool_higher_confidence = has_structured_input or expected_confidence_gain >= 0.18
        material_outcome_improvement = has_structured_input or explicit_tool_request or float(route_confidence or 0.0) >= 0.78
        should_use_tool = bool(tool_id and capability) and tool_necessary and tool_higher_confidence and material_outcome_improvement and not internal_reasoning_sufficient

        if should_use_tool:
            rationale = "Tool execution is justified because it should materially improve confidence or provide structured output."
        elif internal_reasoning_sufficient:
            rationale = "Internal reasoning is sufficient for this turn, so live tool execution is not justified."
        else:
            rationale = "Tool execution would not materially improve the outcome enough to justify running it."

        return ToolThresholdDecision(
            should_use_tool=should_use_tool,
            rationale=rationale,
            expected_confidence_gain=expected_confidence_gain,
            selected_tool=tool_id,
            selected_bundle=tool_id,
            internal_reasoning_sufficient=internal_reasoning_sufficient,
            tool_necessary=tool_necessary,
            tool_higher_confidence=tool_higher_confidence,
            material_outcome_improvement=material_outcome_improvement,
        )
