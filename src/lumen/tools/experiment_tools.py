from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_experiment_params(*, input_path: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    # Placeholder for future structured file support. Current phase is prompt/param driven.
    return dict(params)


def infer_topic(params: dict[str, Any]) -> str:
    topic = str(params.get("topic") or params.get("hypothesis") or params.get("objective") or "").strip()
    return topic or "the study question"


def infer_hypothesis(params: dict[str, Any]) -> str:
    hypothesis = str(params.get("hypothesis") or "").strip()
    if hypothesis:
        return hypothesis
    topic = infer_topic(params)
    return f"If conditions change around {topic}, a measurable outcome should change as well."


def experiment_design_payload(params: dict[str, Any]) -> dict[str, Any]:
    topic = infer_topic(params)
    hypothesis = infer_hypothesis(params)
    independent = str(params.get("independent_variable") or "the manipulated condition").strip()
    dependent = str(params.get("dependent_variable") or "the measured outcome").strip()
    controls = _normalize_list(params.get("controls")) or [
        "Keep baseline environmental conditions stable.",
        "Use the same measurement schedule across all groups.",
        "Use the same instrumentation or rubric for every sample.",
    ]
    procedure = [
        "Define a baseline and at least one comparison condition.",
        f"Manipulate {independent} while keeping the rest of the setup stable.",
        f"Measure {dependent} at regular intervals.",
        "Record observations consistently and compare groups at the end.",
    ]
    return {
        "status": "ok",
        "experiment_type": "design",
        "topic": topic,
        "hypothesis": hypothesis,
        "independent_variable": independent,
        "dependent_variable": dependent,
        "controls": controls,
        "procedure": procedure,
        "measurements": [
            dependent,
            "baseline covariates that could explain variation",
            "notes on unexpected conditions or anomalies",
        ],
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def experiment_variables_payload(params: dict[str, Any]) -> dict[str, Any]:
    topic = infer_topic(params)
    independent = str(params.get("independent_variable") or "the intervention or condition being changed").strip()
    dependent = str(params.get("dependent_variable") or "the response being measured").strip()
    controlled = _normalize_list(params.get("controlled_variables")) or [
        "environmental conditions",
        "measurement timing",
        "sample handling",
    ]
    confounders = _normalize_list(params.get("confounders")) or [
        "selection bias between groups",
        "uncontrolled environmental drift",
        "measurement inconsistency",
    ]
    return {
        "status": "ok",
        "experiment_type": "variables",
        "topic": topic,
        "independent_variable": independent,
        "dependent_variable": dependent,
        "controlled_variables": controlled,
        "confounders": confounders,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def experiment_controls_payload(params: dict[str, Any]) -> dict[str, Any]:
    topic = infer_topic(params)
    controls = _normalize_list(params.get("controls")) or [
        "Use a baseline or untreated comparison group.",
        "Standardize timing, temperature, and equipment across runs.",
        "Randomize assignment when possible.",
    ]
    mitigations = [
        "Document calibration and setup before each run.",
        "Predefine exclusion criteria and keep them consistent.",
        "Record deviations immediately instead of adjusting them later.",
    ]
    return {
        "status": "ok",
        "experiment_type": "controls",
        "topic": topic,
        "controls": controls,
        "bias_mitigations": mitigations,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def experiment_analysis_plan_payload(params: dict[str, Any]) -> dict[str, Any]:
    topic = infer_topic(params)
    dependent = str(params.get("dependent_variable") or "the main measured outcome").strip()
    analysis_steps = [
        "Check raw measurements for missing values or obvious anomalies.",
        "Summarize each group with central tendency and spread.",
        "Compare baseline and experimental conditions against the stated hypothesis.",
        "Visualize the main outcome and report uncertainty or limitations.",
    ]
    outputs = [
        "summary table of groups",
        "trend or comparison chart",
        "written interpretation tied back to the hypothesis",
    ]
    return {
        "status": "ok",
        "experiment_type": "analysis_plan",
        "topic": topic,
        "primary_outcome": dependent,
        "analysis_steps": analysis_steps,
        "recommended_outputs": outputs,
        "runtime_diagnostics": {"runtime_ready": True, "input_ready": True},
    }


def extract_topic_from_prompt(text: str) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return ""
    lowered = normalized.lower()
    for prefix in (
        "design experiment",
        "design an experiment",
        "identify experiment variables",
        "identify variables",
        "identify experiment controls",
        "identify controls",
        "plan experiment analysis",
        "experiment analysis plan",
    ):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip(" :,-")
            lowered = normalized.lower()
            break
    for marker in ("to test whether ", "for ", "about ", "around "):
        index = lowered.find(marker)
        if index != -1:
            candidate = normalized[index + len(marker) :].strip(" :,-")
            if candidate:
                return candidate
    return normalized


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
        return parts
    return []
