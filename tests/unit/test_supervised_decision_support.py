from lumen.reasoning.pipeline_models import IntentDomainResult
from lumen.reasoning.supervised_decision_support import (
    SupervisedDecisionSupport,
    SupervisedExample,
)
from lumen.reasoning.tool_threshold_gate import ToolThresholdDecision


def test_supervised_decision_support_can_apply_low_confidence_intent_domain_tiebreak() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "intent_domain_classification": [
                SupervisedExample(
                    surface="intent_domain_classification",
                    label="planning_strategy",
                    prompt_terms=("roadmap", "milestones", "prioritize"),
                    route_modes=("planning",),
                )
            ]
        }
    )

    updated, recommendation = support.assist_intent_domain(
        prompt="create a roadmap, define milestones, and prioritize rollout order",
        route_mode="planning",
        route_kind="planning.migration",
        current=IntentDomainResult(
            domain="conversational",
            confidence=0.61,
            rationale="Weak default domain guess.",
            signals=["default_conversational"],
        ),
    )

    assert updated is not None
    assert updated.domain == "planning_strategy"
    assert recommendation is not None
    assert recommendation.applied is True


def test_supervised_decision_support_keeps_tool_gate_deterministic() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "tool_use_decision_support": [
                SupervisedExample(
                    surface="tool_use_decision_support",
                    label="skip_tool",
                    prompt_terms=("summarize", "explain"),
                    route_modes=("tool",),
                    tool_id="anh",
                )
            ]
        }
    )

    recommendation = support.advise_tool_decision(
        prompt="summarize the last anh run",
        route_mode="tool",
        route_kind="tool.command_alias",
        tool_id="anh",
        current_decision=ToolThresholdDecision(
            should_use_tool=True,
            rationale="Structured inputs justify a live run.",
            expected_confidence_gain=0.3,
            selected_tool="anh",
            selected_bundle="anh",
            tool_necessary=True,
            tool_higher_confidence=True,
            material_outcome_improvement=True,
        ),
    )

    assert recommendation is not None
    assert recommendation.recommended_label == "skip_tool"
    assert recommendation.applied is False
    assert "deterministic" in str(recommendation.applied_reason).lower()


def test_supervised_decision_support_records_route_recommendation_without_overriding_authority() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "route_recommendation_support": [
                SupervisedExample(
                    surface="route_recommendation_support",
                    label="research:research.summary",
                    prompt_terms=("summarize", "archive"),
                    route_modes=("research",),
                    route_kinds=("research.summary",),
                )
            ]
        }
    )

    recommendation = support.advise_route_decision(
        prompt="summarize the archive structure",
        route_mode="research",
        route_kind="research.summary",
        current_mode="research",
        current_kind="research.summary",
    )

    assert recommendation is not None
    assert recommendation.applied is False
    assert "agrees" in str(recommendation.applied_reason).lower()


def test_supervised_decision_support_can_apply_more_cautious_confidence_calibration() -> None:
    support = SupervisedDecisionSupport(
        examples_by_surface={
            "confidence_calibration_support": [
                SupervisedExample(
                    surface="confidence_calibration_support",
                    label="tentative",
                    prompt_terms=("migration", "summary"),
                    route_modes=("planning",),
                )
            ]
        }
    )

    tier, posture, recommendation = support.assist_confidence_calibration(
        prompt="review the migration summary",
        route_mode="planning",
        route_kind="planning.migration",
        current_tier="medium",
        current_posture="supported",
        route_status="under_tension",
        support_status="moderately_supported",
    )

    assert tier == "low"
    assert posture == "tentative"
    assert recommendation is not None
    assert recommendation.applied is True
