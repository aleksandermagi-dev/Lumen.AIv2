from lumen.reasoning.reasoning_language import ReasoningResponseLanguage
from lumen.reasoning.response_flow_realizer import ResponseFlowRealizer


def test_reasoning_language_maps_confidence_posture_to_language_tier() -> None:
    assert ReasoningResponseLanguage.language_confidence_tier(confidence_posture="strong") == "high"
    assert ReasoningResponseLanguage.language_confidence_tier(confidence_posture="supported") == "medium"
    assert ReasoningResponseLanguage.language_confidence_tier(confidence_posture="tentative") == "low"
    assert ReasoningResponseLanguage.language_confidence_tier(confidence_posture="conflicted") == "low"


def test_response_flow_realizer_uses_medium_intro_for_supported_research_response() -> None:
    intro = ResponseFlowRealizer.intro_for_response(
        response={
            "confidence_posture": "supported",
            "reasoning_state": {
                "selected_mode": "default",
                "intent_domain": "technical_engineering",
            },
            "response_behavior_posture": {"visible_uncertainty": False},
        },
        mode="research",
    )

    assert intro == "Here’s the clearest technical read so far, keeping the uncertainty visible."


def test_response_flow_realizer_uses_low_intro_and_next_check_for_tentative_planning_response() -> None:
    intro = ResponseFlowRealizer.intro_for_response(
        response={
            "confidence_posture": "tentative",
            "reasoning_state": {
                "selected_mode": "default",
                "intent_domain": "planning_strategy",
            },
            "response_behavior_posture": {"visible_uncertainty": True},
        },
        mode="planning",
    )
    next_label = ResponseFlowRealizer.next_label_for_response(
        response={
            "confidence_posture": "tentative",
            "reasoning_state": {"selected_mode": "default"},
            "response_behavior_posture": {"visible_uncertainty": True},
        },
        mode="planning",
    )

    assert intro == "Here’s the roadmap I’d start with, keeping the open assumptions visible."
    assert next_label == "Next check:"


def test_reasoning_language_keeps_supported_route_caution_visible_without_low_confidence_wording() -> None:
    note = ReasoningResponseLanguage.uncertainty_note(
        confidence_posture="supported",
        local_context_assessment="partial",
        reasoning_frame={"primary_anchor": "archive structure"},
        route_caution="Route selection remained close because research and planning cues were both plausible.",
        route_ambiguity=False,
        subject_label="answer",
    )

    assert note is not None
    assert "route choice" in note.lower() or "route" in note.lower()
