from lumen.routing.anchor_registry import (
    detect_action_anchors,
    detect_domain_anchors,
    detect_explanation_mode,
    detect_follow_up_anchor,
    resolve_anchor_context,
)


def test_anchor_registry_detects_requested_domains() -> None:
    assert "astronomy" in detect_domain_anchors("what is a comet")
    assert "physics" in detect_domain_anchors("what is gravity")
    assert "math" in detect_domain_anchors("solve 3x + 2 = 11")
    assert "history" in detect_domain_anchors("tell me about ancient egypt")
    assert "engineering" in detect_domain_anchors("how do I build a ship")
    assert "ai_computing" in detect_domain_anchors("what is an operating system")


def test_anchor_registry_detects_requested_actions() -> None:
    assert "define" in detect_action_anchors("what is a comet")
    assert "compare" in detect_action_anchors("compare gravity and magnetism")
    assert "solve" in detect_action_anchors("solve 3x + 2 = 11")
    assert "break_down" in detect_action_anchors("break it down")
    assert "go_deeper" in detect_action_anchors("go deeper")
    assert "plan" in detect_action_anchors("design a ship blueprint")


def test_anchor_registry_detects_follow_up_confirmation_and_explanation_mode() -> None:
    yes_anchor = detect_follow_up_anchor("yes")
    break_down_anchor = detect_follow_up_anchor("break it down")

    assert yes_anchor is not None
    assert yes_anchor.kind == "confirmation"
    assert yes_anchor.requires_context is True

    assert break_down_anchor is not None
    assert break_down_anchor.kind == "explanation"
    assert break_down_anchor.explanation_mode == "break_down"
    assert detect_explanation_mode("walk me through it") == "step_by_step"


def test_anchor_registry_prefers_current_topic_from_context() -> None:
    resolution = resolve_anchor_context(
        "go deeper",
        recent_interactions=[
            {
                "prompt": "what is a comet",
                "response": {
                    "mode": "research",
                    "domain_surface": {"lane": "knowledge", "topic": "comet"},
                },
            }
        ],
        active_thread={
            "mode": "tool",
            "normalized_topic": "3x + 2 = 11",
            "tool_context": {"tool_id": "math", "capability": "solve_equation"},
        },
    )

    assert resolution.primary_domain == "astronomy"
    assert resolution.explanation_mode == "deeper"
    assert resolution.topic_anchor == "comet"
    assert resolution.capability_hint == "explanation_transform"
