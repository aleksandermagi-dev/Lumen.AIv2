from lumen.reporting.output_formatter import OutputFormatter


def test_output_formatter_hides_confidence_posture_for_belief_discussion() -> None:
    formatter = OutputFormatter()

    rendered = formatter.render_text(
        {
            "schema_type": "assistant_response",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Research response",
            "discussion_domain": "belief_tradition",
            "confidence_posture": "supported",
            "uncertainty_note": "This is better handled as a belief or interpretive question.",
            "findings": ["Interpret the traditions comparatively."],
        }
    )

    assert "discussion_domain: belief_tradition" in rendered
    assert "confidence_posture:" not in rendered
    assert "uncertainty_note: This is better handled as a belief or interpretive question." in rendered


def test_output_formatter_includes_user_facing_answer_when_present() -> None:
    formatter = OutputFormatter()

    rendered = formatter.render_text(
        {
            "schema_type": "assistant_response",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Black holes are regions of space where gravity becomes so strong that not even light can escape.",
            "user_facing_answer": "Black holes are regions of space where gravity becomes so strong that not even light can escape.",
            "findings": [],
        }
    )

    assert "answer: Black holes are regions of space where gravity becomes so strong" in rendered


def test_output_formatter_groups_diagnostics_away_from_user_facing_answer() -> None:
    formatter = OutputFormatter()

    rendered = formatter.render_text(
        {
            "schema_type": "assistant_response",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Black holes are regions of space where gravity becomes so strong that not even light can escape.",
            "user_facing_answer": "Black holes are regions of space where gravity becomes so strong that not even light can escape.",
            "resolved_prompt": "what is a black hole",
            "resolution_strategy": "wake_phrase_strip",
            "route": {
                "reason": "Explicit summary/explanation prompt matched summary-research routing",
                "strength": "high",
            },
            "findings": [],
        }
    )

    assert "answer: Black holes are regions of space where gravity becomes so strong" in rendered
    assert "diagnostics:" in rendered
    assert "resolved_prompt: what is a black hole" in rendered
    assert "route_reason: Explicit summary/explanation prompt matched summary-research routing" in rendered


def test_output_formatter_does_not_leak_route_support_signals_into_main_text() -> None:
    formatter = OutputFormatter()

    rendered = formatter.render_text(
        {
            "schema_type": "assistant_response",
            "mode": "research",
            "kind": "research.summary",
            "summary": "Voltage is electric potential difference.",
            "user_facing_answer": "Voltage is electric potential difference.",
            "route_support_signals": {
                "dominant_intent": "research",
                "broad_explanatory_prompt": True,
            },
            "findings": [],
        }
    )

    assert "answer: Voltage is electric potential difference." in rendered
    assert "route_support_signals" not in rendered
