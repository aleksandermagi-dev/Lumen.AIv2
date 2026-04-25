from __future__ import annotations

from lumen.evaluation import (
    EvaluationRunner,
    LongConversationEvaluator,
    ScriptedConversationScenario,
    ScriptedTurnExpectation,
    phase80_default_scenarios,
)


def _record(
    prompt: str,
    *,
    mode: str = "conversation",
    kind: str = "conversation.reply",
    style: str = "collab",
    signals: list[str] | None = None,
    project_active: bool = False,
    work_intent: str | None = None,
    memory_count: int = 0,
    recall_prompt: bool = False,
    repetition_risk: str = "low",
    follow_up_allowed: bool = False,
    reply: str = "Natural local response.",
) -> dict[str, object]:
    selected = [
        {"source": "personal_memory", "memory_kind": "durable_user_memory"}
        for _ in range(memory_count)
    ]
    response: dict[str, object] = {
        "mode": mode,
        "kind": kind,
        "reply": reply,
        "assistant_quality_posture": {
            "style_mode": style,
            "conversation_boundary_signals": signals or [],
        },
        "assistant_voice_profile": {"style_mode": style},
        "conversation_beat": {
            "response_repetition_risk": repetition_risk,
            "follow_up_offer_allowed": follow_up_allowed,
        },
        "memory_retrieval": {
            "selected": selected,
            "recall_prompt": recall_prompt,
        },
        "project_context_snapshot": {
            "project_context_active": project_active,
        },
    }
    if work_intent:
        response["work_thread_continuity"] = {
            "active": True,
            "intent": work_intent,
        }
    return {"prompt": prompt, "mode": mode, "kind": kind, "response": response}


def test_phase80_default_scenarios_are_long_and_cover_core_tracks() -> None:
    scenarios = phase80_default_scenarios()

    assert {scenario.name for scenario in scenarios} == {
        "phase80_human_chat",
        "phase80_memory_restraint",
        "phase80_work_thread",
    }
    assert all(len(scenario.turns) >= 20 for scenario in scenarios)
    assert all(scenario.min_turns == 20 for scenario in scenarios)


def test_long_conversation_evaluator_accepts_healthy_scripted_transcript() -> None:
    turns = tuple(
        ScriptedTurnExpectation(
            f"turn {index}",
            expected_mode="conversation",
            expected_kind_prefix="conversation.",
            max_personal_memory=0,
        )
        for index in range(20)
    )
    scenario = ScriptedConversationScenario(
        name="healthy",
        description="Healthy long chat.",
        style_mode="collab",
        turns=turns,
    )
    records = [_record(turn.prompt) for turn in turns]

    evaluation = LongConversationEvaluator().evaluate(scenario=scenario, records=records)

    assert evaluation.judgment == "correct"
    assert evaluation.score == 1.0
    assert evaluation.metrics["route_passes"] == 20
    assert evaluation.findings == ()


def test_long_conversation_evaluator_flags_route_memory_repetition_and_scaffold_drift() -> None:
    turns = tuple(
        ScriptedTurnExpectation(
            f"turn {index}",
            expected_mode="conversation",
            expected_kind_prefix="conversation.",
            max_personal_memory=0,
        )
        for index in range(20)
    )
    scenario = ScriptedConversationScenario(
        name="drift",
        description="Broken long chat.",
        style_mode="direct",
        turns=turns,
    )
    records = [
        _record(turn.prompt, style="direct")
        for turn in turns
    ]
    records[1] = _record("turn 1", mode="research", kind="research.summary", style="direct")
    records[2] = _record("turn 2", style="collab")
    records[3] = _record("turn 3", style="direct", memory_count=2)
    records[4] = _record(
        "turn 4",
        style="direct",
        repetition_risk="high",
        follow_up_allowed=True,
    )
    records[5] = _record("turn 5", style="direct", reply="Best first read: route validation says yes.")

    evaluation = LongConversationEvaluator().evaluate(scenario=scenario, records=records)
    categories = {finding.category for finding in evaluation.findings}

    assert evaluation.judgment in {"weak", "incorrect"}
    assert "route_drift" in categories
    assert "tone_drift" in categories
    assert "memory_noise" in categories
    assert "repetition_drift" in categories
    assert "scaffold_leakage" in categories
    assert evaluation.metrics["memory_violations"] == 1
    assert evaluation.metrics["repetition_violations"] == 1
    assert evaluation.metrics["scaffold_violations"] == 1


def test_long_conversation_evaluator_scores_work_thread_and_project_expectations() -> None:
    turns = tuple(
        ScriptedTurnExpectation(
            f"work {index}",
            expected_mode="conversation",
            require_project_context=True,
            required_work_thread_intent="next_step",
        )
        for index in range(20)
    )
    scenario = ScriptedConversationScenario(
        name="work",
        description="Work continuity.",
        style_mode="default",
        turns=turns,
    )
    records = [
        _record(
            turn.prompt,
            style="default",
            project_active=True,
            work_intent="next_step",
        )
        for turn in turns
    ]

    evaluation = EvaluationRunner().evaluate_scripted_conversation(
        scenario=scenario,
        records=records,
    )

    assert evaluation.judgment == "correct"
    assert evaluation.metrics["project_passes"] == 20
    assert evaluation.metrics["work_thread_passes"] == 20


def test_long_conversation_evaluator_requires_scripted_turn_floor() -> None:
    scenario = ScriptedConversationScenario(
        name="too_short",
        description="Short script.",
        turns=(ScriptedTurnExpectation("hi", expected_mode="conversation"),),
    )

    evaluation = LongConversationEvaluator().evaluate(
        scenario=scenario,
        records=[_record("hi", style="default")],
    )

    assert evaluation.judgment == "weak"
    assert any(finding.category == "scenario_shape" for finding in evaluation.findings)
