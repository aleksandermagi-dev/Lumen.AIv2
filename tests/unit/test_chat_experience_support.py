from lumen.desktop.chat_experience_support import (
    build_capability_transparency_line,
    build_context_bar,
    build_pending_label,
    build_tool_transparency_line,
    maybe_attach_momentum_prompt,
)


def test_context_bar_uses_human_readable_math_task() -> None:
    line = build_context_bar(
        mode_label="Collab",
        response={
            "mode": "tool",
            "tool_execution": {"tool_id": "math", "capability": "solve_equation"},
        },
    )

    assert line == "Mode: Collab • Task: Math Solve"


def test_context_bar_falls_back_to_open_reasoning_for_low_signal_prompt() -> None:
    line = build_context_bar(mode_label="Default", prompt="hello there")

    assert line == "Mode: Default • Task: Open Reasoning"


def test_context_bar_uses_academic_support_label_for_academic_research_kind() -> None:
    line = build_context_bar(
        mode_label="Collab",
        response={
            "mode": "research",
            "kind": "research.academic_citation",
        },
    )

    assert line == "Mode: Collab • Task: Academic Support"


def test_pending_labels_map_to_task_family() -> None:
    assert build_pending_label(mode_label="Collab", prompt="solve 3x^2 + 2x - 5 = 0") == "Solving with math tools"
    assert build_pending_label(mode_label="Default", prompt="analyze this system structure") == "Analyzing..."
    assert build_pending_label(mode_label="Default", prompt="find inconsistencies in these claims") == "Checking..."
    assert build_pending_label(mode_label="Default", prompt="tell me about black holes") == "Searching..."


def test_tool_transparency_line_is_hidden_when_tool_execution_is_skipped() -> None:
    assert (
        build_tool_transparency_line(
            {
                "mode": "tool",
                "tool_execution_skipped": True,
                "tool_execution": {"tool_id": "math", "capability": "solve_equation"},
            }
        )
        is None
    )


def test_tool_transparency_line_handles_data_and_invent_tools() -> None:
    assert (
        build_tool_transparency_line(
            {
                "mode": "tool",
                "tool_execution": {"tool_id": "data", "capability": "describe"},
            }
        )
        == "Analyzing supplied data..."
    )
    assert (
        build_tool_transparency_line(
            {
                "mode": "tool",
                "tool_execution": {"tool_id": "invent", "capability": "generate_concepts"},
            }
        )
        == "Exploring a bounded design path..."
    )


def test_capability_transparency_line_surfaces_bounded_and_not_promised_states() -> None:
    assert (
        build_capability_transparency_line(
            {
                "capability_status": {
                    "status": "bounded",
                    "details": "Analysis stays bounded to local data.",
                }
            }
        )
        == "Capability status: bounded. Analysis stays bounded to local data."
    )
    assert (
        build_capability_transparency_line(
            {
                "capability_status": {
                    "status": "not_promised",
                    "details": "This surface is not part of the runtime contract.",
                }
            }
        )
        == "Capability status: not promised. This surface is not part of the runtime contract."
    )


def test_momentum_prompt_is_added_for_substantive_research_reply() -> None:
    prompt = maybe_attach_momentum_prompt(
        {
            "mode": "research",
            "kind": "research.summary",
            "summary": (
                "Black holes are regions of spacetime where gravity becomes so strong that "
                "nothing, including light, can escape once it crosses the event horizon."
            ),
        },
        style="collab",
        recent_assistant_texts=[],
    )

    assert prompt in {
        "Want to go deeper into this?",
        "We can test this next if you want.",
        "I can compare alternatives if that helps.",
        "Want me to break that down more simply?",
    }


def test_momentum_prompt_is_suppressed_for_safety_and_recent_repeat() -> None:
    assert (
        maybe_attach_momentum_prompt(
            {
                "mode": "safety",
                "kind": "safety.refusal",
                "summary": "I can't help with that.",
            },
            style="default",
            recent_assistant_texts=[],
        )
        is None
    )
    assert (
        maybe_attach_momentum_prompt(
            {
                "mode": "planning",
                "kind": "planning.architecture",
                "summary": (
                    "We should split orchestration into smaller helpers so route authority, "
                    "tool execution, and final answer shaping are easier to reason about."
                ),
            },
            style="default",
            recent_assistant_texts=["Want to go deeper into this?"],
        )
        is None
    )
