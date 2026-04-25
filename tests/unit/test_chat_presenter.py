from types import SimpleNamespace

from lumen.desktop.chat_presenter import ChatPresenter
from lumen.desktop.chat_app import LumenDesktopApp
from lumen.desktop.chat_ui_support import COGNITIVE_INDICATOR_POOLS, normalize_cognitive_mode
from lumen.desktop.main import build_parser
from lumen.reasoning.mode_response_shaper import ModeResponseShaper


def test_chat_presenter_renders_research_response() -> None:
    response = {
        "mode": "research",
        "summary": "Black holes overview",
        "findings": ["Black holes bend spacetime.", "They can form from stellar collapse."],
        "recommendation": "Ask a follow-up about event horizons if you want more depth.",
    }

    rendered = ChatPresenter.render(response)

    assert rendered == "Black holes overview"


def test_chat_presenter_hides_structured_scaffold_from_main_reply_surface() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "planning",
            "summary": "We should start with the archive boundary first.",
            "steps": ["Internal scaffold should not be rendered."],
            "next_action": "Also internal.",
        }
    )

    assert rendered == "We should start with the archive boundary first."


def test_chat_presenter_builds_provider_status() -> None:
    status = ChatPresenter.build_status(
        {
            "mode": "research",
            "provider_inference": {
                "provider_id": "openai_responses",
                "model": "gpt-test",
            },
        }
    )

    assert status == "research via openai_responses:gpt-test"


def test_chat_presenter_prefers_user_facing_answer_when_present() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Tentative research response for: black holes",
            "user_facing_answer": "Black holes are regions of space where gravity becomes so strong that not even light can escape.",
            "findings": ["Internal scaffold should not be shown."],
        }
    )

    assert rendered == "Black holes are regions of space where gravity becomes so strong that not even light can escape."


def test_chat_presenter_rewrites_scaffolded_galaxy_answer_into_normal_explanation() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Here's a grounded answer using the best current assumptions.",
            "findings": [
                "A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity."
            ],
            "wake_resolution": {"original_prompt": "what is a galaxy?"},
        }
    )

    assert "galaxy" in rendered.lower()
    assert "first pass" not in rendered.lower()
    assert "best current assumptions" not in rendered.lower()
    assert "- " not in rendered


def test_chat_presenter_keeps_math_why_explanation_direct_without_plan_scaffold() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "conversation",
            "summary": "Here's a grounded answer using the best current assumptions.",
            "reply": "5 - 5 equals 0 because subtracting a number from itself leaves nothing left.",
            "wake_resolution": {"original_prompt": "why?"},
        }
    )

    assert "5 - 5" in rendered or "equals 0" in rendered
    assert "first pass" not in rendered.lower()
    assert "\n- " not in rendered


def test_chat_presenter_rewrites_scaffolded_praise_reply_to_natural_acknowledgment() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "conversation",
            "summary": "Here's a grounded answer using the best current assumptions.",
            "wake_resolution": {"original_prompt": "good job"},
        }
    )

    assert "thanks" in rendered.lower() or "glad" in rendered.lower()
    assert "first pass" not in rendered.lower()


def test_chat_presenter_suppresses_best_first_read_scaffold_for_greeting() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Here's the best first read, and I'd still hold it provisionally. Best next check: validate the route.",
            "wake_resolution": {"original_prompt": "hey buddy"},
        }
    )

    lowered = rendered.lower()
    assert "best first read" not in lowered
    assert "provisionally" not in lowered
    assert "best next check" not in lowered
    assert rendered == "Let me answer that more directly."


def test_chat_presenter_keeps_anh_knowledge_answer_clean() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "ANH is Lumen's Astronomical Node Heuristics tool for scanning compatible astronomy spectra.",
            "domain_surface": {"lane": "knowledge", "topic": "ANH", "entry_id": "astronomy.anh"},
            "wake_resolution": {"original_prompt": "what is ANH"},
        }
    )

    lowered = rendered.lower()
    assert "anh" in lowered
    assert "best first read" not in lowered
    assert "validation plan" not in lowered
    assert "route" not in lowered


def test_chat_presenter_keeps_broad_domain_knowledge_answer_clean() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Chemistry studies matter, atoms, molecules, bonding, reactions, materials, and how substances change.",
            "domain_surface": {"lane": "knowledge", "topic": "Chemistry", "entry_id": "chemistry.chemistry"},
            "wake_resolution": {"original_prompt": "how does chemistry work"},
        }
    )

    lowered = rendered.lower()
    assert "chemistry" in lowered
    assert "best first read" not in lowered
    assert "provisionally" not in lowered
    assert "route" not in lowered
    assert "validation" not in lowered


def test_chat_presenter_uses_grounded_body_when_summary_is_internal_draft() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Here's a grounded answer.",
            "findings": ["A galaxy is a massive system of stars and gas bound together by gravity."],
            "recommendation": "Ask about black holes if you want a related example.",
            "wake_resolution": {"original_prompt": "what is a galaxy?"},
        }
    )

    assert "galaxy" in rendered.lower()
    assert "workable answer" not in rendered.lower()


def test_chat_presenter_can_add_tool_transparency_and_momentum_when_decorated() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "tool",
            "summary": (
                "Solved equation for x. The quadratic has two real roots, and both satisfy "
                "the original expression when substituted back into the equation."
            ),
            "tool_execution": {"tool_id": "math", "capability": "solve_equation"},
        },
        decorate=True,
        style="default",
        recent_assistant_texts=[],
    )

    assert rendered.startswith("Using math tools...")
    assert "Solved equation for x." in rendered
    assert any(
        prompt in rendered
        for prompt in {
            "Want to go deeper into this?",
            "I can compare alternatives if that helps.",
            "I can break that down more simply if you want.",
            "We can test this next if that helps.",
        }
    )


def test_chat_presenter_can_add_capability_status_line_when_decorated() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "tool",
            "summary": "Generated concept set for the propulsion brief.",
            "tool_execution": {"tool_id": "invent", "capability": "generate_concepts"},
            "capability_status": {
                "status": "bounded",
                "details": "Design support stays conceptual and non-signoff.",
            },
        },
        decorate=True,
        style="default",
        recent_assistant_texts=[],
    )

    assert rendered.startswith("Exploring a bounded design path...")
    assert "Capability status: bounded. Design support stays conceptual and non-signoff." in rendered


def test_chat_presenter_prefers_tool_user_facing_answer_when_present() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "tool",
            "summary": "Solved equation for x",
            "user_facing_answer": "Solved equation for x: x = 3",
        }
    )

    assert rendered == "Solved equation for x: x = 3"


def test_mode_response_shaper_surfaces_runtime_diagnostic_without_trace_noise() -> None:
    response = {
        "mode": "tool",
        "summary": "The math.solve_equation tool reached execution but failed with RuntimeError.",
        "tool_runtime_status": {
            "failure_class": "runtime_dependency_failure",
            "runtime_diagnostics": {"exception_message": "simulated adapter failure"},
        },
        "runtime_diagnostic": {
            "failure_stage": "execution",
            "failure_class": "runtime_dependency_failure",
            "tool_id": "math",
            "capability": "solve_equation",
            "exception_type": "RuntimeError",
            "safe_message": "The math.solve_equation tool reached execution but failed with RuntimeError.",
            "debug_details": {"exception_message": "simulated adapter failure"},
        },
    }

    ModeResponseShaper.apply(response=response, interaction_profile=SimpleNamespace(interaction_style="collab"))
    rendered = ChatPresenter.render(response)

    assert "an execution stage issue" in rendered.lower()
    assert "runtimeerror" in rendered.lower()
    assert "simulated adapter failure" not in rendered.lower()


def test_chat_presenter_keeps_safety_replies_undecorated() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "safety",
            "summary": "I can't help with that.",
            "boundary_explanation": "I can't help with that.",
            "safe_redirects": ["I can help with safety best practices instead."],
        },
        decorate=True,
        style="collab",
        recent_assistant_texts=[],
    )

    assert "Want to go deeper" not in rendered
    assert "Using math tools..." not in rendered


def test_chat_presenter_marks_constrained_research_as_high_level() -> None:
    rendered = ChatPresenter.render(
        {
            "mode": "research",
            "summary": "Here is a high-level overview of how satellite tracking works.",
            "response_constraint": {"level": "high_level_only"},
        }
    )

    assert "Here is a high-level overview of how satellite tracking works." in rendered
    assert "Keeping this high-level for safety." in rendered


def test_desktop_modules_import_cleanly() -> None:
    parser = build_parser()

    assert parser.prog == "lumen-desktop"
    assert callable(LumenDesktopApp._new_session_id)
    assert LumenDesktopApp.MODE_OPTIONS["Direct"] == "direct"
    assert LumenDesktopApp.MODE_OPTIONS["Default"] == "default"
    assert LumenDesktopApp.MODE_OPTIONS["Collab"] == "collab"
    assert any("entropy" in prompt.lower() for prompt in LumenDesktopApp.STARTER_PROMPT_OPTIONS)
    assert any("run anh" in prompt.lower() for prompt in LumenDesktopApp.STARTER_PROMPT_OPTIONS)


def test_cognitive_indicator_pools_stay_mode_locked() -> None:
    conversation = set(COGNITIVE_INDICATOR_POOLS["conversation"])
    research = set(COGNITIVE_INDICATOR_POOLS["research"])
    engineering = set(COGNITIVE_INDICATOR_POOLS["engineering"])

    assert conversation.isdisjoint(research)
    assert conversation.isdisjoint(engineering)
    assert research.isdisjoint(engineering)


def test_normalize_cognitive_mode_maps_engineering_modes() -> None:
    assert normalize_cognitive_mode("planning") == "engineering"
    assert normalize_cognitive_mode("tool") == "engineering"
    assert normalize_cognitive_mode("research") == "research"
    assert normalize_cognitive_mode("conversation") == "conversation"
    assert normalize_cognitive_mode("clarification") == "conversation"
