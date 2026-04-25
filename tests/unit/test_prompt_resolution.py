from lumen.routing.prompt_resolution import PromptResolver


def test_prompt_resolver_leaves_plain_prompt_unchanged() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve("create a migration plan for lumen", active_thread=None)

    assert resolution.original_prompt == "create a migration plan for lumen"
    assert resolution.resolved_prompt == "create a migration plan for lumen"
    assert resolution.changed is False
    assert resolution.strategy == "none"
    assert resolution.reason == "No prompt rewrite applied"


def test_prompt_resolver_expands_compare_shorthand() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "now compare that",
        active_thread={
            "thread_summary": "Planning response for: create a migration plan for lumen",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "compare the migration plan for lumen"
    assert resolution.strategy == "compare_shorthand"
    assert "comparison shorthand" in resolution.reason


def test_prompt_resolver_expands_anh_shorthand_for_ga_context() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "do that with anh",
        active_thread={
            "prompt": "run anh",
            "objective": "Execute tool task: run anh",
            "thread_summary": "GA Local Analysis Kit run completed",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "run anh"
    assert resolution.strategy == "anh_tool_shorthand"
    assert "GA tool command" in resolution.reason


def test_prompt_resolver_expands_reference_follow_up() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "what about that",
        active_thread={
            "objective": "Plan work for: create a migration plan for lumen",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "what about the migration plan for lumen"
    assert resolution.strategy == "reference_follow_up"
    assert "reference-style follow-up" in resolution.reason


def test_prompt_resolver_expands_addressed_reference_follow_up() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "Hey Lumen, what about that?",
        active_thread={
            "objective": "Plan work for: create a migration plan for lumen",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "what about the migration plan for lumen"
    assert resolution.strategy == "reference_follow_up"


def test_prompt_resolver_expands_thread_follow_up() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "expand that further",
        active_thread={
            "thread_summary": "create a migration plan for lumen",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "expand the migration plan for lumen"
    assert resolution.strategy == "thread_follow_up"
    assert "active thread subject" in resolution.reason


def test_prompt_resolver_reuses_active_tool_prompt_for_repeat_shorthand() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "run that again",
        active_thread={
            "mode": "tool",
            "prompt": "run anh",
            "thread_summary": "GA Local Analysis Kit run completed",
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "run anh"
    assert resolution.strategy == "tool_repeat_shorthand"
    assert "repeat-style follow-up" in resolution.reason


def test_prompt_resolver_recovers_from_stale_shorthand_tool_prompt() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "run that again",
        active_thread={
            "mode": "tool",
            "prompt": "run that again",
            "tool_context": {
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
            },
        },
    )

    assert resolution.changed is True
    assert resolution.resolved_prompt == "run anh"
    assert resolution.strategy == "tool_repeat_shorthand"


def test_prompt_resolver_expands_research_follow_ups_naturally() -> None:
    resolver = PromptResolver()

    compare_resolution = resolver.resolve(
        "compare that",
        active_thread={
            "objective": "Research topic: summarize the current archive structure",
        },
    )
    reference_resolution = resolver.resolve(
        "what about that",
        active_thread={
            "objective": "Research topic: summarize the current archive structure",
        },
    )

    assert compare_resolution.resolved_prompt == "compare the current archive structure"
    assert reference_resolution.resolved_prompt == "what about the current archive structure"


def test_prompt_resolver_ignores_modern_user_facing_summary_as_subject_fallback() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "expand that further",
        active_thread={
            "thread_summary": "Here's a workable first pass.",
        },
    )

    assert resolution.changed is False
    assert resolution.resolved_prompt == "expand that further"


def test_prompt_resolver_does_not_expand_follow_up_from_conversation_thread() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "what about that",
        active_thread={
            "mode": "conversation",
            "kind": "conversation.greeting",
            "prompt": "good to see you too",
            "thread_summary": "Good to see you too.",
        },
    )

    assert resolution.changed is False
    assert resolution.resolved_prompt == "what about that"


def test_prompt_resolver_does_not_expand_compare_shorthand_from_conversation_thread() -> None:
    resolver = PromptResolver()

    resolution = resolver.resolve(
        "compare that",
        active_thread={
            "mode": "conversation",
            "kind": "conversation.check_in",
            "prompt": "how are you",
            "thread_summary": "I'm doing well.",
        },
    )

    assert resolution.changed is False
    assert resolution.resolved_prompt == "compare that"

