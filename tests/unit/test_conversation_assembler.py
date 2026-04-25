from lumen.reasoning.conversation_assembler import ConversationAssembler


def test_conversation_assembler_can_compose_modular_fragments() -> None:
    reply = ConversationAssembler.assemble(
        style="default",
        seed_parts=["greeting"],
        opener=("Hey.",),
        content=("What are we working on?",),
    )

    assert reply == "Hey. What are we working on?"


def test_conversation_assembler_handles_optional_parts_without_overconstruction() -> None:
    reply = ConversationAssembler.assemble(
        style="collab",
        seed_parts=["thought_mode"],
        content=("I've been thinking about how framing changes what you notice.",),
    )

    assert reply == "I've been thinking about how framing changes what you notice."


def test_conversation_assembler_keeps_mode_changes_surface_only() -> None:
    straight = ConversationAssembler.assemble(
        style="direct",
        seed_parts=["farewell"],
        content=("Later.",),
    )
    collab = ConversationAssembler.assemble(
        style="collab",
        seed_parts=["farewell"],
        opener=("Alright,",),
        content=("talk soon.",),
        closer=("I'll be here.",),
    )

    assert straight == "Later."
    assert collab == "Alright, talk soon. I'll be here."


def test_conversation_assembler_does_not_inject_internal_scaffold_language() -> None:
    reply = ConversationAssembler.assemble(
        style="default",
        seed_parts=["uncertainty"],
        content=("I don't have enough grounded detail to answer that confidently yet.",),
    )

    assert "routing" not in reply.lower()
    assert "working hypothesis" not in reply.lower()
    assert "checkpoint" not in reply.lower()

