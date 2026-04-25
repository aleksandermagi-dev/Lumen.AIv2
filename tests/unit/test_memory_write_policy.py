from lumen.memory.memory_models import MemoryClassification
from lumen.memory.write_policy import MemoryWritePolicy


def test_memory_write_policy_allows_research_note_on_main_surface() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide(
        classification=MemoryClassification.research_candidate(
            confidence=0.8,
            reason="Research-oriented interaction.",
        ),
        client_surface="main",
        mobile_research_note_auto_save=False,
    )

    assert decision.action == "save_research_note"
    assert decision.save_research_note is True
    assert decision.save_personal_memory is False


def test_memory_write_policy_blocks_research_note_on_mobile_by_default() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide(
        classification=MemoryClassification.research_candidate(
            confidence=0.8,
            reason="Research-oriented interaction.",
        ),
        client_surface="mobile",
        mobile_research_note_auto_save=False,
    )

    assert decision.action == "skip"
    assert decision.blocked_by_surface_policy is True


def test_memory_write_policy_only_allows_personal_save_when_explicit() -> None:
    policy = MemoryWritePolicy()

    explicit_decision = policy.decide(
        classification=MemoryClassification.personal_candidate(
            confidence=0.9,
            reason="Personal context.",
            explicit_save_requested=True,
        ),
        client_surface="main",
        mobile_research_note_auto_save=False,
    )
    implicit_decision = policy.decide(
        classification=MemoryClassification.personal_candidate(
            confidence=0.9,
            reason="Personal context.",
            explicit_save_requested=False,
        ),
        client_surface="main",
        mobile_research_note_auto_save=False,
    )

    assert explicit_decision.action == "save_personal_memory"
    assert implicit_decision.action == "skip"
