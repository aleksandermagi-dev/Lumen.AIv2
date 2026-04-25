from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.reasoning.explanation_response_builder import ExplanationResponseBuilder


def test_explanation_response_builder_consults_local_knowledge_for_research_summary() -> None:
    assert ExplanationResponseBuilder.should_consult_local_knowledge(
        prompt="what do watts mean",
        route_mode="research",
        route_kind="research.summary",
        entities=(),
    ) is True


def test_explanation_response_builder_consults_local_knowledge_for_research_comparison() -> None:
    assert ExplanationResponseBuilder.should_consult_local_knowledge(
        prompt="ohms vs watts",
        route_mode="research",
        route_kind="research.comparison",
        entities=(),
    ) is True


def test_explanation_response_builder_consults_local_knowledge_for_research_general() -> None:
    assert ExplanationResponseBuilder.should_consult_local_knowledge(
        prompt="voltage",
        route_mode="research",
        route_kind="research.general",
        entities=(),
    ) is True


def test_explanation_response_builder_does_not_consult_local_knowledge_for_non_research_route() -> None:
    assert ExplanationResponseBuilder.should_consult_local_knowledge(
        prompt="what do watts mean",
        route_mode="conversation",
        route_kind="conversation.answer",
        entities=(),
    ) is False


def test_explanation_response_builder_returns_structured_lookup_result() -> None:
    result = ExplanationResponseBuilder.build_answer(
        prompt="what do watts mean",
        interaction_style="default",
        knowledge_service=KnowledgeService.in_memory(),
    )

    assert result.source == "lookup"
    assert result.lookup_succeeded is True
    assert "power" in result.answer.lower()


def test_explanation_response_builder_collab_lookup_does_not_prefix_repeated_sure() -> None:
    result = ExplanationResponseBuilder.build_answer(
        prompt="what is a galaxy",
        interaction_style="collab",
        knowledge_service=KnowledgeService.in_memory(),
    )

    assert result.source == "lookup"
    assert not result.answer.lower().startswith("sure.")
    assert "galaxy" in result.answer.lower()


def test_explanation_response_builder_fallback_explains_why_without_could_tell_wrapper() -> None:
    result = ExplanationResponseBuilder.build_answer(
        prompt="what is the flarnoptic moon gate",
        interaction_style="collab",
        knowledge_service=KnowledgeService.in_memory(),
    )

    text = result.answer.lower()
    assert result.source in {"fallback", "generic"}
    assert "because" in text
    assert "could explain" not in text
