from lumen.nlu.prompt_nlu import PromptNLU
from lumen.retrieval.semantic_matcher import SemanticCandidate, SemanticMatcher


def test_semantic_matcher_scores_topic_entity_and_intent_overlap() -> None:
    matcher = SemanticMatcher()
    query = PromptNLU().analyze("plan the lumen routing migration")

    result = matcher.score(
        query,
        SemanticCandidate(
            prompt="design the routing migration plan for lumen",
            normalized_topic="routing migration plan for lumen",
            dominant_intent="planning",
            extracted_entities=("routing", "migration", "lumen"),
        ),
    )

    assert result.score > 0
    assert result.intent_match is True
    assert "routing" in result.shared_prompt_tokens
    assert "migration" in result.shared_topic_tokens
    assert "routing" in result.shared_entities
    assert "migration" in result.shared_entities


def test_semantic_matcher_ignores_single_weak_prompt_token_overlap() -> None:
    matcher = SemanticMatcher()
    query = PromptNLU().analyze("inspect routing confidence")

    result = matcher.score(
        query,
        SemanticCandidate(
            prompt="routing logs",
            normalized_topic=None,
            dominant_intent=None,
            extracted_entities=(),
        ),
    )

    assert result.score == 0
    assert result.shared_prompt_tokens == ()


def test_semantic_matcher_keeps_topic_backed_overlap() -> None:
    matcher = SemanticMatcher()
    query = PromptNLU().analyze("inspect routing confidence")

    result = matcher.score(
        query,
        SemanticCandidate(
            prompt="routing logs",
            normalized_topic="routing confidence",
            dominant_intent="research",
            extracted_entities=("routing",),
        ),
    )

    assert result.score > 0
    assert "routing" in result.shared_topic_tokens or "confidence" in result.shared_topic_tokens
