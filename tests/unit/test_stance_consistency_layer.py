from types import SimpleNamespace

from lumen.app.models import InteractionProfile
from lumen.reasoning.stance_consistency_layer import StanceConsistencyLayer


def test_stance_consistency_supports_full_agreement_without_duplication() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="you're right about that",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile.default(),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="agreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="exploratory"),
        recent_interactions=[],
    )

    assert result.category == "full_agreement"
    assert "Here's my read so far." in str(result.applied_lead)
    assert str(result.applied_lead).count("That tracks.") <= 1


def test_stance_consistency_uses_partial_agreement_for_softer_alignment() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="that makes sense to a point",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile.default(),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="agreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="exploratory"),
        recent_interactions=[],
    )

    assert result.category == "partial_agreement"
    assert "part" in str(result.applied_lead).lower() or "truth in that" in str(result.applied_lead).lower()


def test_stance_consistency_qualifies_agreement_when_support_is_weak() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="you're right",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile.default(),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="agreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="exploratory"),
        recent_interactions=[],
        support_status="insufficiently_grounded",
    )

    assert result.category == "agreement_with_qualification"
    assert "qualify" in str(result.applied_lead).lower()


def test_stance_consistency_surfaces_respectful_disagreement() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="i disagree with that part",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile.default(),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="disagreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="assertive"),
        recent_interactions=[],
    )

    assert result.category == "respectful_disagreement"
    assert (
        "not fully convinced" in str(result.applied_lead).lower()
        or "frame that part differently" in str(result.applied_lead).lower()
        or "push back" in str(result.applied_lead).lower()
    )


def test_stance_consistency_softens_reversal_against_recent_opposite_stance() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="i disagree with that part",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile.default(),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="disagreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="assertive"),
        recent_interactions=[
            {
                "response": {
                    "stance_consistency": {
                        "category": "full_agreement",
                    }
                }
            }
        ],
    )

    assert result.contradiction_aware is True
    assert result.previous_category == "full_agreement"
    assert "On this part" in str(result.applied_lead)


def test_stance_consistency_keeps_mode_expression_distinct() -> None:
    straight = StanceConsistencyLayer.assess(
        prompt="i disagree with that part",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile(
            interaction_style="direct",
            reasoning_depth="normal",
            selection_source="user",
        ),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="disagreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="assertive"),
        recent_interactions=[],
    )
    collab = StanceConsistencyLayer.assess(
        prompt="i disagree with that part",
        base_lead="That tracks. Here's my read so far.",
        interaction_profile=InteractionProfile(
            interaction_style="collab",
            reasoning_depth="normal",
            selection_source="user",
        ),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="disagreeing"),
        human_language_layer=SimpleNamespace(epistemic_stance="assertive"),
        recent_interactions=[],
    )

    assert straight.category == collab.category == "respectful_disagreement"
    assert straight.applied_lead != collab.applied_lead
    assert (
        "I can see the line you're drawing" in str(collab.applied_lead)
        or "I get what you're reaching for" in str(collab.applied_lead)
        or "I can see part of it" in str(collab.applied_lead)
    )


def test_stance_consistency_uses_livelier_collab_uncertainty_surface() -> None:
    result = StanceConsistencyLayer.assess(
        prompt="maybe",
        base_lead="Here's my read so far.",
        interaction_profile=InteractionProfile(
            interaction_style="collab",
            reasoning_depth="normal",
            selection_source="user",
        ),
        conversation_awareness=SimpleNamespace(recent_intent_pattern="hesitating"),
        human_language_layer=SimpleNamespace(epistemic_stance="unsure"),
        recent_interactions=[],
    )

    assert result.category == "uncertainty"
    assert any(
        cue in str(result.applied_lead)
        for cue in {
            "Could be.",
            "Maybe, yeah.",
            "I'd hold that a little lightly, though.",
            "Mm, maybe.",
            "Could be, honestly.",
        }
    )

