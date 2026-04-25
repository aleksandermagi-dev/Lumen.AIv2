from lumen.routing.intent_signals import IntentSignalExtractor


def test_intent_signal_extractor_marks_explicit_planning_and_scores() -> None:
    signals = IntentSignalExtractor().extract("create a migration plan for lumen")

    assert signals.normalized_prompt == "create a migration plan for lumen"
    assert signals.explicit_planning_kind == "planning.migration"
    assert signals.planning_score > 0
    assert "migration" in signals.migration_hints


def test_intent_signal_extractor_marks_explicit_summary_and_answer_bias() -> None:
    signals = IntentSignalExtractor().extract("summarize the current archive structure")

    assert signals.explicit_summary is True
    assert signals.answer_score > 0
    assert signals.research_score > 0


def test_intent_signal_extractor_marks_follow_up_and_comparison_hints() -> None:
    signals = IntentSignalExtractor().extract("now compare that")

    assert signals.follow_up is True
    assert "compare" in signals.comparison_hints


def test_intent_signal_extractor_handles_messy_social_prompt() -> None:
    signals = IntentSignalExtractor().extract("tysm lumin")

    assert signals.normalized_prompt == "thank you so much lumen"
    assert signals.explicit_social_kind == "conversation.gratitude"


def test_intent_signal_extractor_handles_messy_summary_prompt() -> None:
    signals = IntentSignalExtractor().extract("plz summarize whats going on w the archive rn")

    assert signals.explicit_summary is True
    assert signals.research_score > 0


def test_intent_signal_extractor_handles_soft_prefixed_planning_prompt() -> None:
    signals = IntentSignalExtractor().extract("umm can ya real quick create a migration plan for lumen")

    assert signals.explicit_planning_kind == "planning.migration"
    assert signals.planning_score > 0


def test_intent_signal_extractor_handles_soft_prefixed_comparison_prompt() -> None:
    signals = IntentSignalExtractor().extract("yo could you compare ga vs shapley")

    assert signals.normalized_prompt == "compare ga vs shapley"
    assert signals.explicit_comparison is True
    assert "vs" in signals.comparison_hints


def test_intent_signal_extractor_marks_praise_as_social() -> None:
    signals = IntentSignalExtractor().extract("great job")

    assert signals.explicit_social_kind == "conversation.affirmation"


def test_intent_signal_extractor_marks_relational_greeting_as_social() -> None:
    signals = IntentSignalExtractor().extract("good to see you too")

    assert signals.explicit_social_kind == "conversation.greeting"


def test_intent_signal_extractor_handles_social_inventory_phrases_consistently() -> None:
    gratitude = IntentSignalExtractor().extract("yo lumen thanks")
    check_in = IntentSignalExtractor().extract("what is up")
    farewell = IntentSignalExtractor().extract("talk soon")

    assert gratitude.explicit_social_kind == "conversation.gratitude"
    assert check_in.explicit_social_kind == "conversation.check_in"
    assert farewell.explicit_social_kind == "conversation.farewell"


def test_intent_signal_extractor_uses_shared_follow_up_inventory() -> None:
    signals = IntentSignalExtractor().extract("continue with that")

    assert signals.follow_up is True


def test_intent_signal_extractor_exposes_anchor_resolution_metadata() -> None:
    signals = IntentSignalExtractor().extract("what is gravity")

    assert signals.anchor_resolution["primary_domain"] == "physics"
    assert signals.anchor_resolution["primary_action"] in {"define", "explain"}
    assert signals.anchor_resolution["capability_hint"] == "knowledge"


def test_intent_signal_extractor_boosts_named_subject_explanation() -> None:
    signals = IntentSignalExtractor().extract("George Washington")

    assert signals.answer_score >= 2
    assert signals.research_score >= 2
    assert any(entity["label"] == "person" for entity in signals.extracted_entities)


def test_intent_signal_extractor_uses_starter_prompt_word_clusters() -> None:
    signals = IntentSignalExtractor().extract("help me think through this problem under these constraints")

    assert "reasoning" in signals.starter_categories
    assert signals.starter_category_scores["reasoning"] >= 2
    assert signals.action_score >= 2


def test_intent_signal_extractor_uses_soft_entry_cluster() -> None:
    signals = IntentSignalExtractor().extract("i have an idea but i'm not sure if it makes sense yet")

    assert "exploration" in signals.starter_categories
    assert signals.research_score >= 2
