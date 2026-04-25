from lumen.memory.classification import MemoryClassifier


def test_memory_classifier_marks_research_interaction_as_save_eligible() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="create a migration plan for lumen routing",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: create a migration plan for lumen routing",
    )

    assert result.candidate_type == "research_memory_candidate"
    assert result.save_eligible is True
    assert result.requires_explicit_user_consent is False


def test_memory_classifier_requires_explicit_consent_for_personal_context() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="remember this about me: I feel anxious about my health",
        resolved_prompt=None,
        mode="research",
        dominant_intent="research",
        summary="Tentative research response for: remember this about me: I feel anxious about my health",
    )

    assert result.candidate_type == "personal_context_candidate"
    assert result.save_eligible is False
    assert result.requires_explicit_user_consent is True
    assert result.explicit_save_requested is True


def test_memory_classifier_defaults_uncertain_cases_to_ephemeral() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="hello there",
        resolved_prompt=None,
        mode="research",
        dominant_intent="unknown",
        summary="Tentative research response for: hello there",
    )

    assert result.candidate_type == "ephemeral_conversation_context"
    assert result.save_eligible is False
    assert result.requires_explicit_user_consent is False


def test_memory_classifier_keeps_non_explicit_personal_context_unsaved() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="I feel anxious about my health lately",
        resolved_prompt=None,
        mode="research",
        dominant_intent="research",
        summary="Tentative research response for: I feel anxious about my health lately",
    )

    assert result.candidate_type == "personal_context_candidate"
    assert result.save_eligible is False
    assert result.requires_explicit_user_consent is True
    assert result.explicit_save_requested is False


def test_memory_classifier_requires_stronger_project_signal_before_auto_save() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="can you help me think about this",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: can you help me think about this",
    )

    assert result.candidate_type == "ephemeral_conversation_context"
    assert result.save_eligible is False


def test_memory_classifier_keeps_mixed_personal_technical_content_out_of_auto_save() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="I feel anxious about my health and the migration plan",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: I feel anxious about my health and the migration plan",
    )

    assert result.candidate_type == "personal_context_candidate"
    assert result.save_eligible is False
    assert result.requires_explicit_user_consent is True


def test_memory_classifier_keeps_project_thread_with_help_language_save_eligible() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="can you help me think through the migration plan for lumen routing",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: can you help me think through the migration plan for lumen routing",
    )

    assert result.candidate_type == "research_memory_candidate"
    assert result.save_eligible is True


def test_memory_classifier_treats_explicit_preference_memory_as_personal() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="remember that I prefer direct responses",
        resolved_prompt=None,
        mode="conversation",
        dominant_intent="unknown",
        summary="Conversation response for: remember that I prefer direct responses",
    )

    assert result.candidate_type == "personal_context_candidate"
    assert result.save_eligible is False
    assert result.requires_explicit_user_consent is True
    assert result.explicit_save_requested is True


def test_memory_classifier_does_not_let_summary_only_project_language_force_auto_save() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="thanks, that helped",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="unknown",
        summary="Grounded planning response for: create a migration plan for lumen routing",
    )

    assert result.candidate_type == "ephemeral_conversation_context"
    assert result.save_eligible is False


def test_memory_classifier_keeps_lightweight_thread_continuation_unsaved() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="keep going",
        resolved_prompt=None,
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: create a migration plan for lumen routing",
    )

    assert result.candidate_type == "ephemeral_conversation_context"
    assert result.save_eligible is False


def test_memory_classifier_allows_explicit_expanded_follow_up_thread_action() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="expand that further",
        resolved_prompt="expand the migration plan for lumen",
        mode="planning",
        dominant_intent="planning",
        summary="Grounded planning response for: expand the migration plan for lumen",
    )

    assert result.candidate_type == "research_memory_candidate"
    assert result.save_eligible is True


def test_memory_classifier_treats_durable_conversational_preference_as_personal_memory() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="from now on keep it brief with me",
        resolved_prompt=None,
        mode="conversation",
        dominant_intent="unknown",
        summary="Conversation response for: from now on keep it brief with me",
    )

    assert result.candidate_type == "personal_context_candidate"
    assert result.explicit_save_requested is True
    assert result.requires_explicit_user_consent is True


def test_memory_classifier_keeps_one_off_conversational_preference_ephemeral() -> None:
    classifier = MemoryClassifier()

    result = classifier.classify(
        prompt="be more direct on this one",
        resolved_prompt=None,
        mode="conversation",
        dominant_intent="unknown",
        summary="Conversation response for: be more direct on this one",
    )

    assert result.candidate_type == "ephemeral_conversation_context"
    assert result.save_eligible is False
