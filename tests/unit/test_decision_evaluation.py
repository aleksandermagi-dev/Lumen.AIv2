from __future__ import annotations

from lumen.evaluation.decision_evaluation import DecisionEvaluation
from lumen.evaluation.evaluation_runner import EvaluationRunner


def _base_record() -> dict[str, object]:
    return {
        "session_id": "default",
        "interaction_path": "data/interactions/default/sample.json",
        "created_at": "2026-04-08T00:00:00+00:00",
        "mode": "planning",
        "kind": "planning.debug",
        "summary": "Let’s isolate the parser regression first.",
        "confidence_posture": "supported",
        "route_status": "grounded",
        "support_status": "supported",
        "response": {
            "reasoning_state": {
                "confidence": 0.84,
                "confidence_tier": "high",
                "intent_domain": "technical_engineering",
                "route_decision": {"mode": "planning", "kind": "planning.debug"},
                "memory_context_used": [{"label": "Parser regression", "source": "bug_log"}],
            }
        },
        "trainability_trace": {
            "route_recommendation_support": {
                "mode": "planning",
                "kind": "planning.debug",
                "route_confidence": 0.84,
                "route_status": "grounded",
                "support_status": "supported",
            },
            "intent_domain_classification": {
                "intent_domain": "technical_engineering",
                "intent_domain_confidence": 0.88,
                "route_mode": "planning",
                "response_depth": "standard",
            },
            "memory_relevance_ranking": {
                "selected_count": 1,
                "rejected_count": 0,
                "selected_labels": ["Parser regression"],
                "rejected_labels": [],
                "memory_context_used_labels": ["Parser regression"],
                "memory_context_used_count": 1,
            },
            "tool_use_decision_support": {
                "should_use_tool": True,
                "selected_tool": "workspace",
                "rationale": "Tool inspection will materially improve confidence.",
                "execution_attempted": True,
                "execution_status": "ok",
            },
            "response_style_selection": {
                "intent_domain": "technical_engineering",
            },
            "confidence_calibration_support": {
                "confidence_tier": "high",
                "confidence_score": 0.84,
                "confidence_posture": "supported",
                "support_status": "supported",
                "route_status": "grounded",
                "memory_signal_present": True,
                "tool_verified": True,
            },
            "supervised_decision_support": {
                "enabled": True,
                "recommended_surfaces": ["intent_domain_classification"],
                "applied_surfaces": [],
                "deterministic_authority_preserved": True,
            },
        },
        "supervised_support_trace": {
            "enabled": True,
            "recommendations": {
                "intent_domain_classification": {
                    "surface": "intent_domain_classification",
                    "recommended_label": "technical_engineering",
                    "confidence": 0.9,
                    "applied": False,
                }
            },
            "applied_surfaces": [],
            "deterministic_authority_preserved": True,
        },
    }


def test_decision_evaluation_scores_aligned_interaction_as_mostly_correct() -> None:
    evaluator = DecisionEvaluation()

    evaluation = evaluator.evaluate_record(_base_record())

    assert evaluation.overall_judgment == "correct"
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}
    assert judgments["route_quality"] == "correct"
    assert judgments["intent_domain_quality"] == "correct"
    assert judgments["tool_use_justification_quality"] == "correct"
    assert judgments["supervised_support_quality"] == "correct"


def test_decision_evaluation_marks_low_signal_case_as_weak_not_incorrect() -> None:
    evaluator = DecisionEvaluation()
    record = _base_record()
    record["trainability_trace"]["intent_domain_classification"] = {  # type: ignore[index]
        "intent_domain": "technical_engineering",
        "intent_domain_confidence": 0.51,
        "route_mode": "planning",
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["intent_domain_quality"] == "weak"


def test_decision_evaluation_detects_tool_use_mismatch() -> None:
    evaluator = DecisionEvaluation()
    record = _base_record()
    record["trainability_trace"]["tool_use_decision_support"] = {  # type: ignore[index]
        "should_use_tool": False,
        "selected_tool": "workspace",
        "execution_attempted": True,
        "execution_status": "ok",
        "rationale": "Execution already happened.",
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["tool_use_justification_quality"] == "incorrect"


def test_decision_evaluation_detects_noisy_memory_selection() -> None:
    evaluator = DecisionEvaluation()
    record = _base_record()
    record["trainability_trace"]["memory_relevance_ranking"] = {  # type: ignore[index]
        "selected_count": 1,
        "rejected_count": 3,
        "selected_labels": ["Parser regression"],
        "rejected_labels": ["Favorite tea", "Weekend plans", "Movie quote"],
        "memory_context_used_labels": ["Parser regression", "Favorite tea"],
        "memory_context_used_count": 2,
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["memory_relevance_quality"] == "incorrect"


def test_decision_evaluation_confirms_supervised_support_stayed_bounded() -> None:
    evaluator = DecisionEvaluation()

    evaluation = evaluator.evaluate_record(_base_record())
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["supervised_support_quality"] == "correct"


def test_decision_evaluation_scores_bounded_assistant_turn_as_correct() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "interaction_path": "data/interactions/assistant/sample.json",
        "created_at": "2026-04-08T00:00:00+00:00",
        "mode": "conversation",
        "kind": "conversation.check_in",
        "summary": "Hey. I'm here. What are we working on?",
        "response": {
            "assistant_quality_posture": {
                "profile": "normal_chat",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
                "conversation_boundary_signals": [],
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "route_kind": "conversation.check_in",
                "prompt_class": "general",
                "recent_turn_count": 2,
                "memory_item_count": 1,
            },
            "assistant_voice_profile": {
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": True,
                "continuity_mode": "live_project",
                "continuity_source": "active_thread",
                "project_recent_turn_count": 1,
                "secondary_project_memory_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "correct"


def test_decision_evaluation_scores_self_overview_boundary_as_correct() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.self_overview",
        "summary": "I'm Lumen. I try to be clear, grounded, and helpful.",
        "response": {
            "assistant_quality_posture": {
                "profile": "normal_chat",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
                "conversation_boundary_signals": ["self_overview", "research_threshold_blocked"],
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "route_kind": "conversation.self_overview",
                "prompt_class": "self_referential",
                "recent_turn_count": 1,
                "memory_item_count": 0,
            },
            "assistant_voice_profile": {
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": False,
                "continuity_mode": "general_chat",
                "continuity_source": "none",
                "project_recent_turn_count": 0,
                "secondary_project_memory_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "correct"


def test_decision_evaluation_marks_over_researched_self_chat_as_incorrect() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "research",
        "kind": "research.summary",
        "summary": "Tentative research response for: tell me about yourself",
        "response": {
            "assistant_quality_posture": {
                "profile": "normal_chat",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "assistant_context_snapshot": {
                "route_mode": "research",
                "route_kind": "research.summary",
                "prompt_class": "self_referential",
                "recent_turn_count": 0,
                "memory_item_count": 0,
            },
            "assistant_voice_profile": {
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "incorrect"


def test_decision_evaluation_scores_active_work_thread_continuity_as_correct() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.work_thread_next_step",
        "summary": "Next, keep the work anchored on the release checklist.",
        "response": {
            "assistant_quality_posture": {
                "profile": "low_latency_follow_up",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
                "work_thread_continuity_active": True,
                "work_thread_intent": "next_step",
                "work_thread_source": "active_thread",
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "route_kind": "conversation.work_thread_next_step",
                "prompt_class": "general",
                "recent_turn_count": 2,
                "memory_item_count": 0,
            },
            "assistant_voice_profile": {
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": True,
                "continuity_mode": "live_project",
                "continuity_source": "active_thread",
                "project_recent_turn_count": 1,
                "secondary_project_memory_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "correct"


def test_decision_evaluation_rejects_work_thread_continuity_without_active_thread_source() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.work_thread_next_step",
        "summary": "Next, keep going.",
        "response": {
            "assistant_quality_posture": {
                "profile": "low_latency_follow_up",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
                "work_thread_continuity_active": True,
                "work_thread_intent": "next_step",
                "work_thread_source": "recent_project_interactions",
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "recent_turn_count": 1,
                "memory_item_count": 0,
            },
            "assistant_voice_profile": {
                "style_mode": "default",
                "voice_profile": "calm_grounded",
                "tone_signature": "clear_grounded_warm",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": True,
                "continuity_mode": "live_project",
                "continuity_source": "recent_project_interactions",
                "project_recent_turn_count": 1,
                "secondary_project_memory_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "incorrect"


def test_decision_evaluation_rejects_personal_memory_overinjection_in_ordinary_chat() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.follow_up",
        "summary": "That tracks.",
        "response": {
            "assistant_quality_posture": {
                "profile": "normal_chat",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "recent_turn_count": 2,
                "memory_item_count": 2,
            },
            "assistant_voice_profile": {
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": False,
                "continuity_mode": "general_chat",
                "continuity_source": "none",
                "project_recent_turn_count": 0,
                "secondary_project_memory_count": 0,
            },
            "memory_retrieval": {
                "recall_prompt": False,
                "selected": [
                    {"source": "personal_memory", "memory_kind": "durable_user_memory", "label": "Tea"},
                    {"source": "personal_memory", "memory_kind": "durable_user_memory", "label": "Music"},
                ],
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "incorrect"


def test_decision_evaluation_detects_overstuffed_assistant_context() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.follow_up",
        "summary": "Let's keep going.",
        "response": {
            "assistant_quality_posture": {
                "profile": "low_latency_follow_up",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "recent_turn_count": 5,
                "memory_item_count": 4,
            },
            "assistant_voice_profile": {
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": True,
                "continuity_mode": "live_project",
                "continuity_source": "active_thread",
                "project_recent_turn_count": 4,
                "secondary_project_memory_count": 2,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "incorrect"


def test_decision_evaluation_detects_long_chat_repetition_without_cooldown() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.follow_up",
        "summary": "We can keep going if you want.",
        "response": {
            "assistant_quality_posture": {
                "profile": "low_latency_follow_up",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "recent_turn_count": 3,
                "memory_item_count": 0,
            },
            "conversation_beat": {
                "conversation_depth": 7,
                "continuity_state": "continuing",
                "response_repetition_risk": "high",
                "follow_up_offer_allowed": True,
                "long_chat_stamina": {"long_chat": True, "continuation_offer_cooldown": True},
            },
            "assistant_voice_profile": {
                "style_mode": "collab",
                "voice_profile": "warm_partner",
                "tone_signature": "present_warm_partnered",
                "reasoning_depth": "normal",
                "reasoning_depth_separate": True,
            },
            "project_context_snapshot": {
                "project_context_active": False,
                "continuity_mode": "general_chat",
                "continuity_source": "none",
                "project_recent_turn_count": 0,
                "secondary_project_memory_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "incorrect"


def test_decision_evaluation_marks_missing_voice_separation_as_weak() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "assistant",
        "mode": "conversation",
        "kind": "conversation.check_in",
        "summary": "Hey. What are we working on?",
        "response": {
            "assistant_quality_posture": {
                "profile": "normal_chat",
                "direct_answer_first": True,
                "clarification_restraint": True,
                "memory_budget": 2,
            },
            "assistant_context_snapshot": {
                "route_mode": "conversation",
                "recent_turn_count": 1,
                "memory_item_count": 0,
            },
            "provider_inference": {
                "provider_id": "local_reasoning",
                "model": "builtin_assistant",
                "response_path": "general_assistant",
            },
        },
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)
    judgments = {review.surface: review.judgment for review in evaluation.surface_reviews}

    assert judgments["assistant_reply_quality"] == "weak"


def test_evaluation_runner_aggregates_surface_and_overall_metrics() -> None:
    runner = EvaluationRunner()
    weak_record = _base_record()
    weak_record["trainability_trace"]["confidence_calibration_support"] = {  # type: ignore[index]
        "confidence_tier": "medium",
        "confidence_score": 0.55,
        "confidence_posture": "tentative",
        "support_status": "supported",
        "route_status": "grounded",
        "memory_signal_present": False,
        "tool_verified": False,
    }

    batch = runner.evaluate_records([_base_record(), weak_record], session_id="default")

    assert batch.evaluated_count == 2
    assert batch.judgment_counts["correct"] >= 1
    aggregates = {aggregate.surface: aggregate for aggregate in batch.surface_aggregates}
    assert aggregates["route_quality"].reviewed_count == 2
    assert aggregates["route_quality"].judgment_counts["correct"] == 2
    behavior_slices = {item["name"]: item for item in batch.behavior_slices}
    assert "low_confidence_caution" in behavior_slices
    assert "retrieval_reason_stability" in behavior_slices


def test_decision_evaluation_handles_partial_legacy_like_records() -> None:
    evaluator = DecisionEvaluation()
    record = {
        "session_id": "default",
        "mode": "planning",
        "kind": "planning.migration",
        "summary": "Legacy record",
        "response": {},
        "trainability_trace": {},
        "supervised_support_trace": {},
    }

    evaluation = evaluator.evaluate_record(record)

    assert evaluation.overall_judgment in {"insufficient_evidence", "weak"}
    assert len(evaluation.surface_reviews) == 7
