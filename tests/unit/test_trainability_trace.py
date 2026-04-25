from lumen.reasoning.trainability_trace import TrainabilityTrace


def test_trainability_trace_builds_from_cognitive_response_surface() -> None:
    response = {
        "mode": "planning",
        "kind": "planning.debug",
        "intent_domain": "technical_engineering",
        "intent_domain_confidence": 0.88,
        "response_depth": "standard",
        "conversation_phase": "execution",
        "route_status": "grounded",
        "support_status": "supported",
        "reasoning_state": {
            "confidence": 0.81,
            "confidence_tier": "high",
            "intent_domain": "technical_engineering",
            "response_depth": "standard",
            "conversation_phase": "execution",
            "route_decision": {"mode": "planning", "kind": "planning.debug", "confidence": 0.84},
            "memory_context_used": [{"label": "Parser regression", "source": "bug_log"}],
            "tool_usage_intent": {"tool_id": "workspace", "capability": "inspect.structure"},
            "tool_decision": {
                "should_use_tool": True,
                "selected_tool": "workspace",
                "expected_confidence_gain": 0.2,
            },
            "response_style": {
                "interaction_style": "collab",
                "intent_domain": "technical_engineering",
                "response_depth": "standard",
                "conversation_phase": "execution",
                "structure": "systematic",
                "next_steps_enabled": True,
                "tool_suggestions_enabled": False,
            },
            "rationale_summary": "Routing and evidence aligned around a parser-focused debug turn.",
        },
        "tool_threshold_decision": {
            "should_use_tool": True,
            "selected_tool": "workspace",
            "expected_confidence_gain": 0.2,
            "rationale": "Tool inspection will materially improve confidence.",
        },
        "execution_outcome": {
            "execution_attempted": True,
            "execution_status": "ok",
        },
        "memory_retrieval": {
            "diagnostics": {
                "memory_context_classifier": {
                    "selected": [{"label": "Parser regression"}],
                    "rejected": [{"label": "Favorite tea"}],
                }
            }
        },
        "supervised_support_trace": {
            "schema_version": "1",
            "enabled": True,
            "surfaces_with_examples": [
                "intent_domain_classification",
                "tool_use_decision_support",
            ],
            "recommendations": {
                "intent_domain_classification": {
                    "surface": "intent_domain_classification",
                    "recommended_label": "technical_engineering",
                    "confidence": 0.91,
                    "applied": False,
                    "applied_reason": "Deterministic domain already matched.",
                }
            },
            "applied_surfaces": [],
            "deterministic_authority_preserved": True,
        },
    }

    trace = TrainabilityTrace.build(response=response).to_dict()

    assert trace["intent_domain_classification"]["intent_domain"] == "technical_engineering"
    assert trace["route_recommendation_support"]["mode"] == "planning"
    assert trace["memory_relevance_ranking"]["selected_labels"] == ["Parser regression"]
    assert trace["tool_use_decision_support"]["should_use_tool"] is True
    assert trace["confidence_calibration_support"]["tool_verified"] is True
    assert trace["response_style_selection"]["structure"] == "systematic"
    assert trace["supervised_decision_support"]["enabled"] is True
    assert trace["supervised_decision_support"]["recommended_surfaces"] == ["intent_domain_classification"]
