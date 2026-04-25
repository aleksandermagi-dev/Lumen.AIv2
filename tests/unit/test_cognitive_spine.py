from lumen.reasoning.confidence_gradient import ConfidenceGradient
from lumen.reasoning.memory_context_classifier import MemoryContextClassifier
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult, RetrievedMemory
from lumen.reasoning.reasoning_state import CognitiveStateFrame
from lumen.reasoning.tool_threshold_gate import ToolThresholdGate
from lumen.services.reasoning_state_service import ReasoningStateService


def test_cognitive_state_frame_round_trip_preserves_phase1_fields() -> None:
    frame = CognitiveStateFrame(
        route_decision={"mode": "planning", "kind": "planning.roadmap"},
        intent_domain="planning_strategy",
        response_depth="deep",
        conversation_phase="execution",
        confidence=0.84,
        confidence_tier="high",
        memory_context_used=[{"source": "saved_project", "label": "Lumen roadmap"}],
        tool_usage_intent={"tool_id": "design", "capability": "system_spec"},
        tool_decision={"should_use_tool": True, "selected_tool": "design"},
        response_style={"interaction_style": "default", "structure": "roadmap"},
        uncertainty_flags=["needs_validation"],
        failure_flags=[],
        rationale_summary="Route, memory, and tool posture are aligned.",
    )

    payload = frame.to_dict()
    restored = CognitiveStateFrame.from_mapping(payload)

    assert restored.route_decision["mode"] == "planning"
    assert restored.intent_domain == "planning_strategy"
    assert restored.response_depth == "deep"
    assert restored.conversation_phase == "execution"
    assert restored.confidence_tier == "high"
    assert restored.memory_context_used[0]["label"] == "Lumen roadmap"
    assert restored.tool_usage_intent["tool_id"] == "design"
    assert restored.tool_decision["selected_tool"] == "design"
    assert restored.response_style["structure"] == "roadmap"
    assert restored.rationale_summary == "Route, memory, and tool posture are aligned."


def test_tool_threshold_gate_declines_unnecessary_tool_use() -> None:
    decision = ToolThresholdGate().decide(
        prompt="maybe inspect this later",
        route_mode="tool",
        route_kind="tool.inspect",
        route_confidence=0.41,
        tool_id="workspace",
        capability="inspect.structure",
        input_path=None,
        params={},
    )

    assert decision.should_use_tool is False
    assert decision.internal_reasoning_sufficient is True
    assert decision.selected_tool == "workspace"


def test_tool_threshold_gate_allows_structured_tool_execution() -> None:
    decision = ToolThresholdGate().decide(
        prompt="run anh on this file",
        route_mode="tool",
        route_kind="tool.analysis",
        route_confidence=0.91,
        tool_id="anh",
        capability="spectral_dip_scan",
        input_path="sample.csv",
        params={"window": 5},
    )

    assert decision.should_use_tool is True
    assert decision.tool_necessary is True
    assert decision.tool_higher_confidence is True
    assert decision.material_outcome_improvement is True


def test_memory_context_classifier_rejects_irrelevant_personal_memory_for_technical_turn() -> None:
    retrieval = MemoryRetrievalResult(
        query="debug the workspace ingestion failure",
        selected=[
            RetrievedMemory(
                source="personal_memory",
                memory_kind="profile",
                label="Music preference",
                summary="The user likes ambient synth playlists while working.",
                relevance=0.18,
                metadata={"category": "preferences", "confidence": 0.4},
            ),
            RetrievedMemory(
                source="bug_log",
                memory_kind="bug_log",
                label="Workspace parser issue",
                summary="A previous bug caused workspace ingestion to fail after schema drift.",
                relevance=0.94,
                metadata={"category": "debug", "confidence": 0.88, "recency_days": 3},
            ),
        ],
    )

    decision = MemoryContextClassifier().classify(
        retrieval=retrieval,
        route_mode="planning",
        intent_domain="technical_engineering",
        prompt="debug the workspace ingestion failure",
    )

    assert len(decision.selected) == 1
    assert decision.selected[0].label == "Workspace parser issue"
    assert len(decision.rejected) == 1
    assert decision.rejected[0].label == "Music preference"


def test_memory_context_classifier_rejects_stale_generic_memory_for_technical_turn() -> None:
    retrieval = MemoryRetrievalResult(
        query="debug the workspace ingestion failure",
        selected=[
            RetrievedMemory(
                source="research_notes",
                memory_kind="durable_project_memory",
                label="first pass summary",
                summary="Older workspace notes from the first pass without the current schema details.",
                relevance=0.81,
                metadata={
                    "age_bucket": "old",
                    "reaffirmed": False,
                    "generic_label_penalty": 0.08,
                    "source_reliability": 0.84,
                    "confidence_hint": 0.78,
                },
            ),
            RetrievedMemory(
                source="research_notes",
                memory_kind="durable_project_memory",
                label="workspace parser issue",
                summary="The current schema drift breaks ingestion in the workspace parser.",
                relevance=0.93,
                metadata={
                    "age_bucket": "recent",
                    "reaffirmed": True,
                    "source_reliability": 0.84,
                    "confidence_hint": 0.78,
                },
            ),
        ],
    )

    decision = MemoryContextClassifier().classify(
        retrieval=retrieval,
        route_mode="planning",
        intent_domain="technical_engineering",
        prompt="debug the workspace ingestion failure",
    )

    assert len(decision.selected) == 1
    assert decision.selected[0].label == "workspace parser issue"
    assert any(item.label == "first pass summary" for item in decision.rejected)


def test_memory_context_classifier_blocks_personal_memory_noise_in_casual_chat() -> None:
    retrieval = MemoryRetrievalResult(
        query="that's interesting",
        selected=[
            RetrievedMemory(
                source="personal_memory",
                memory_kind="durable_user_memory",
                label="Tea preference",
                summary="The user likes green tea in the afternoon.",
                relevance=0.74,
                metadata={"age_bucket": "recent", "source_reliability": 0.92, "focus_overlap": 0},
            )
        ],
    )

    decision = MemoryContextClassifier().classify(
        retrieval=retrieval,
        route_mode="conversation",
        intent_domain="conversational",
        prompt="that's interesting",
    )

    assert not decision.selected
    assert decision.rejected[0].metadata["memory_exposure_policy"] == "blocked_noise"


def test_memory_context_classifier_allows_bounded_personal_familiarity_when_relevant() -> None:
    retrieval = MemoryRetrievalResult(
        query="can you keep it brief for me",
        selected=[
            RetrievedMemory(
                source="personal_memory",
                memory_kind="durable_user_memory",
                label="Brief style preference",
                summary="The user prefers brief answers.",
                relevance=0.82,
                metadata={"age_bucket": "recent", "source_reliability": 0.92, "focus_overlap": 2},
            ),
            RetrievedMemory(
                source="personal_memory",
                memory_kind="durable_user_memory",
                label="Playlist preference",
                summary="The user likes ambient music.",
                relevance=0.79,
                metadata={"age_bucket": "recent", "source_reliability": 0.92, "focus_overlap": 1},
            ),
        ],
    )

    decision = MemoryContextClassifier().classify(
        retrieval=retrieval,
        route_mode="conversation",
        intent_domain="conversational",
        prompt="can you keep it brief for me",
    )

    assert len(decision.selected) == 1
    assert decision.selected[0].label == "Brief style preference"
    assert decision.selected[0].metadata["memory_exposure_policy"] == "bounded_familiarity"
    assert any(item.metadata["memory_exposure_policy"] == "capped_familiarity" for item in decision.rejected)


def test_memory_context_classifier_keeps_explicit_memory_recall_authoritative() -> None:
    retrieval = MemoryRetrievalResult(
        query="what do you remember about my preferences",
        selected=[
            RetrievedMemory(
                source="personal_memory",
                memory_kind="durable_user_memory",
                label="Brief style preference",
                summary="The user prefers brief answers.",
                relevance=0.62,
                metadata={"age_bucket": "recent", "source_reliability": 0.92, "focus_overlap": 1},
            )
        ],
        recall_prompt=True,
    )

    decision = MemoryContextClassifier().classify(
        retrieval=retrieval,
        route_mode="conversation",
        intent_domain="conversational",
        prompt="what do you remember about my preferences",
    )

    assert len(decision.selected) == 1
    assert decision.selected[0].metadata["memory_exposure_policy"] == "explicit_recall"


def test_reasoning_state_service_tool_verification_raises_confidence_tier() -> None:
    service = ReasoningStateService()
    state = CognitiveStateFrame(confidence=0.44, confidence_tier="low")
    state = service.apply_tool_decision(
        state=state,
        decision=ToolThresholdGate().decide(
            prompt="run anh on the spectrum file",
            route_mode="tool",
            route_kind="tool.analysis",
            route_confidence=0.9,
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path="sample.csv",
            params={"window": 8},
        ),
    )
    outcome = service.classify_execution_outcome(
        tool_result=type(
            "ToolResult",
            (),
            {
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
                "status": "ok",
                "summary": "Tool verification complete.",
                "structured_data": {},
            },
        )()
    )

    updated = service.apply_execution_outcome(state=state, outcome=outcome)

    assert updated.confidence > state.confidence
    assert updated.confidence_tier in {"medium", "high"}
    assert ConfidenceGradient.tier_for_score(updated.confidence) == updated.confidence_tier
