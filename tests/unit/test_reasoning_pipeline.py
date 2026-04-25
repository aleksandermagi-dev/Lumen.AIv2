from types import SimpleNamespace

from lumen.app.models import InteractionProfile
from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.reasoning_pipeline import ReasoningPipeline
from lumen.reasoning.low_confidence_recovery import LowConfidenceRecoveryPolicy
from lumen.reasoning.pipeline_models import (
    LowConfidenceRecoveryResult,
    FrontHalfPipelineResult,
    InputIntake,
    LightweightEvidenceModel,
    NLUExtraction,
    ReasoningFrameAssembly,
    RouteDecisionView,
    TensionResolutionResult,
    ValidationContextResult,
)
from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.domain_router import DomainRoute, DomainRouter
from lumen.routing.prompt_resolution import PromptResolver
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def make_pipeline() -> ReasoningPipeline:
    capability_manager = CapabilityManager(
        manifests={
            "anh": BundleManifest(
                id="anh",
                name="Astronomical Node Heuristics",
                version="0.1.0",
                entrypoint="bundle.py",
                capabilities=[
                    CapabilityManifest(
                        id="spectral_dip_scan",
                        adapter="anh_spectral_scan_adapter",
                        app_capability_key="astronomy.anh_spectral_scan",
                        command_aliases=["run anh"],
                    )
                ],
            )
        }
    )
    return ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
    )


def test_reasoning_pipeline_front_half_produces_typed_intake_nlu_and_route_decision() -> None:
    pipeline = make_pipeline()

    result, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
            "latest_posture": "supported",
            "recent_posture_mix": "stable:supported",
        },
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()

    assert payload["intake"]["raw_input"] == "create a migration plan for lumen routing"
    assert payload["intake"]["interaction_profile"]["interaction_style"] == "collab"
    assert payload["intake"]["detected_language"] == "en"
    assert payload["nlu"]["dominant_intent"] == "planning"
    assert payload["nlu"]["topic"] == "migration plan for lumen routing"
    assert payload["nlu"]["confidence_estimate"] > 0.0
    assert payload["vibe_catcher"]["normalized_prompt"] == "create a migration plan for lumen routing"
    assert payload["low_confidence_recovery"]["recovery_mode"] in {"none", "silent_recovery"}
    assert payload["srd_diagnostic"]["stage"] == "baseline"
    assert payload["srd_diagnostic"]["failure_types"] == []
    assert payload["empathy_model"]["response_sensitivity"] == "normal"
    assert payload["human_language_layer"]["flow_style"] == "loose"
    assert payload["human_language_layer"]["epistemic_stance"] == "exploratory"
    assert payload["dialogue_management"]["interaction_mode"] == "analytical"
    assert payload["dialogue_management"]["idea_state"] == "introduced"
    assert payload["dialogue_management"]["response_strategy"] == "answer"
    assert payload["conversation_awareness"]["recent_intent_pattern"] == "building"
    assert payload["conversation_awareness"]["conversation_momentum"] == "building"
    assert payload["conversation_awareness"]["unresolved_thread_open"] is False
    assert payload["conversation_awareness"]["adaptive_posture"] == "push"
    assert payload["state_interpretation"]["trigger"] == "clarity_achieved"
    assert payload["state_control"]["core_state"] == "focus"
    assert payload["state_control"]["anti_spiral_active"] is False
    assert payload["thought_framing"]["response_kind_label"] == "direct_answer"
    assert payload["thought_framing"]["conversation_activity"] == "turning the current idea into a more workable plan"
    assert payload["thought_framing"]["research_questions"] == [
        "What would count as a convincing next step here?",
    ]
    assert payload["intent_domain"]["domain"] == "planning_strategy"
    assert payload["response_depth"]["level"] == "standard"
    assert payload["conversation_phase"]["phase"] == "intake"
    assert payload["route_candidates"]
    assert payload["route_decision"]["selected"]["candidate"]["mode"] == "planning"
    assert payload["route_decision"]["normalized_scores"]
    assert route.mode == "planning"
    assert route.kind == "planning.migration"
    trace = pipeline.create_trace(result)
    trace_payload = trace.to_dict()
    assert trace_payload["intake_frame"]["raw_input"] == "create a migration plan for lumen routing"
    assert trace_payload["dialogue_management"]["interaction_mode"] == "analytical"
    assert trace_payload["state_interpretation"]["trigger"] == "clarity_achieved"
    assert trace_payload["state_control"]["core_state"] == "focus"
    assert trace_payload["human_language_layer"]["flow_style"] == "loose"
    assert trace_payload["thought_framing"]["response_kind_label"] == "direct_answer"
    assert trace_payload["route_decision"]["selected"]["candidate"]["mode"] == "planning"
    assert trace_payload["stage_sequence"] == [
        "input_intake",
        "nlu_extraction",
        "vibe_catcher",
        "route_candidate_generation",
        "dialogue_management",
        "conversation_awareness",
        "low_confidence_recovery",
        "srd_diagnostic",
        "state_interpreter",
        "state_control",
        "empathy_model",
        "human_language_layer",
        "thought_framing",
        "intent_domain_policy",
        "route_scoring_and_comparison",
    ]
    assert trace_payload["stage_contracts"]["input_intake"]["required_inputs"] == [
        "prompt",
        "session_id",
        "interaction_profile",
        "interaction_summary",
        "active_thread",
    ]
    assert trace_payload["stage_contracts"]["route_scoring_and_comparison"]["failure_state"] is None
    assert trace_payload["stage_contracts"]["dialogue_management"]["stage_name"] == "dialogue_management"
    assert trace_payload["stage_contracts"]["conversation_awareness"]["stage_name"] == "conversation_awareness"
    assert trace_payload["stage_contracts"]["vibe_catcher"]["stage_name"] == "vibe_catcher"
    assert trace_payload["stage_contracts"]["low_confidence_recovery"]["stage_name"] == "low_confidence_recovery"
    assert trace_payload["stage_contracts"]["srd_diagnostic"]["stage_name"] == "srd_diagnostic"
    assert trace_payload["stage_contracts"]["state_interpreter"]["stage_name"] == "state_interpreter"
    assert trace_payload["stage_contracts"]["state_control"]["stage_name"] == "state_control"
    assert trace_payload["stage_contracts"]["empathy_model"]["stage_name"] == "empathy_model"
    assert trace_payload["stage_contracts"]["thought_framing"]["stage_name"] == "thought_framing"
    assert trace_payload["stage_contracts"]["intent_domain_policy"]["stage_name"] == "intent_domain_policy"


def test_reasoning_pipeline_infers_learning_domain_and_deep_depth_from_explanatory_prompt() -> None:
    pipeline = make_pipeline()

    result, route = pipeline.run_front_half(
        prompt="teach me step by step how black holes work",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 1, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()
    assert route.mode == "research"
    assert payload["intent_domain"]["domain"] == "learning_teaching"
    assert payload["response_depth"]["level"] == "deep"
    assert payload["conversation_phase"]["phase"] == "intake"


def test_reasoning_pipeline_infers_grounded_emotional_support_domain() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="I'm overwhelmed and stressed and need help thinking clearly",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 1, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()
    assert payload["intent_domain"]["domain"] == "emotional_support_grounded"
    assert payload["conversation_phase"]["phase"] == "reflection"


def test_reasoning_pipeline_marks_exploration_checkpoint_for_long_multi_turn_dialogue() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="what do you think about this migration idea?",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 3,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "research", "summary": "Discussing migration ideas"},
        ],
        active_thread={"mode": "research", "kind": "research.general"},
    )

    payload = result.to_dict()
    assert payload["dialogue_management"]["interaction_mode"] == "hybrid"
    assert payload["dialogue_management"]["idea_state"] == "exploring"
    assert payload["dialogue_management"]["response_strategy"] == "summarize"
    assert payload["conversation_awareness"]["recent_intent_pattern"] == "asking_questions"
    assert payload["conversation_awareness"]["conversation_momentum"] == "building"
    assert payload["conversation_awareness"]["unresolved_thread_open"] is True
    assert payload["conversation_awareness"]["unresolved_thread_reason"] == "The current thread still has live work in it."
    assert payload["conversation_awareness"]["adaptive_posture"] == "push"
    assert payload["dialogue_management"]["synthesis_checkpoint_due"] is True
    assert "synthesis checkpoint" in payload["dialogue_management"]["checkpoint_reason"].lower()
    assert payload["thought_framing"]["response_kind_label"] == "checkpoint_summary"
    assert payload["thought_framing"]["checkpoint_summary"]["current_direction"].startswith(
        "We are currently centered on"
    )
    assert payload["thought_framing"]["checkpoint_summary"]["strongest_point"] == "Discussing migration ideas"
    assert payload["thought_framing"]["checkpoint_summary"]["weakest_point"] == "The central idea is still broad and has not been tightened enough yet."
    assert payload["thought_framing"]["checkpoint_summary"]["open_questions"] == [
        "What assumption are we actually testing here?",
        "Have you considered the strongest alternative explanation?",
    ]
    assert payload["thought_framing"]["checkpoint_summary"]["next_step"] == "Capture the current synthesis, then decide whether to refine or branch."


def test_reasoning_pipeline_marks_checkpoint_early_when_main_thread_pressure_is_high() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="refine this migration idea a bit more",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "planning", "summary": "Migration plan focused on routing cleanup"},
            {"mode": "planning", "summary": "Earlier migration checkpoint"},
        ],
        active_thread={"mode": "planning", "kind": "planning.architecture"},
    )

    payload = result.to_dict()
    assert payload["dialogue_management"]["synthesis_checkpoint_due"] is True
    assert "earlier synthesis checkpoint" in payload["dialogue_management"]["checkpoint_reason"].lower()
    assert payload["thought_framing"]["response_kind_label"] == "checkpoint_summary"


def test_reasoning_pipeline_marks_checkpoint_early_for_branch_pressure() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="what about a plugin route instead",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "planning", "summary": "Migration plan focused on routing cleanup"},
        ],
        active_thread={"mode": "planning", "kind": "planning.architecture"},
    )

    payload = result.to_dict()
    assert payload["dialogue_management"]["synthesis_checkpoint_due"] is True
    assert "branching or challenge pressure" in payload["dialogue_management"]["checkpoint_reason"].lower()


def test_reasoning_pipeline_marks_reorientation_turn_explicitly() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="where are we now?",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
        },
        recent_interactions=[
            {
                "mode": "planning",
                "summary": "Migration plan focused on routing cleanup",
                "response": {
                    "research_questions": ["What assumption should we tighten first?"],
                },
            },
        ],
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "thread_summary": "Migration plan focused on routing cleanup",
        },
    )

    payload = result.to_dict()
    assert payload["dialogue_management"]["interaction_mode"] == "synthesis"
    assert payload["thought_framing"]["response_kind_label"] == "thread_reorientation"
    assert payload["thought_framing"]["conversation_activity"] == "re-orienting the conversation around the live thread"
    assert payload["thought_framing"]["checkpoint_summary"]["current_direction"] == "Migration plan focused on routing cleanup"
    assert payload["thought_framing"]["checkpoint_summary"]["weakest_point"] == (
        "The main unresolved point is still: What assumption should we tighten first?"
    )
    assert payload["thought_framing"]["checkpoint_summary"]["next_step"] == (
        "Re-anchor on the live thread, then either resolve the main open question or tighten the next step."
    )


def test_reasoning_pipeline_tracks_hesitation_as_step_back_local_state() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="maybe this works, but I'm not sure",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 1,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "research", "summary": "Testing an earlier hypothesis"},
        ],
        active_thread={"mode": "research", "kind": "research.general"},
    )

    payload = result.to_dict()
    assert payload["conversation_awareness"]["recent_intent_pattern"] == "hesitating"
    assert payload["conversation_awareness"]["conversation_momentum"] == "doubting"
    assert payload["conversation_awareness"]["unresolved_thread_open"] is True
    assert payload["conversation_awareness"]["adaptive_posture"] == "step_back"
    assert payload["state_interpretation"]["trigger"] == "confusion"
    assert payload["state_interpretation"]["uncertainty_stacking"] is False
    assert payload["state_control"]["core_state"] == "curiosity"
    assert payload["state_control"]["anti_spiral_active"] is True
    assert payload["empathy_model"]["response_sensitivity"] in {"careful", "stabilizing"}
    assert "pause escalation" in payload["state_control"]["anti_spiral_reason"].lower() or "grounded" in payload["state_control"]["anti_spiral_reason"].lower()


def test_reasoning_pipeline_empathy_model_detects_heavy_signal() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="I'm overwhelmed by this migration plan and a bit stressed",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 1, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()
    assert payload["empathy_model"]["emotional_signal_detected"] is True
    assert payload["empathy_model"]["feeling_label"] in {"heavy", "overloaded"}
    assert payload["empathy_model"]["response_sensitivity"] in {"gentle", "stabilizing"}
    assert payload["empathy_model"]["grounded_acknowledgment"] is not None


def test_reasoning_pipeline_tracks_side_branch_and_return_target() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="what about a plugin route instead",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "planning", "summary": "Migration plan focused on routing cleanup"},
        ],
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "thread_summary": "Migration plan focused on routing cleanup",
        },
    )

    payload = result.to_dict()
    assert payload["dialogue_management"]["idea_state"] == "branching"
    assert payload["conversation_awareness"]["branch_state"] == "side_branch_open"
    assert payload["conversation_awareness"]["return_target"] == "Migration plan focused on routing cleanup"
    assert payload["thought_framing"]["branch_return_hint"] == (
        "We can follow this branch, but the main thread to return to is: Migration plan focused on routing cleanup"
    )
    assert payload["thought_framing"]["conversation_activity"] == (
        "testing a side branch while keeping the main thread in view"
    )


def test_reasoning_pipeline_marks_return_to_main_thread_when_user_requests_it() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="go back to the main thread",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 3,
            "clarification_count": 0,
        },
        recent_interactions=[
            {"mode": "planning", "summary": "Plugin route branch discussion"},
        ],
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "thread_summary": "Migration plan focused on routing cleanup",
        },
    )

    payload = result.to_dict()
    assert payload["conversation_awareness"]["branch_state"] == "returning_to_main"
    assert payload["conversation_awareness"]["return_target"] == "Migration plan focused on routing cleanup"
    assert payload["conversation_awareness"]["return_requested"] is True
    assert payload["thought_framing"]["branch_return_hint"] == (
        "We're back on the main thread: Migration plan focused on routing cleanup"
    )
    assert payload["thought_framing"]["conversation_activity"] == (
        "rejoining the main thread and tightening the live line of work"
    )


def test_reasoning_pipeline_carries_live_unresolved_question_across_turns() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="keep going",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={
            "interaction_count": 2,
            "clarification_count": 0,
        },
        recent_interactions=[
            {
                "mode": "planning",
                "summary": "Planning response for routing cleanup",
                "response": {
                    "research_questions": [
                        "What assumption should we tighten first?",
                    ],
                    "conversation_turn": {
                        "next_move": "What assumption should we tighten first?",
                    },
                },
            },
        ],
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "thread_summary": "Migration plan focused on routing cleanup",
        },
    )

    payload = result.to_dict()
    assert payload["conversation_awareness"]["live_unresolved_question"] == "What assumption should we tighten first?"
    assert payload["thought_framing"]["research_questions"][0] == "What assumption should we tighten first?"


def test_reasoning_pipeline_marks_weak_route_when_normalized_score_is_low() -> None:
    pipeline = make_pipeline()

    result, route = pipeline.run_front_half(
        prompt="hello there",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    assert route.mode == "research"
    assert result.route_decision.weak_route is True


def test_reasoning_pipeline_clarification_gate_returns_adaptive_trigger() -> None:
    pipeline = make_pipeline()
    route = DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="sketch the migration summary",
        confidence=0.82,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        evidence=[],
        decision_summary={
            "selected": {},
            "alternatives": [
                {
                    "candidate": {
                        "mode": "research",
                        "kind": "research.summary",
                        "source": "heuristic_research",
                        "confidence": 0.79,
                    }
                }
            ],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates had very similar confidence and closely ranked source priority.",
        },
    )

    decision = pipeline.run_clarification_gate(
        route=route,
        interaction_summary={
            "interaction_count": 5,
            "clarification_count": 2,
            "clarification_drift": "increasing",
            "recent_clarification_mix": "clarification_heavy_mixed",
            "recent_posture_mix": "mixed",
        },
    )

    payload = decision.to_dict()
    assert payload["action"] == "clarify"
    assert payload["trigger"] == "adaptive_threshold"
    assert payload["question_style"] == "adaptive_disambiguation"


def test_reasoning_pipeline_low_confidence_recovery_can_trigger_clarification() -> None:
    pipeline = make_pipeline()
    result, route = pipeline.run_front_half(
        prompt="aintnobodygonnadoitlikethat",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    assert result.vibe_catcher is not None
    assert result.vibe_catcher.low_confidence is True
    assert result.low_confidence_recovery is not None
    assert result.low_confidence_recovery.recovery_mode == "soft_clarify"
    assert result.srd_diagnostic is not None
    assert result.srd_diagnostic.stage == "repair_attempt"
    assert "coherence_failure" in result.srd_diagnostic.failure_types

    decision = pipeline.run_clarification_gate(
        route=route,
        interaction_summary={"clarification_count": 0},
        low_confidence_recovery=result.low_confidence_recovery,
    )

    assert decision.should_clarify is True
    assert decision.trigger == "low_confidence_recovery"
    assert decision.question_style == "targeted_directional_recovery"


def test_reasoning_pipeline_suppresses_repeated_low_confidence_clarification_loops() -> None:
    pipeline = make_pipeline()
    result, route = pipeline.run_front_half(
        prompt="aintnobodygonnadoitlikethat",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 2},
        recent_interactions=[],
        active_thread=None,
    )

    decision = pipeline.run_clarification_gate(
        route=route,
        interaction_summary={
            "clarification_count": 2,
            "clarification_drift": "increasing",
            "recent_clarification_mix": "clarification_heavy_mixed",
        },
        low_confidence_recovery=result.low_confidence_recovery,
    )

    assert decision.should_clarify is False
    assert decision.action == "degrade"
    assert decision.trigger == "clarification_suppressed"
    assert decision.question_style == "narrow_and_proceed"
    assert decision.degradation_mode == "narrow_and_proceed"


def test_reasoning_pipeline_srd_marks_agency_block_for_hard_clarify_case() -> None:
    pipeline = make_pipeline()
    result, _ = pipeline.run_front_half(
        prompt="x",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    policy = LowConfidenceRecoveryPolicy()
    recovery = policy.assess(
        nlu=result.nlu,
        route_decision=result.route_decision,
        vibe_catcher=result.vibe_catcher,
    )
    assert recovery.recovery_mode in {"none", "silent_recovery"}

    forced = pipeline.srd_diagnostic.diagnose(
        dialogue_management=result.dialogue_management,
        conversation_awareness=result.conversation_awareness,
        route_decision=result.route_decision,
        low_confidence_recovery=LowConfidenceRecoveryResult(
            recovery_mode="hard_clarify",
            acknowledge_partial_understanding=True,
            clarifying_question_style="directional_recovery",
            rationale="Intent confidence is too low to continue honestly without clarification.",
        ),
    )
    assert forced.stage == "agency_block"
    assert forced.should_exit_early is True


def test_reasoning_pipeline_human_language_layer_detects_frustration_and_correction() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="wait no thats not what i meant, this didn't work",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 1, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()

    assert payload["human_language_layer"]["correction_detected"] is True
    assert payload["human_language_layer"]["user_energy"] == "frustrated"
    assert payload["human_language_layer"]["emotional_alignment"] == "calm_supportive"


def test_reasoning_pipeline_human_language_layer_distinguishes_epistemic_stances() -> None:
    pipeline = make_pipeline()

    exploratory, _ = pipeline.run_front_half(
        prompt="what if this is happening?",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    unsure, _ = pipeline.run_front_half(
        prompt="I think this is probably what's happening",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    assertive, _ = pipeline.run_front_half(
        prompt="this is how it works",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    assert exploratory.human_language_layer is not None
    assert unsure.human_language_layer is not None
    assert assertive.human_language_layer is not None
    assert exploratory.human_language_layer.epistemic_stance == "exploratory"
    assert unsure.human_language_layer.epistemic_stance == "unsure"
    assert assertive.human_language_layer.epistemic_stance == "assertive"


def test_reasoning_pipeline_srd_marks_trust_risk_for_hard_clarify_under_doubt() -> None:
    pipeline = make_pipeline()

    forced = pipeline.srd_diagnostic.diagnose(
        dialogue_management=SimpleNamespace(interaction_mode="analytical"),
        conversation_awareness=SimpleNamespace(
            unresolved_thread_open=True,
            conversation_momentum="doubting",
        ),
        route_decision=RouteDecisionView(
            selected={},
            alternatives=[],
            normalized_scores=[],
            caution_notes=[],
            weak_route=False,
        ),
        low_confidence_recovery=LowConfidenceRecoveryResult(
            recovery_mode="hard_clarify",
            acknowledge_partial_understanding=False,
            clarifying_question_style="directional_recovery",
            rationale="Need clarification before continuing.",
        ),
    )

    assert "trust_failure" in forced.failure_types
    assert forced.should_exit_early is True


class _ArchiveServiceStub:
    def summary(self, *, session_id: str) -> dict[str, object]:
        return {
            "record_count": 2,
            "status_counts": {"ok": 2},
            "tool_counts": {"anh": 1},
        }

    def retrieve_context(self, query: str, *, session_id: str) -> dict[str, object]:
        return {
            "query": query,
            "record_count": 1,
            "top_matches": [
                {
                    "score": 8,
                    "matched_fields": ["semantic"],
                    "score_breakdown": {"keyword_score": 2, "semantic_score": 6},
                    "record": {
                        "summary": "Closest archive run for lumen migration",
                        "capability": "astronomy.anh_spectral_scan",
                        "created_at": "2026-03-15T00:00:00+00:00",
                    },
                }
            ],
        }


class _InteractionHistoryStub:
    def retrieve_context(self, query: str, *, session_id: str) -> dict[str, object]:
        return {
            "interaction_record_count": 1,
            "top_interaction_matches": [
                {
                    "score": 7,
                    "matched_fields": ["semantic"],
                    "score_breakdown": {"keyword_score": 3, "semantic_score": 4},
                    "record": {
                        "summary": "Prior session prompt about lumen routing migration",
                        "prompt": "create a migration plan",
                        "resolved_prompt": "create a migration plan",
                        "created_at": "2026-03-15T00:00:00+00:00",
                    },
                }
            ],
        }


class _SessionContextStub:
    def get_active_thread(self, session_id: str) -> dict[str, object]:
        return {
            "prompt": "create a migration plan",
            "summary": "Continuing lumen routing migration work",
            "mode": "planning",
            "kind": "planning.migration",
            "normalized_topic": "migration plan for lumen routing",
        }


class _DeepArchiveServiceStub:
    def summary(self, *, session_id: str) -> dict[str, object]:
        return {
            "record_count": 3,
            "status_counts": {"ok": 3},
            "tool_counts": {"anh": 2},
        }

    def retrieve_context(self, query: str, *, session_id: str) -> dict[str, object]:
        if query == "create a migration plan for lumen routing":
            return {
                "query": query,
                "record_count": 1,
                "top_matches": [
                    {
                        "score": 8,
                        "matched_fields": ["semantic"],
                        "score_breakdown": {"keyword_score": 2, "semantic_score": 6},
                        "record": {
                            "summary": "Closest archive run for lumen migration",
                            "capability": "astronomy.anh_spectral_scan",
                            "created_at": "2025-12-01T00:00:00+00:00",
                        },
                    }
                ],
            }
        return {
            "query": query,
            "record_count": 1,
            "top_matches": [
                {
                    "score": 7,
                    "matched_fields": ["summary"],
                    "score_breakdown": {"keyword_score": 4, "semantic_score": 1},
                    "record": {
                        "summary": "Secondary archive run for routing checkpoints",
                        "capability": "astronomy.anh_spectral_scan",
                        "created_at": "2026-03-10T00:00:00+00:00",
                    },
                }
            ],
        }


class _ContradictoryInteractionHistoryStub:
    def retrieve_context(self, query: str, *, session_id: str) -> dict[str, object]:
        return {
            "interaction_record_count": 1,
            "top_interaction_matches": [
                {
                    "score": 7,
                    "matched_fields": ["semantic"],
                    "score_breakdown": {"keyword_score": 3, "semantic_score": 4},
                    "record": {
                        "summary": "Compiler packaging strategy for deployment",
                        "prompt": "compare compiler packaging options",
                        "resolved_prompt": "compare compiler packaging options",
                        "created_at": "2026-03-15T00:00:00+00:00",
                    },
                }
            ],
        }


class _OldArchiveServiceStub:
    def summary(self, *, session_id: str) -> dict[str, object]:
        return {
            "record_count": 1,
            "status_counts": {"ok": 1},
            "tool_counts": {"anh": 1},
        }

    def retrieve_context(self, query: str, *, session_id: str) -> dict[str, object]:
        return {
            "query": query,
            "record_count": 1,
            "top_matches": [
                {
                    "score": 8,
                    "matched_fields": ["semantic"],
                    "score_breakdown": {"keyword_score": 2, "semantic_score": 6},
                    "record": {
                        "summary": "Old routing migration archive evidence",
                        "capability": "astronomy.anh_spectral_scan",
                        "created_at": "2025-08-01T00:00:00+00:00",
                    },
                }
            ],
        }


def test_reasoning_pipeline_builds_typed_validation_context() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={
            "interaction_count": 3,
            "clarification_count": 0,
            "retrieval_observation_count": 0,
        },
    )

    payload = validation.to_dict()
    assert payload["assistant_context"]["matched_record_count"] == 1
    assert payload["assistant_context"]["interaction_record_count"] == 1
    assert payload["evidence_quality_score"] > 0.5
    assert payload["retrieval_lead_summary"] == "archive=blended, interactions=blended"
    assert payload["failure_modes"]["weak_context"] is False
    assert payload["targets"][0]["source"] == "active_thread"
    assert payload["evidence_model"]["evidence_strength"] in {"supported", "strong"}
    assert payload["evidence_model"]["evidence_sources"][0]["source"] == "active_thread"
    assert payload["evidence_ledger"][0]["source"] == "active_thread"
    assert any(unit["source"] == "archive" for unit in payload["evidence_ledger"])
    assert any(unit["source"] == "interaction" for unit in payload["evidence_ledger"])
    trace = pipeline.create_trace(
        FrontHalfPipelineResult(
            intake=InputIntake(
                raw_input="create a migration plan for lumen routing",
                cleaned_input="create a migration plan for lumen routing",
                detected_language="en",
                session_id="default",
            ),
            nlu=NLUExtraction(
                dominant_intent="planning",
                secondary_intents=[],
                topic="migration plan for lumen routing",
                entities=[],
                action_cues={},
                ambiguity_flags=[],
                confidence_estimate=0.8,
            ),
            route_candidates=[],
            route_decision=RouteDecisionView(
                selected={},
                alternatives=[],
                normalized_scores=[],
                caution_notes=[],
                weak_route=False,
            ),
            resolved_prompt="create a migration plan for lumen routing",
            resolution_strategy="none",
            resolution_reason="No rewrite needed.",
            resolution_changed=False,
        )
    )
    pipeline.record_validation_context(trace, validation)
    assert trace.to_dict()["stage_contracts"]["context_retrieval_validation"]["confidence_signal"] == "evidence_quality_score"


def test_reasoning_pipeline_uses_deep_profile_to_broaden_archive_validation() -> None:
    capability_manager = CapabilityManager(manifests={})
    deep_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
        allow_suggestions=True,
    )
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_DeepArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=deep_profile,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        interaction_profile=deep_profile,
    )

    payload = validation.to_dict()
    assert payload["assistant_context"]["matched_record_count"] == 2
    assert len(payload["assistant_context"]["top_matches"]) == 2
    assert payload["evidence_model"]["evidence_strength"] == "strong"


def test_reasoning_pipeline_uses_deep_profile_for_stronger_contradiction_checking() -> None:
    capability_manager = CapabilityManager(manifests={})
    deep_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
        allow_suggestions=True,
    )
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_ContradictoryInteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=deep_profile,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        interaction_profile=deep_profile,
    )

    payload = validation.to_dict()
    assert "cross_source_topic_mismatch" in payload["contradiction_flags"]
    assert "cross_source_topic_mismatch" in payload["evidence_model"]["contradiction_flags"]


def test_reasoning_pipeline_applies_evidence_decay_and_reaffirmation() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_OldArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )

    payload = validation.to_dict()
    archive_unit = next(unit for unit in payload["evidence_ledger"] if unit["source"] == "archive")
    interaction_unit = next(unit for unit in payload["evidence_ledger"] if unit["source"] == "interaction")

    assert archive_unit["age_bucket"] in {"stale", "old"}
    assert archive_unit["decay_factor"] < 1.0
    assert archive_unit["reaffirmed"] is True
    assert archive_unit["authority_score"] < 8.0
    assert interaction_unit["age_bucket"] == "recent"
    assert interaction_unit["decay_factor"] == 1.0


def test_reasoning_pipeline_keeps_reaffirmed_interaction_evidence_more_durable_than_archive() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_OldArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )

    payload = validation.to_dict()
    archive_unit = next(unit for unit in payload["evidence_ledger"] if unit["source"] == "archive")
    interaction_unit = next(unit for unit in payload["evidence_ledger"] if unit["source"] == "interaction")

    assert archive_unit["age_bucket"] in {"stale", "old"}
    assert interaction_unit["age_bucket"] == "recent"
    assert interaction_unit["authority_score"] > archive_unit["authority_score"]


def test_reasoning_pipeline_assembles_typed_reasoning_frame() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )

    assembled = pipeline.assemble_reasoning_frame(
        prompt="create a migration plan for lumen routing",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile.default(),
        validation_context=validation,
    )

    payload = assembled.to_dict()
    assert payload["frame_type"] == "plan-next-step"
    assert payload["problem_interpretation"].startswith("Interpret this as a migration-planning request:")
    assert payload["working_hypothesis"]
    assert payload["validation_plan"]
    assert payload["route_quality"] in {"strong", "supported", "weak"}
    assert payload["confidence_posture"] in {"strong", "supported", "tentative", "conflicted"}
    assert "failure_modes" in payload
    trace = pipeline.create_trace(
        FrontHalfPipelineResult(
            intake=InputIntake(
                raw_input="create a migration plan for lumen routing",
                cleaned_input="create a migration plan for lumen routing",
                detected_language="en",
                session_id="default",
            ),
            nlu=NLUExtraction(
                dominant_intent="planning",
                secondary_intents=[],
                topic="migration plan for lumen routing",
                entities=[],
                action_cues={},
                ambiguity_flags=[],
                confidence_estimate=0.8,
            ),
            route_candidates=[],
            route_decision=RouteDecisionView(
                selected={},
                alternatives=[],
                normalized_scores=[],
                caution_notes=[],
                weak_route=False,
            ),
            resolved_prompt="create a migration plan for lumen routing",
            resolution_strategy="none",
            resolution_reason="No rewrite needed.",
            resolution_changed=False,
        )
    )
    pipeline.record_reasoning_frame(trace, assembled)
    assert trace.to_dict()["stage_contracts"]["reasoning_frame_assembly"]["failure_state"] is None


def test_reasoning_pipeline_synthesizes_response_view() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )
    assembled = pipeline.assemble_reasoning_frame(
        prompt="create a migration plan for lumen routing",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile.default(),
        validation_context=validation,
    )

    synthesis = pipeline.synthesize_response(
        mode="planning",
        kind=route.kind,
        reasoning_frame_assembly=assembled,
        response_payload={
            "steps": ["step one", "step two"],
            "next_action": "validate the archive first",
            "uncertainty_note": "still tentative",
        },
    )

    payload = synthesis.to_dict()
    assert payload["mode"] == "planning"
    assert payload["response_body"][0].startswith("Support status: strongly supported")
    assert payload["response_body"][1].startswith("Anchor evidence:")
    assert payload["response_body"][2].startswith("Supporting evidence:")
    assert payload["response_body"][3:] == ["step one", "step two"]
    assert payload["suggested_next_step"] == "validate the archive first"
    assert payload["route_evidence_distinction"] in {
        "weak_route_but_supported_evidence",
        "right_route_but_weak_evidence",
        "weak_route_and_weak_evidence",
        "route_and_evidence_generally_aligned",
    }


def test_reasoning_pipeline_carries_evidence_model_into_reasoning_frame() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )

    assembled = pipeline.assemble_reasoning_frame(
        prompt="create a migration plan for lumen routing",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile.default(),
        validation_context=validation,
    )

    payload = assembled.to_dict()
    assert payload["evidence_model"]["evidence_strength"] in {"supported", "strong"}
    assert payload["evidence_model"]["contradiction_flags"] == []
    assert "archive" in [item["source"] for item in payload["evidence_model"]["evidence_sources"]]
    assert payload["status_snapshot"]["support_status"] in {
        "strongly_supported",
        "moderately_supported",
        "insufficiently_grounded",
    }
    assert payload["status_snapshot"]["route_status"] in {
        "stable",
        "weakened",
        "under_tension",
        "revised",
        "unresolved",
    }
    assert payload["anchor_evidence_id"] in {"archive:0", "interaction:0", "active_thread:0"}
    assert isinstance(payload["evidence_ledger"], list)
    assert any(unit["selected_as_anchor"] for unit in payload["evidence_ledger"])
    assert payload["tension_resolution"]["tension_detected"] is False


def test_reasoning_pipeline_synthesizes_direct_answer_why_action() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    _, route = pipeline.run_front_half(
        prompt="summarize the archive structure",
        session_id="default",
        interaction_profile=InteractionProfile(
            interaction_style="direct",
            reasoning_depth="normal",
            selection_source="user",
            allow_suggestions=True,
        ),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    validation = pipeline.build_validation_context(
        prompt="summarize the archive structure",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )
    assembled = pipeline.assemble_reasoning_frame(
        prompt="summarize the archive structure",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile(
            interaction_style="direct",
            reasoning_depth="normal",
            selection_source="user",
            allow_suggestions=True,
        ),
        validation_context=validation,
    )

    synthesis = pipeline.synthesize_response(
        mode="research",
        kind=route.kind,
        reasoning_frame_assembly=assembled,
        response_payload={
            "findings": ["detail one", "detail two", "detail three"],
            "recommendation": "Use the archive summary first.",
            "uncertainty_note": "keep it narrow",
        },
    )

    payload = synthesis.to_dict()
    assert payload["response_body"][0] == "Answer: Use the archive summary first."
    assert payload["response_body"][1].startswith("Why:")
    assert payload["response_body"][2] == "Action: Use the archive summary first."
    assert len(payload["validation_advice"]) <= 1


def test_reasoning_pipeline_surfaces_missing_context_in_synthesis() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=pipeline.assemble_reasoning_frame(
            prompt="summarize the archive structure",
            route=DomainRoute(
                mode="research",
                kind="research.summary",
                normalized_prompt="summarize the archive structure",
                confidence=0.72,
                reason="Research cues matched with score 3",
                source="heuristic_research",
            ),
            kind="research.summary",
            interaction_profile=InteractionProfile.default(),
            validation_context=ValidationContextResult(
                assistant_context=AssistantContext.from_mapping({}),
                evidence_quality_score=0.2,
                retrieval_lead_summary=None,
                missing_context_note="No archive, interaction, or active-thread context was available for validation.",
                contradiction_flags=[],
                failure_modes={"weak_context": True},
                targets=[],
                evidence_model=LightweightEvidenceModel(
                    evidence_sources=[],
                    evidence_strength="missing",
                    contradiction_flags=[],
                    missing_sources=["archive", "interactions"],
                ),
            ),
        ),
        response_payload={
            "findings": ["detail one", "detail two"],
            "recommendation": "Start with a narrower summary.",
            "uncertainty_note": "still tentative",
        },
    )

    payload = synthesis.to_dict()
    assert payload["response_body"][0] == "Missing supporting context: archive, interactions."


def test_reasoning_pipeline_surfaces_light_support_in_synthesis() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=pipeline.assemble_reasoning_frame(
            prompt="summarize the archive structure",
            route=DomainRoute(
                mode="research",
                kind="research.summary",
                normalized_prompt="summarize the archive structure",
                confidence=0.82,
                reason="Research cues matched with score 4",
                source="heuristic_research",
            ),
            kind="research.summary",
            interaction_profile=InteractionProfile.default(),
            validation_context=ValidationContextResult(
                assistant_context=AssistantContext.from_mapping({}),
                evidence_quality_score=0.4,
                retrieval_lead_summary=None,
                missing_context_note=None,
                contradiction_flags=[],
                failure_modes={"weak_evidence": True},
                targets=[],
                evidence_model=LightweightEvidenceModel(
                    evidence_sources=[{"source": "archive", "record_count": 1}],
                    evidence_strength="light",
                    contradiction_flags=[],
                    missing_sources=[],
                ),
            ),
        ),
        response_payload={
            "findings": ["detail one", "detail two"],
            "recommendation": "Start with a narrower summary.",
            "uncertainty_note": "still tentative",
        },
    )

    payload = synthesis.to_dict()
    assert payload["response_body"][0] == "Support status: moderately supported. Keep this as a narrower first pass."


def test_reasoning_pipeline_distinguishes_weak_route_but_supported_evidence_in_synthesis() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="analyze",
            problem_interpretation="Interpret this as an analysis request.",
            local_context_summary="Closest archive run: test",
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="The archive evidence is decent, but the route is still weak.",
            validation_plan=["Validate the route choice itself before leaning too hard on the result."],
            interaction_profile={
                "interaction_style": "conversational",
                "reasoning_depth": "normal",
            },
            route_quality="weak",
            grounding_strength="high",
            route_status="weakened",
            support_status="strongly_supported",
            tension_status="stable",
            confidence_posture="tentative",
            uncertainty_posture="tentative",
            evidence_model=LightweightEvidenceModel(
                evidence_sources=[{"source": "archive"}, {"source": "interaction"}],
                evidence_strength="strong",
                contradiction_flags=[],
                missing_sources=[],
            ),
        ),
        response_payload={
            "findings": ["detail one", "detail two"],
            "recommendation": "Validate the route before broadening the answer.",
            "uncertainty_note": "the route itself is still weak",
        },
    )

    payload = synthesis.to_dict()
    assert payload["route_evidence_distinction"] == "weak_route_but_supported_evidence"
    assert payload["response_body"][0].startswith("Support status: strongly supported")
    assert payload["validation_advice"][0] == "Validate the route choice itself before leaning too hard on the result."


def test_reasoning_pipeline_distinguishes_right_route_but_weak_evidence_in_synthesis() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="retrieve-and-summarize",
            problem_interpretation="Interpret this as a summary request.",
            local_context_summary=None,
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="The route is reasonable, but support is thin.",
            validation_plan=["Treat the first answer pass as exploratory until another local signal confirms it."],
            interaction_profile={
                "interaction_style": "conversational",
                "reasoning_depth": "normal",
            },
            route_quality="supported",
            grounding_strength="low",
            route_status="stable",
            support_status="insufficiently_grounded",
            tension_status="stable",
            confidence_posture="tentative",
            uncertainty_posture="tentative",
            evidence_model=LightweightEvidenceModel(
                evidence_sources=[],
                evidence_strength="missing",
                contradiction_flags=[],
                missing_sources=["archive", "interactions"],
            ),
        ),
        response_payload={
            "findings": ["detail one", "detail two"],
            "recommendation": "Start with a narrow answer and confirm it locally.",
            "uncertainty_note": "support is still thin",
        },
    )

    payload = synthesis.to_dict()
    assert payload["route_evidence_distinction"] == "right_route_but_weak_evidence"
    assert payload["response_body"][0] == "Missing supporting context: archive, interactions."
    assert payload["validation_advice"][0] == "Treat the first answer pass as exploratory until another local signal confirms it."


def test_reasoning_pipeline_marks_ambiguous_tension_for_clarification_resolution() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_ContradictoryInteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    route = DomainRoute(
        mode="research",
        kind="research.summary",
        normalized_prompt="review the migration summary",
        confidence=0.74,
        reason="Research and planning cues were close.",
        source="heuristic_research",
        decision_summary={
            "selected": {},
            "alternatives": [],
            "ambiguous": True,
            "ambiguity_reason": "Top route candidates were close.",
        },
    )

    validation = pipeline.build_validation_context(
        prompt="review the migration summary",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )
    assembled = pipeline.assemble_reasoning_frame(
        prompt="review the migration summary",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile.default(),
        validation_context=validation,
    )

    payload = assembled.to_dict()
    assert payload["tension_resolution"]["tension_detected"] is True
    assert payload["tension_resolution"]["category"] == "clarification_tension"
    assert payload["tension_resolution"]["resolution_path"] == "clarification"
    assert payload["tension_resolution"]["anchor_status"] == "uncertain"
    assert payload["tension_resolution"]["recommended_action"] == "clarify_conflict"


def test_reasoning_pipeline_uses_alternate_hypothesis_resolution_for_non_ambiguous_tension() -> None:
    capability_manager = CapabilityManager(manifests={})
    deep_profile = InteractionProfile(
        interaction_style="conversational",
        reasoning_depth="deep",
        selection_source="user",
        allow_suggestions=True,
    )
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_ContradictoryInteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    route = DomainRoute(
        mode="research",
        kind="research.summary",
        normalized_prompt="review the migration summary",
        confidence=0.81,
        reason="Research cues matched with score 4",
        source="heuristic_research",
        decision_summary={
            "selected": {},
            "alternatives": [],
            "ambiguous": False,
            "ambiguity_reason": None,
        },
    )
    validation = pipeline.build_validation_context(
        prompt="review the migration summary",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        interaction_profile=deep_profile,
    )
    assembled = pipeline.assemble_reasoning_frame(
        prompt="review the migration summary",
        route=route,
        kind=route.kind,
        interaction_profile=deep_profile,
        validation_context=validation,
    )

    payload = assembled.to_dict()
    assert payload["tension_resolution"]["tension_detected"] is True
    assert payload["tension_resolution"]["category"] == "evidence_tension"
    assert payload["tension_resolution"]["resolution_path"] == "alternate_hypothesis"
    assert payload["tension_resolution"]["alternate_hypotheses"]
    assert payload["tension_resolution"]["leading_hypothesis_label"] is not None
    assert payload["tension_resolution"]["recommended_action"].startswith("compare_hypotheses")
    assert payload["tension_resolution"]["anchor_status"] in {"stable", "weakened"}


def test_reasoning_pipeline_compresses_tension_resolution_for_direct() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="retrieve-and-summarize",
            problem_interpretation="Interpret this as a summary request.",
            local_context_summary="Closest archive run: test",
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="Use the archive summary.",
            interaction_profile={
                "interaction_style": "direct",
                "reasoning_depth": "deep",
            },
            evidence_model=LightweightEvidenceModel(
                evidence_sources=[{"source": "archive"}],
                evidence_strength="supported",
            ),
            tension_resolution={
                "tension_detected": True,
                "category": "evidence_tension",
                "resolution_path": "alternate_hypothesis",
                "rationale": "Conflict detected: X competes with Y. Alternate hypotheses required.",
            },
        ),
        response_payload={
            "findings": ["detail one"],
            "recommendation": "Use X.",
        },
    )

    payload = synthesis.to_dict()
    assert payload["response_body"][0] == "Answer: Use X."
    assert "Conflict detected:" in payload["response_body"][1]


def test_reasoning_pipeline_surfaces_competing_hypotheses_and_guidance_in_conversational_synthesis() -> None:
    pipeline = make_pipeline()
    synthesis = pipeline.synthesize_response(
        mode="research",
        kind="research.summary",
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="analyze",
            problem_interpretation="Interpret this as an analysis request.",
            local_context_summary="Mixed local context.",
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="Hypothesis A currently fits best.",
            validation_plan=["Validate the route choice itself before leaning too hard on the result."],
            interaction_profile={
                "interaction_style": "conversational",
                "reasoning_depth": "deep",
            },
            evidence_model=LightweightEvidenceModel(
                evidence_sources=[{"source": "archive"}, {"source": "interaction"}],
                evidence_strength="supported",
            ),
            tension_resolution={
                "tension_detected": True,
                "category": "evidence_tension",
                "resolution_path": "alternate_hypothesis",
                "rationale": "There is a meaningful conflict between the current anchor and competing evidence.",
                "status": "under_tension",
                "recommended_action": "compare_hypotheses_a",
                "alternate_hypotheses": [
                    {"label": "A", "summary": "Archive-led interpretation", "status": "leading"},
                    {"label": "B", "summary": "Interaction-led interpretation", "status": "competing"},
                ],
            },
        ),
        response_payload={
            "findings": ["detail one", "detail two"],
            "recommendation": "Compare the strongest conflicting sources first.",
        },
    )

    payload = synthesis.to_dict()
    assert payload["response_body"][0] in {
        "This answer is still carrying real tension.",
        "This answer is still under real tension.",
        "There is still real tension in this answer.",
    }
    assert payload["response_body"][2].startswith(
        (
            "Right now, hypothesis A is carrying",
            "At the moment, hypothesis A is carrying",
            "Right now, hypothesis A is leading on weight,",
        )
    )
    assert payload["response_body"][3].startswith(
        (
            "The competing explanations are still live:",
            "The competing explanations are still in play:",
            "The competing explanations are still active:",
        )
    )
    assert payload["validation_advice"][0] == "Compare hypotheses A vs B before treating the current anchor as settled."


def test_reasoning_pipeline_packages_reasoned_execution_stage() -> None:
    pipeline = make_pipeline()

    execution = pipeline.package_execution_stage(
        mode="planning",
        kind="planning.migration",
        response_payload={
            "summary": "Grounded planning response for: create a migration plan for lumen routing",
            "confidence_posture": "supported",
        },
    )

    payload = execution.to_dict()
    assert payload["execution_type"] == "reasoned_response"
    assert payload["executed"] is True
    assert payload["execution_metadata"]["confidence_posture"] == "supported"


def test_reasoning_pipeline_packages_tool_execution_stage() -> None:
    pipeline = make_pipeline()

    execution = pipeline.package_execution_stage(
        mode="tool",
        kind="tool.command_alias",
        response_payload={
            "summary": "GA Local Analysis Kit run completed",
            "tool_execution": {
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
            },
            "tool_route_origin": "nlu_hint_alias",
        },
    )

    payload = execution.to_dict()
    assert payload["execution_type"] == "tool_call"
    assert payload["execution_metadata"]["tool_id"] == "anh"
    assert "NLU hint alias" in payload["warnings"][0]


def test_reasoning_pipeline_packages_final_response_view() -> None:
    pipeline = make_pipeline()
    route = DomainRoute(
        mode="planning",
        kind="planning.migration",
        normalized_prompt="create a migration plan for lumen routing",
        confidence=0.82,
        reason="Planning cues matched with score 4",
        source="heuristic_planning",
    )

    packaging = pipeline.package_response(
        mode="planning",
        kind="planning.migration",
        route=route,
        response_payload={
            "summary": "Grounded planning response for: create a migration plan for lumen routing",
            "next_action": "Validate the closest archive run first.",
            "confidence_posture": "supported",
            "best_evidence": "Routing selected planning because Planning cues matched with score 4.",
        },
    )

    payload = packaging.to_dict()
    assert payload["package_type"] == "structured"
    assert payload["answer"] == "Validate the closest archive run first."
    assert payload["confidence"] == "supported"
    assert payload["route_summary"]["source"] == "heuristic_planning"


def test_reasoning_pipeline_packages_persistence_observation() -> None:
    pipeline = make_pipeline()
    route = DomainRoute(
        mode="planning",
        kind="planning.migration",
        normalized_prompt="create a migration plan for lumen routing",
        confidence=0.82,
        reason="Planning cues matched with score 4",
        source="heuristic_planning",
    )
    front_half, _ = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    packaging = pipeline.package_response(
        mode="planning",
        kind="planning.migration",
        route=route,
        response_payload={
            "summary": "Grounded planning response for: create a migration plan for lumen routing",
            "next_action": "Validate the closest archive run first.",
            "confidence_posture": "supported",
        },
    )

    observation = pipeline.package_persistence_observation(
        session_id="default",
        prompt="create a migration plan for lumen routing",
        front_half=front_half,
        route=route,
        clarification_decision=None,
        validation_context=None,
        reasoning_frame_assembly=None,
        execution_stage=None,
        response_packaging=packaging,
    )

    payload = observation.to_dict()
    assert payload["session_id"] == "default"
    assert payload["route_summary"]["reason"] == "Planning cues matched with score 4"
    assert payload["response_summary"]["answer"] == "Validate the closest archive run first."


def test_reasoning_pipeline_persists_ledger_references_in_reasoning_summary() -> None:
    capability_manager = CapabilityManager(manifests={})
    pipeline = ReasoningPipeline(
        prompt_resolver=PromptResolver(capability_manager=capability_manager),
        domain_router=DomainRouter(capability_manager=capability_manager),
        archive_service=_ArchiveServiceStub(),
        interaction_history_service=_InteractionHistoryStub(),
        session_context_service=_SessionContextStub(),
    )
    front_half, route = pipeline.run_front_half(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )
    validation = pipeline.build_validation_context(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        route=route,
        interaction_summary={"interaction_count": 3, "clarification_count": 0},
    )
    assembled = pipeline.assemble_reasoning_frame(
        prompt="create a migration plan for lumen routing",
        route=route,
        kind=route.kind,
        interaction_profile=InteractionProfile.default(),
        validation_context=validation,
    )
    packaging = pipeline.package_response(
        mode="planning",
        kind=route.kind,
        route=route,
        response_payload={
            "summary": "Grounded planning response for: create a migration plan for lumen routing",
            "next_action": "Validate the closest archive run first.",
            "confidence_posture": "supported",
        },
    )

    observation = pipeline.package_persistence_observation(
        session_id="default",
        prompt="create a migration plan for lumen routing",
        front_half=front_half,
        route=route,
        clarification_decision=None,
        validation_context=validation,
        reasoning_frame_assembly=assembled,
        execution_stage=None,
        response_packaging=packaging,
    )

    payload = observation.to_dict()
    assert payload["reasoning_summary"]["anchor_evidence_id"] in {"archive:0", "interaction:0", "active_thread:0"}
    assert "tension_evidence_ids" in payload["reasoning_summary"]
    assert payload["reasoning_summary"]["anchor_evidence_summary"] is not None


def test_reasoning_status_policy_normalizes_impossible_status_combinations() -> None:
    pipeline = make_pipeline()

    snapshot = pipeline.reasoning_status_policy.build_snapshot(
        route_strength="high",
        route_quality="strong",
        grounding_strength="high",
        local_context_assessment="aligned",
        route_ambiguity=False,
        contradiction_flags=[],
        evidence_strength="strong",
        failure_modes={"weak_context": True},
        tension_resolution=TensionResolutionResult(
            tension_detected=False,
            category=None,
            resolution_path=None,
            rationale=None,
            status="stable",
        ),
    )

    assert snapshot.support_status == "insufficiently_grounded"


def test_reasoning_status_policy_carries_unresolved_tension_into_route_status() -> None:
    pipeline = make_pipeline()

    snapshot = pipeline.reasoning_status_policy.build_snapshot(
        route_strength="high",
        route_quality="strong",
        grounding_strength="high",
        local_context_assessment="aligned",
        route_ambiguity=False,
        contradiction_flags=["ambiguous_route"],
        evidence_strength="strong",
        failure_modes={"high_ambiguity": True},
        tension_resolution=TensionResolutionResult(
            tension_detected=True,
            category="clarification_tension",
            resolution_path="clarification",
            rationale="Clarification should come first.",
            status="unresolved",
        ),
    )

    assert snapshot.tension_status == "unresolved"
    assert snapshot.route_status == "unresolved"


def test_reasoning_pipeline_keeps_active_profile_authoritative_while_emitting_advice() -> None:
    pipeline = make_pipeline()

    result, _ = pipeline.run_front_half(
        prompt="give me a brief direct answer about the archive structure",
        session_id="default",
        interaction_profile=InteractionProfile.default(),
        interaction_summary={"interaction_count": 0, "clarification_count": 0},
        recent_interactions=[],
        active_thread=None,
    )

    payload = result.to_dict()
    assert payload["intake"]["interaction_profile"]["interaction_style"] == "collab"
    assert payload["nlu"]["profile_advice"]["interaction_style"] == "direct"
    assert payload["nlu"]["profile_mismatch"] is True


def test_reasoning_pipeline_uses_profile_to_shape_package_type() -> None:
    pipeline = make_pipeline()
    route = DomainRoute(
        mode="research",
        kind="research.summary",
        normalized_prompt="summarize the archive structure",
        confidence=0.82,
        reason="Research cues matched with score 4",
        source="heuristic_research",
    )

    packaging = pipeline.package_response(
        mode="research",
        kind="research.summary",
        route=route,
        response_payload={
            "summary": "Grounded research response for: summarize the archive structure",
            "recommendation": "Summarize the strongest local evidence first.",
            "confidence_posture": "supported",
        },
        interaction_profile=InteractionProfile(
            interaction_style="direct",
            reasoning_depth="normal",
            selection_source="user",
            allow_suggestions=True,
        ),
    )

    payload = packaging.to_dict()
    assert payload["package_type"] == "structured"
    assert payload["route_summary"] == {
        "source": "heuristic_research",
        "strength": "medium",
    }
    assert payload["follow_up_suggestions"] == ["Summarize the strongest local evidence first."]


