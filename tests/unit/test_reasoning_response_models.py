from pathlib import Path

from lumen.app.models import InteractionProfile
from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.anti_roleplay_policy import AntiRoleplayPolicy
from lumen.reasoning.conversation_policy import ConversationPolicy
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.pipeline_models import ReasoningFrameAssembly
from lumen.reasoning.planner import Planner
from lumen.reasoning.research_engine import ResearchEngine
from lumen.reasoning.response_models import RouteMetadata, ToolAssistantResponse, ToolExecutionDetails
from lumen.routing.domain_router import DomainRoute
from lumen.tools.registry_types import ToolResult


def test_planner_returns_versioned_typed_response_payload() -> None:
    response = Planner().respond(
        "create a migration plan for lumen",
        kind="planning.migration",
        context=AssistantContext.from_mapping(
            {
                "record_count": 2,
                "route": {
                    "confidence": 0.82,
                    "reason": "Planning cues matched with score 4",
                    "decision_summary": {"selected": {"normalized_score": 1.92}},
                },
                "top_matches": [
                    {
                        "score": 5,
                        "record": {
                            "tool_id": "anh",
                            "capability": "spectral_dip_scan",
                            "summary": "Great Attractor confirmation candidate",
                        },
                    }
                ],
            }
        ),
    )

    assert response["schema_type"] == "assistant_response"
    assert response["schema_version"] == "1"
    assert response["mode"] == "planning"
    assert response["kind"] == "planning.migration"
    assert response["summary"] in {
        "Here’s a workable first pass.",
        "Here’s a solid first plan.",
    }
    assert isinstance(response["steps"], list)
    assert response["context"]["record_count"] == 2
    assert response["grounding_strength"] == "high"
    assert response["best_evidence"].startswith("Routing selected planning because")
    assert "Closest archive run: anh/spectral_dip_scan" in response["local_context_summary"]
    assert response["working_hypothesis"].startswith("The best working hypothesis is to use")
    assert response.get("uncertainty_note") is None
    assert "The strongest local signal for this planning response is anh/spectral_dip_scan" in response["grounded_interpretation"]
    assert response["reasoning_frame"]["primary_anchor"].startswith("anh/spectral_dip_scan")
    assert response["local_context_assessment"] == "partial"
    assert response["confidence_posture"] == "supported"
    assert response["closing_strategy"] == "proceed_with_context"
    assert response["steps"][0] == "Define the current state, target state, and non-negotiable migration constraints."
    assert all(not step.startswith("Working hypothesis:") for step in response["steps"])
    assert any(
        option in response["next_action"]
        for option in {
            "Use the closest archive run as the baseline.",
            "Use the closest archive run as the starting baseline.",
            "Keep the closest archive run as the baseline reference.",
        }
    )


def test_research_engine_returns_versioned_typed_response_payload() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "status_counts": {"ok": 2},
                "route": {
                    "confidence": 0.8,
                    "reason": "Explicit summary prompt matched summary routing",
                    "decision_summary": {"selected": {"normalized_score": 1.88}},
                },
                "top_interaction_matches": [
                    {
                        "score": 4,
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "summarize the archive structure",
                                "original_prompt": "summarize the archive structure",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["schema_type"] == "assistant_response"
    assert response["schema_version"] == "1"
    assert response["mode"] == "research"
    assert response["kind"] == "research.summary"
    assert response["summary"] in {
        "Here’s a workable answer.",
        "Here’s the clearest current answer.",
    }
    assert isinstance(response["findings"], list)
    assert response["context"]["status_counts"]["ok"] == 2
    assert response["grounding_strength"] == "high"
    assert response["best_evidence"].startswith("Routing selected research because")
    assert "Closest prior session prompt: summarize the archive structure" in response["local_context_summary"]
    assert response["working_hypothesis"].startswith("The best working hypothesis is to use")
    assert response.get("uncertainty_note") is None
    assert "The strongest local signal for this research response is summarize the archive structure" in response["grounded_interpretation"]
    assert response["reasoning_frame"]["primary_anchor"] == "summarize the archive structure"
    assert response["local_context_assessment"] == "partial"
    assert response["confidence_posture"] == "supported"
    assert response["closing_strategy"] == "proceed_with_context"
    assert response["findings"][0] == "State the topic in one concise sentence."
    assert all(not item.startswith("Working hypothesis:") for item in response["findings"])
    assert any(
        option in response["recommendation"]
        for option in {
            "Reconcile it with the closest prior session prompt first.",
            "First reconcile it with the closest prior session prompt.",
            "Anchor it against the closest prior session prompt first.",
        }
    )


def test_research_engine_uses_aligned_local_context_as_anchor() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.8,
                    "reason": "Explicit summary prompt matched summary routing",
                    "strength": "high",
                },
                "top_matches": [
                    {
                        "score": 5,
                        "record": {
                            "tool_id": "anh",
                            "capability": "spectral_dip_scan",
                            "summary": "Archive structure migration summary",
                        },
                    }
                ],
                "top_interaction_matches": [
                    {
                        "score": 4,
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "archive structure migration summary",
                                "original_prompt": "archive structure migration summary",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["local_context_assessment"] == "aligned"
    assert response["confidence_posture"] == "strong"
    assert response["closing_strategy"] == "anchor_and_execute"
    assert response["summary"] in {
        "Here’s the grounded answer.",
        "Here’s a grounded first pass.",
    }
    assert response["working_hypothesis"].startswith("The best working hypothesis is to treat")
    assert response.get("uncertainty_note") is None
    assert "Local evidence is aligned and points to a consistent research direction anchored by" in response["grounded_interpretation"]
    assert "Use this aligned anchor as the strongest signal:" in response["findings"][1]
    assert response["findings"][2] == "Promote the shared signal to the main conclusion before adding secondary detail."
    assert "Use the agreement between archived evidence and prior session context as the main anchor." in response["recommendation"]


def test_research_engine_mentions_cross_source_coherence_topic_when_semantically_aligned() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.78,
                    "reason": "Explicit summary prompt matched summary routing",
                    "strength": "high",
                },
                "top_matches": [
                    {
                        "score": 6,
                        "matched_fields": ["semantic"],
                        "record": {
                            "tool_id": "anh",
                            "capability": "spectral_dip_scan",
                            "summary": "Lumen routing migration summary",
                        },
                    }
                ],
                "top_interaction_matches": [
                    {
                        "score": 6,
                        "matched_fields": ["semantic"],
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "routing migration summary for lumen",
                                "original_prompt": "routing migration summary for lumen",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["grounding_strength"] == "high"
    assert response["reasoning_frame"]["coherence_topic"]
    assert "The strongest local evidence sources reinforce the same topic:" in response["findings"][2]
    assert "Keep the shared local topic explicit while interpreting the result." in response["recommendation"]


def test_planner_softens_next_action_when_grounding_is_low() -> None:
    response = Planner().respond(
        "create a migration plan for lumen",
        kind="planning.migration",
        context=AssistantContext.from_mapping(
            {
                "route": {"confidence": 0.4, "reason": "Defaulted to planning after weak cues"},
            }
        ),
    )

    assert response["grounding_strength"] == "low"
    assert response["confidence_posture"] == "tentative"
    assert response["closing_strategy"] == "exploratory_validation"
    assert response["summary"] in {
        "Here’s a first pass using the best current assumptions.",
        "Here’s a useful first plan, with the assumptions kept visible.",
    }
    assert response.get("working_hypothesis") is None
    assert "settle this plan with confidence" in response["uncertainty_note"]
    assert response["grounded_interpretation"] == "Local evidence for this planning response is thin, so treat the result as exploratory."
    assert response["steps"][1] == "Treat the opening milestone as evidence-gathering and validation, not commitment."
    assert "confirm the assumptions with an additional local check" in response["next_action"]


def test_planner_synthesizes_first_pass_design_for_engine_prompt() -> None:
    response = Planner().respond(
        "design me an engine",
        kind="planning.architecture",
        context=AssistantContext.from_mapping(
            {
                "route": {"confidence": 0.81, "reason": "Planning cues matched with score 4"},
            }
        ),
    )

    assert response["mode"] == "planning"
    assert response["kind"] == "planning.architecture"
    assert "first-pass design concept" in response["summary"].lower()
    assert response["steps"][0].startswith("Assumptions:")
    assert any(step.startswith("High-level system:") for step in response["steps"])
    assert any(step.startswith("Key components:") for step in response["steps"])
    assert any(step.startswith("Interaction:") for step in response["steps"])
    assert any(step.startswith("Next refinement:") for step in response["steps"])
    assert response["next_action"].startswith("Next refinement:")


def test_research_engine_treats_weak_route_score_as_more_tentative() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.76,
                    "reason": "Research cues matched with score 3",
                    "strength": "medium",
                    "caution": "Route selection remained close because research and planning cues were both plausible.",
                    "decision_summary": {"selected": {"normalized_score": 1.22}},
                    "ambiguity": {"reason": "Top research and planning candidates were close in rank."},
                },
                "top_interaction_matches": [
                    {
                        "score": 5,
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "archive structure",
                                "original_prompt": "archive structure",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["confidence_posture"] == "tentative"
    assert "validate the route choice itself against the closest prior session prompt before leaning on the conclusion" in response["recommendation"]


def test_planner_marks_strong_evidence_but_weak_route_as_route_validation_case() -> None:
    response = Planner().respond(
        "review the migration summary",
        kind="planning.migration",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.82,
                    "reason": "Planning cues narrowly outranked research cues",
                    "strength": "high",
                    "decision_summary": {"selected": {"normalized_score": 1.22}},
                },
                "top_matches": [
                    {
                        "score": 5,
                        "record": {
                            "tool_id": "anh",
                            "capability": "spectral_dip_scan",
                            "summary": "Migration summary validation run",
                        },
                    }
                ],
            }
        ),
    )

    assert response["grounding_strength"] == "medium"
    assert response["confidence_posture"] == "tentative"
    assert "Local evidence is fairly strong for this planning response, but the winning route remains comparatively weak" in response["grounded_interpretation"]
    assert (
        "Treat the first milestone as a route-validation checkpoint, because the local evidence is fairly strong and the route should be validated against the closest archive evidence before broadening scope."
        in response["steps"]
    )


def test_research_engine_marks_strong_evidence_but_weak_route_as_route_validation_case() -> None:
    response = ResearchEngine().respond(
        "summarize the migration summary",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.82,
                    "reason": "Research cues narrowly outranked planning cues",
                    "strength": "high",
                    "decision_summary": {"selected": {"normalized_score": 1.22}},
                },
                "top_interaction_matches": [
                    {
                        "score": 5,
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "migration summary",
                                "original_prompt": "migration summary",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["grounding_strength"] == "medium"
    assert response["confidence_posture"] == "tentative"
    assert "Local evidence is fairly strong for this research response, but the winning route remains comparatively weak" in response["grounded_interpretation"]
    assert (
        "Treat the first conclusion as a route-validation checkpoint, because the local evidence is fairly strong and the chosen route should be validated against the closest prior session prompt."
        in response["findings"]
    )
    assert "validate the route choice itself against the closest prior session prompt before leaning on the conclusion" in response["recommendation"]


def test_planner_uses_tension_resolution_to_sharpen_mixed_context_step() -> None:
    response = Planner().respond(
        "review the migration summary",
        kind="planning.migration",
        context=AssistantContext.from_mapping({}),
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="plan-next-step",
            problem_interpretation="Interpret this as a planning request.",
            local_context_summary="Mixed local context.",
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="Keep the current anchor, but test the conflict.",
            reasoning_frame={"tension": "Archive evidence conflicts with prior interaction evidence."},
            local_context_assessment="mixed",
            grounding_strength="high",
            route_quality="supported",
            interaction_profile={"interaction_style": "conversational", "reasoning_depth": "deep"},
            tension_resolution={
                "tension_detected": True,
                "resolution_path": "alternate_hypothesis",
                "leading_hypothesis_label": "A",
                "recommended_action": "compare_hypotheses_a",
            },
        ),
    )

    assert "Archive evidence conflicts with prior interaction evidence." in response["steps"]
    assert (
        "Keep the first milestone centered on the leading hypothesis (A) while keeping the competing explanation explicit."
        in response["steps"]
    )


def test_research_engine_uses_tension_resolution_to_sharpen_mixed_context_finding() -> None:
    response = ResearchEngine().respond(
        "summarize the migration summary",
        kind="research.summary",
        context=AssistantContext.from_mapping({}),
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="analyze",
            problem_interpretation="Interpret this as a research request.",
            local_context_summary="Mixed local context.",
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis="Hypothesis A currently fits best.",
            reasoning_frame={"tension": "Archive evidence conflicts with prior interaction evidence."},
            local_context_assessment="mixed",
            grounding_strength="high",
            route_quality="supported",
            interaction_profile={"interaction_style": "conversational", "reasoning_depth": "deep"},
            tension_resolution={
                "tension_detected": True,
                "resolution_path": "alternate_hypothesis",
                "leading_hypothesis_label": "A",
                "recommended_action": "compare_hypotheses_a",
            },
        ),
    )

    assert "Call out this local tension before drawing conclusions:" in response["findings"][1]
    assert (
        "Keep the first conclusion centered on the leading hypothesis (A) while keeping the competing explanation explicit."
        in response["findings"]
    )


def test_honesty_and_uncertainty_language_do_not_depend_on_style() -> None:
    conversational = ConversationPolicy.insufficient_evidence_note(subject_label="answer", anchor="archive run")
    direct_profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    conversational_profile = InteractionProfile.default()

    assert "settle this answer with confidence" in conversational
    assert InteractionStylePolicy.is_direct(direct_profile) is True
    assert InteractionStylePolicy.is_conversational(conversational_profile) is True


def test_warm_conversational_research_response_keeps_explicit_uncertainty() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {"confidence": 0.65, "reason": "Research cues matched with score 3"},
            }
        ),
        reasoning_frame_assembly=ReasoningFrameAssembly(
            frame_type="retrieve-and-summarize",
            problem_interpretation="Interpret this as a summary request.",
            local_context_summary=None,
            grounded_interpretation="Grounded interpretation.",
            working_hypothesis=None,
            interaction_profile={"interaction_style": "conversational", "reasoning_depth": "normal"},
            local_context_assessment="partial",
            grounding_strength="medium",
            route_quality="supported",
            reasoning_frame={"primary_anchor": "archive structure"},
        ),
    )

    assert response["uncertainty_note"] is not None
    assert "settle this answer with confidence" in response["uncertainty_note"]
    assert response["summary"] in {
        "Here’s a first pass, with the assumptions kept visible.",
        "Here’s a useful first pass using the best current assumptions.",
    }


def test_anti_roleplay_guardrails_are_profile_independent() -> None:
    conversational_notes = AntiRoleplayPolicy.guardrail_notes()
    direct_notes = AntiRoleplayPolicy.guardrail_notes()

    assert conversational_notes == direct_notes
    assert "Do not enter explicit roleplay" in conversational_notes[0]
    assert "Do not imply personal knowledge" in conversational_notes[1]
    assert "do not intensify emotional bonding" in conversational_notes[2].lower()
    assert "Do not present guesses" in conversational_notes[3]
    assert "grounded research-partner tone is allowed" in conversational_notes[4]


def test_emotional_support_limits_allow_warmth_without_dependency() -> None:
    limits = AntiRoleplayPolicy.emotional_support_limits()
    note = ConversationPolicy.grounded_emotional_acknowledgment(feeling_label="heavy")

    assert "acknowledge feelings directly" in limits[0].lower()
    assert "escalating emotional attachment" in limits[1].lower()
    assert "feel heavy" in note
    assert "what is uncertain" in note


def test_conversation_policy_names_emotional_context_without_escalation() -> None:
    assert any("emotional context" in rule for rule in ConversationPolicy.UNIVERSAL_RULES)
    assert any("dependence" in rule for rule in ConversationPolicy.EMOTIONAL_CONTEXT_RULES)


def test_research_engine_softens_recommendation_when_grounding_is_medium() -> None:
    response = ResearchEngine().respond(
        "summarize the current archive structure",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {"confidence": 0.65, "reason": "Research cues matched with score 3"},
            }
        ),
    )

    assert response["grounding_strength"] == "medium"
    assert response["closing_strategy"] == "validate_before_commit"
    assert response["summary"] in {
        "Here’s a first pass, with the assumptions kept visible.",
        "Here’s a useful first pass using the best current assumptions.",
    }
    assert response.get("working_hypothesis") is None
    assert "settle this answer with confidence" in response["uncertainty_note"]
    assert response["findings"][1] == "Keep the first conclusion close to the strongest local signal and avoid broad extrapolation."
    assert "Keep one local validation step before acting on it." in response["recommendation"]


def test_planner_mentions_route_caution_when_route_is_bias_driven() -> None:
    response = Planner().respond(
        "review the migration summary",
        kind="planning.migration",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.7,
                    "reason": "Ambiguous prompt was biased toward the active session thread",
                    "source": "active_thread_bias",
                    "caution": "Route selection relied on active-thread bias because planning and research cues were close.",
                }
            }
        ),
    )

    assert any(item.startswith("Route caution:") for item in response["evidence"])
    assert "route choice was somewhat tentative" in response["next_action"]


def test_planner_softens_strategy_when_route_is_ambiguous() -> None:
    response = Planner().respond(
        "review the migration summary",
        kind="planning.migration",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.82,
                    "reason": "Planning cues narrowly outranked research cues",
                    "strength": "high",
                    "caution": "Route selection remained close because planning and research cues were both plausible.",
                    "ambiguity": {
                        "reason": "Top planning and research candidates were close in rank.",
                    },
                },
                "top_matches": [
                    {
                        "score": 5,
                        "record": {
                            "tool_id": "anh",
                            "capability": "spectral_dip_scan",
                            "summary": "Migration summary validation run",
                        },
                    }
                ],
            }
        ),
    )

    assert response["grounding_strength"] == "high"
    assert response["confidence_posture"] == "supported"
    assert response["closing_strategy"] == "cautious_execution"
    assert (
        "Keep the opening milestone narrow until the prompt intent is clearer, because the route decision was close."
        in response["steps"]
    )
    assert "Validate it against the closest local evidence first" not in response["next_action"]
    assert "Keep the route caution in mind while validating the first step." in response["next_action"]


def test_research_engine_mentions_route_caution_when_route_falls_back() -> None:
    response = ResearchEngine().respond(
        "hello there",
        kind="research.general",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.35,
                    "reason": "Defaulted to research mode because no stronger planning or tool cues matched",
                    "source": "fallback",
                    "caution": "Route selection fell back to a general research response because stronger intent cues were not found.",
                }
            }
        ),
    )

    assert response["summary"] in {
        "Here’s a first pass, with the assumptions kept visible.",
        "Here’s a useful first pass using the best current assumptions.",
    }
    assert response["confidence_posture"] == "tentative"
    assert response["closing_strategy"] == "exploratory_validation"
    assert response["grounded_interpretation"] == "Local evidence for this research response is thin, so treat the result as exploratory."
    assert any(item.startswith("Route caution:") for item in response["evidence"])
    assert "Treat this as provisional" in response["recommendation"]


def test_research_engine_softens_strategy_when_route_is_ambiguous() -> None:
    response = ResearchEngine().respond(
        "summarize the migration summary",
        kind="research.summary",
        context=AssistantContext.from_mapping(
            {
                "route": {
                    "confidence": 0.82,
                    "reason": "Research cues narrowly outranked planning cues",
                    "strength": "high",
                    "caution": "Route selection remained close because research and planning cues were both plausible.",
                    "ambiguity": {
                        "reason": "Top research and planning candidates were close in rank.",
                    },
                },
                "top_interaction_matches": [
                    {
                        "score": 5,
                        "record": {
                            "prompt_view": {
                                "canonical_prompt": "migration summary",
                                "original_prompt": "migration summary",
                                "rewritten": False,
                            }
                        },
                    }
                ],
            }
        ),
    )

    assert response["grounding_strength"] == "high"
    assert response["confidence_posture"] == "supported"
    assert response["closing_strategy"] == "cautious_execution"
    assert (
        "Keep the first conclusion narrow until the prompt intent is clearer, because the route decision was close."
        in response["findings"]
    )
    assert "Keep the route caution in mind while checking the first conclusion." in response["recommendation"]
    assert "There is not enough evidence to confidently settle this answer yet, because the prompt intent is still ambiguous" not in str(
        response.get("uncertainty_note")
    )


def test_domain_route_converts_to_route_metadata() -> None:
    route = DomainRoute(
        mode="planning",
        kind="planning.migration",
        normalized_prompt="create a migration plan for lumen",
        confidence=0.82,
        reason="Planning cues matched with score 4",
    )

    metadata = route.to_metadata()

    assert isinstance(metadata, RouteMetadata)
    assert metadata.to_dict() == {
        "confidence": 0.82,
        "reason": "Planning cues matched with score 4",
        "source": "classifier",
        "strength": "medium",
    }


def test_tool_assistant_response_serializes_tool_execution_contract() -> None:
    payload = ToolAssistantResponse(
        mode="tool",
        kind="tool.command_alias",
        summary="GA Local Analysis Kit run completed",
        route=RouteMetadata(confidence=0.95, reason="Matched manifest-declared tool alias 'run anh'"),
        tool_execution=ToolExecutionDetails(
            tool_id="anh",
            capability="spectral_dip_scan",
            input_path=Path("data/examples/cf4_ga_cone_template.csv"),
            params={"h0": 70},
        ),
        tool_result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
        ),
    ).to_dict()

    assert payload["schema_type"] == "assistant_response"
    assert payload["mode"] == "tool"
    assert payload["route"]["confidence"] == 0.95
    assert payload["tool_execution"]["tool_id"] == "anh"
    assert payload["tool_execution"]["input_path"].endswith("cf4_ga_cone_template.csv")
    assert payload["tool_result"].summary == "GA Local Analysis Kit run completed"


