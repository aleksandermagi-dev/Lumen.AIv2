from lumen.routing.capability_manager import CapabilityManager
from lumen.routing.domain_router import DomainRouter
from lumen.routing.intent_signals import IntentSignalExtractor
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.tools.registry_types import BundleManifest, CapabilityManifest


def make_router() -> DomainRouter:
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
    return DomainRouter(capability_manager=capability_manager)


def test_domain_router_routes_tool_aliases() -> None:
    router = make_router()

    route = router.route("run anh")

    assert route.mode == "tool"
    assert route.kind == "tool.command_alias"
    assert route.confidence >= 0.9
    assert route.source == "manifest_alias"
    assert route.evidence[0]["label"] == "tool_alias"
    assert route.comparison is not None
    assert route.comparison["candidate"]["source"] == "manifest_alias"
    assert route.decision_summary is not None
    assert route.decision_summary["selected"]["candidate"]["source"] == "manifest_alias"


def test_domain_router_does_not_run_anh_for_concept_question() -> None:
    router = make_router()

    route = router.route("what is ANH")

    assert route.mode == "research"
    assert route.kind == "research.summary"


def test_domain_router_routes_planning_prompts() -> None:
    router = make_router()

    route = router.route("create a migration plan for lumen")
    metadata = route.to_metadata().to_dict()

    assert route.mode == "planning"
    assert route.kind == "planning.migration"
    assert "planning" in route.reason.lower()
    assert route.source == "explicit_planning"
    assert metadata["strength"] == "high"
    assert "caution" not in metadata


def test_intent_signal_extractor_surfaces_nlu_fields() -> None:
    signals = IntentSignalExtractor().extract("create a migration plan for lumen routing")

    assert signals.detected_language == "en"
    assert signals.normalized_topic == "migration plan for lumen routing"
    assert signals.dominant_intent == "planning"
    assert signals.intent_confidence > 0.0
    assert any(entity["value"] == "migration" for entity in signals.extracted_entities)


def test_intent_signal_extractor_uses_entities_to_reinforce_hint_scores() -> None:
    signals = IntentSignalExtractor().extract("review the archive")

    assert signals.research_score >= 2
    assert any(entity["value"] == "archive" for entity in signals.extracted_entities)


def test_domain_router_defaults_to_research() -> None:
    router = make_router()

    route = router.route("summarize the current archive structure")

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.confidence > 0.4
    assert route.source == "explicit_summary"


def test_domain_router_routes_comparison_prompts() -> None:
    router = make_router()

    route = router.route("compare local archive retrieval versus indexed retrieval")

    assert route.mode == "research"
    assert route.kind == "research.comparison"
    assert route.source == "explicit_compare"


def test_domain_router_routes_who_are_you_as_conversation_self_overview() -> None:
    router = make_router()

    route = router.route("who are you")

    assert route.mode == "conversation"
    assert route.kind == "conversation.self_overview"
    assert route.source == "self_overview"


def test_domain_router_keeps_tell_me_about_yourself_out_of_research() -> None:
    router = make_router()

    route = router.route("tell me about yourself")

    assert route.mode == "conversation"
    assert route.kind == "conversation.self_overview"


def test_domain_router_treats_design_architecture_as_explicit_planning() -> None:
    router = make_router()

    route = router.route("design the architecture for lumen routing")

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source == "explicit_planning"


def test_domain_router_uses_structure_reconstruction_for_broad_design_prompt() -> None:
    router = make_router()

    route = router.route("come up with a design for a lumen api workflow")

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"


def test_domain_router_normalizes_fragmented_educational_prompt_into_research() -> None:
    router = make_router()

    route = router.route("black holes like I'm smart but not a physist")

    assert route.mode == "research"
    assert "physicist" in route.normalized_prompt


def test_domain_router_accepts_canonical_prompt_understanding_view() -> None:
    router = make_router()
    understanding = PromptNLU().analyze("create a migration plan for lumen routing")

    route = router.route(understanding)

    assert route.mode == "planning"
    assert route.kind == "planning.migration"


def test_domain_router_keeps_explicit_summary_authoritative_over_conversational_context() -> None:
    router = make_router()

    route = router.route(
        "what is voltage",
        active_thread={
            "mode": "conversation",
            "kind": "conversation.greeting",
            "prompt": "hello lumen",
            "summary": "Hey. What are we looking at?",
        },
    )

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.source == "explicit_summary"


def test_domain_router_inherits_recent_follow_up_route() -> None:
    router = make_router()

    route = router.route(
        "expand that further",
        recent_interactions=[
            {
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "Planning response for: create a migration plan for lumen",
            }
        ],
    )

    assert route.mode == "planning"
    assert route.kind == "planning.migration"
    assert "Follow-up prompt" in route.reason
    assert route.source == "recent_interaction"


def test_domain_router_prefers_active_thread_for_follow_up() -> None:
    router = make_router()

    route = router.route(
        "continue with that",
        recent_interactions=[
            {
                "mode": "research",
                "kind": "research.summary",
                "summary": "Research response for: summarize the current archive structure",
            }
        ],
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing layer",
            "summary": "Planning response for: design the routing layer",
        },
    )

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert "active session thread" in route.reason
    assert route.source == "active_thread"


def test_domain_router_keeps_strong_reference_follow_up_with_active_thread() -> None:
    router = make_router()

    route = router.route(
        "what about that",
        active_thread={
            "mode": "planning",
            "kind": "planning.migration",
            "prompt": "create a migration plan for lumen",
            "summary": "Here’s a workable first pass.",
        },
    )

    assert route.mode == "planning"
    assert route.kind == "planning.migration"
    assert route.source == "active_thread"


def test_domain_router_biases_ambiguous_prompt_toward_active_thread_mode() -> None:
    router = make_router()

    route = router.route(
        "review the migration summary",
        active_thread={
            "mode": "planning",
            "kind": "planning.migration",
            "prompt": "create a migration plan for lumen",
            "summary": "Planning response for: create a migration plan for lumen",
        },
    )
    metadata = route.to_metadata().to_dict()

    assert route.mode == "planning"
    assert route.kind == "planning.migration"
    assert route.source == "active_thread_bias"
    assert "Ambiguous prompt" in route.reason
    assert metadata["strength"] == "medium"
    assert "active-thread bias" in metadata["caution"]


def test_domain_router_can_use_active_topic_continuity_as_secondary_signal() -> None:
    router = make_router()

    route = router.route(
        "architecture constraints",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Planning response for: design the routing architecture",
            "normalized_topic": "routing architecture",
        },
    )
    metadata = route.to_metadata().to_dict()

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source == "active_topic"
    assert "active session topic" in route.reason
    assert metadata["strength"] == "medium"
    assert "active topic continuity" in metadata["caution"]


def test_domain_router_can_use_active_intent_continuity_as_secondary_signal() -> None:
    router = make_router()

    route = router.route(
        "architecture direction",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Planning response for: design the routing architecture",
            "dominant_intent": "planning",
        },
    )
    metadata = route.to_metadata().to_dict()

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source == "active_intent"
    assert "active session intent" in route.reason
    assert metadata["strength"] == "medium"
    assert "active intent continuity" in metadata["caution"]


def test_domain_router_can_use_active_entity_continuity_as_secondary_signal() -> None:
    router = make_router()

    route = router.route(
        "routing constraints",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Planning response for: design the routing architecture",
            "extracted_entities": [
                {"label": "domain", "value": "routing", "confidence": 0.8},
            ],
        },
    )
    metadata = route.to_metadata().to_dict()

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source == "active_entities"
    assert "active session" in route.reason
    assert metadata["strength"] == "medium"
    assert "active entity continuity" in metadata["caution"]


def test_domain_router_keeps_explicit_summary_over_active_thread_bias() -> None:
    router = make_router()

    route = router.route(
        "summarize the migration plan",
        active_thread={
            "mode": "planning",
            "kind": "planning.migration",
            "prompt": "create a migration plan for lumen",
            "summary": "Planning response for: create a migration plan for lumen",
        },
    )

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.source in {"explicit_summary", "heuristic_research"}


def test_domain_router_biases_do_language_toward_planning() -> None:
    router = make_router()

    route = router.route("what should we do about lumen routing next")

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source in {"heuristic_planning", "active_thread_bias"}
    assert any(item["label"] == "action_intent" for item in route.evidence or [])


def test_domain_router_biases_answer_language_toward_research() -> None:
    router = make_router()

    route = router.route("what does lumen routing do")

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.source == "heuristic_research"
    assert any(item["label"] == "answer_intent" for item in route.evidence or [])
    assert route.comparison is not None
    assert route.comparison["intent_weight"] > 0


def test_domain_route_marks_fallback_as_low_strength_with_caution() -> None:
    router = make_router()

    route = router.route("hello there")
    metadata = route.to_metadata().to_dict()

    assert route.source == "fallback"
    assert metadata["strength"] == "low"
    assert "fell back to a general research response" in metadata["caution"]
    assert route.comparison is not None
    assert route.comparison["candidate"]["source"] == "fallback"
    assert "decision_summary" in metadata


def test_domain_router_decays_stale_planning_context_for_explanatory_prompt() -> None:
    router = make_router()

    route = router.route(
        "tell me about black holes",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Planning response for: design the routing architecture",
            "dominant_intent": "planning",
            "normalized_topic": "routing architecture",
        },
    )

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.comparison is not None
    assert route.comparison["intent_weight"] > 0
    assert route.source == "explicit_summary"


def test_domain_router_treats_weak_follow_up_wording_as_fresh_explanatory_prompt() -> None:
    router = make_router()

    route = router.route(
        "also tell me about black holes",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Here’s a workable first pass.",
            "dominant_intent": "planning",
            "normalized_topic": "routing architecture",
        },
    )

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.source in {"explicit_summary", "heuristic_research"}


def test_domain_router_prefers_social_route_for_relational_greeting() -> None:
    router = make_router()

    route = router.route(
        "good to see you too",
        active_thread={
            "mode": "research",
            "kind": "research.summary",
            "prompt": "tell me about black holes",
            "summary": "Here’s the grounded answer.",
            "dominant_intent": "research",
            "normalized_topic": "black holes",
        },
    )

    assert route.mode == "conversation"
    assert route.kind == "conversation.greeting"


def test_domain_router_prefers_social_check_in_over_stale_planning_context() -> None:
    router = make_router()

    route = router.route(
        "how are you?",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Here’s a workable first pass.",
            "dominant_intent": "planning",
            "normalized_topic": "routing architecture",
        },
    )

    assert route.mode == "conversation"
    assert route.kind == "conversation.check_in"


def test_domain_router_prefers_named_explanatory_subject_over_stale_planning_context() -> None:
    router = make_router()

    route = router.route(
        "George Washington",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Here’s a workable first pass.",
            "dominant_intent": "planning",
            "normalized_topic": "routing architecture",
            "extracted_entities": [
                {"label": "domain", "value": "routing", "confidence": 0.8},
            ],
        },
    )

    assert route.mode == "research"
    assert route.kind == "research.summary"
    assert route.source == "heuristic_research"


def test_domain_route_metadata_includes_alternative_candidates() -> None:
    router = make_router()

    route = router.route("review the migration summary")
    metadata = route.to_metadata().to_dict()

    assert "decision_summary" in metadata
    alternatives = metadata["decision_summary"]["alternatives"]
    assert isinstance(alternatives, list)
    assert len(alternatives) >= 1


def test_domain_route_marks_near_tie_as_ambiguous() -> None:
    router = make_router()

    route = router.route(
        "review the migration summary",
        active_thread={
            "mode": "planning",
            "kind": "planning.migration",
            "prompt": "create a migration plan for lumen",
            "summary": "Planning response for: create a migration plan for lumen",
        },
    )
    metadata = route.to_metadata().to_dict()

    assert metadata["ambiguity"]["ambiguous"] is True
    assert "active-thread bias" in metadata["ambiguity"]["reason"]
    assert metadata["caution"] == "Route selection relied on active-thread bias because planning and research cues were close."
    assert route.should_clarify() is True


def test_domain_route_semantic_bonus_can_reflect_intent_confidence() -> None:
    router = make_router()

    route = router.route("summarize the archive routing")

    assert route.comparison is not None
    assert route.comparison["semantic_bonus"] > 0


def test_domain_route_does_not_clarify_explicit_routes() -> None:
    router = make_router()

    route = router.route("create a migration plan for lumen")

    assert route.should_clarify() is False


def test_domain_router_exposes_semantic_bonus_in_route_comparison() -> None:
    router = make_router()

    route = router.route(
        "routing migration architecture",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "normalized_topic": "routing architecture",
            "dominant_intent": "planning",
            "extracted_entities": [
                {"label": "domain", "value": "routing", "confidence": 0.8},
                {"label": "domain", "value": "migration", "confidence": 0.8},
            ],
        },
    )

    assert route.comparison is not None
    assert route.comparison["semantic_bonus"] > 0
    assert route.comparison["normalized_score"] > route.comparison["candidate"]["confidence"]


def test_domain_router_uses_semantic_bonus_to_reinforce_active_thread_route() -> None:
    router = make_router()

    route = router.route(
        "routing migration architecture",
        active_thread={
            "mode": "planning",
            "kind": "planning.architecture",
            "prompt": "design the routing architecture",
            "summary": "Planning response for: design the routing architecture",
            "normalized_topic": "routing architecture",
            "dominant_intent": "planning",
            "extracted_entities": [
                {"label": "domain", "value": "routing", "confidence": 0.8},
                {"label": "domain", "value": "migration", "confidence": 0.8},
            ],
        },
    )

    assert route.mode == "planning"
    assert route.kind == "planning.architecture"
    assert route.source in {"active_topic", "active_entities", "active_intent", "active_thread_bias", "heuristic_planning"}
    assert route.comparison["semantic_bonus"] > 0
    assert route.comparison["normalized_score"] > 0

