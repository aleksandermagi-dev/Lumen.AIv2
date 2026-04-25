from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.evidence_builder import EvidenceBuilder


def test_evidence_builder_prioritizes_route_and_local_matches() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Planning cues matched with score 5",
                "source": "heuristic_planning",
                "evidence": [{"label": "planning_score", "detail": "planning=5, research=1"}],
            },
            "active_thread": {
                "prompt": "create a migration plan for lumen",
                "objective": "Plan work for: create a migration plan for lumen",
            },
            "matched_record_count": 2,
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
            "top_interaction_matches": [
                {
                    "score": 4,
                    "record": {
                        "prompt_view": {
                            "canonical_prompt": "what about the migration plan for lumen",
                            "original_prompt": "what about that",
                            "rewritten": True,
                        }
                    },
                }
            ],
        }
    )

    evidence = builder.build(mode="planning", context=context)

    assert evidence[0].startswith("Routing selected planning because")
    assert any(item.startswith("Routing evidence:") for item in evidence)
    assert any(item.startswith("Closest archive match: anh/spectral_dip_scan") for item in evidence)
    assert any("Closest prior session prompt: what about the migration plan for lumen" in item for item in evidence)


def test_evidence_builder_dedupes_and_limits_results() -> None:
    builder = EvidenceBuilder(limit=3)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Research cues matched with score 4",
                "source": "heuristic_research",
                "evidence": [{"label": "research_score", "detail": "research=4"}],
            },
            "active_thread": {
                "prompt": "summarize the current archive structure",
                "thread_summary": "summarize the current archive structure",
            },
            "status_counts": {"ok": 2},
            "record_count": 3,
        }
    )

    evidence = builder.build(mode="research", context=context)

    assert len(evidence) == 3
    assert evidence[0].startswith("Routing selected research because")


def test_evidence_builder_prioritizes_strong_retrieved_matches_over_thread_context() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Research cues matched with score 4",
                "source": "heuristic_research",
                "evidence": [{"label": "research_score", "detail": "research=4"}],
            },
            "active_thread": {
                "prompt": "summarize the current archive structure",
                "objective": "Understand the current archive structure",
            },
            "top_matches": [
                {
                    "score": 7,
                    "record": {
                        "tool_id": "anh",
                        "capability": "spectral_dip_scan",
                        "summary": "Great Attractor confirmation candidate",
                    },
                }
            ],
            "top_interaction_matches": [
                {
                    "score": 6,
                    "record": {
                        "prompt_view": {
                            "canonical_prompt": "what about the migration plan for lumen",
                            "original_prompt": "what about that",
                            "rewritten": True,
                        }
                    },
                }
            ],
        }
    )

    evidence = builder.build(mode="research", context=context)

    assert any(item.startswith("Closest archive match:") for item in evidence)
    assert any(item.startswith("Closest prior session prompt:") for item in evidence)
    assert any(item == "Current active prompt: summarize the current archive structure." for item in evidence)


def test_evidence_builder_surfaces_semantic_archive_alignment() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Research cues matched with score 4",
                "source": "heuristic_research",
            },
            "top_matches": [
                {
                    "score": 6,
                    "matched_fields": ["semantic"],
                    "record": {
                        "tool_id": "anh",
                        "capability": "spectral_dip_scan",
                        "run_id": "run_2026_03_16_213045",
                        "target_label": "GA Local Analysis Kit",
                        "result_quality": "scientific_output_present",
                        "summary": "Great Attractor confirmation candidate",
                    },
                }
            ],
        }
    )

    evidence = builder.build(mode="research", context=context)
    summary = builder.summarize_local_context(context=context)
    grounding_strength = builder.grounding_strength(context=context)

    assert any(item.startswith("Closest archive match (semantic): anh/spectral_dip_scan [GA Local Analysis Kit]") for item in evidence)
    assert "run_2026_03_16_213045" in summary
    assert grounding_strength == "medium"


def test_evidence_builder_surfaces_semantic_interaction_alignment() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Planning cues matched with score 4",
                "source": "heuristic_planning",
            },
            "top_interaction_matches": [
                {
                    "score": 6,
                    "matched_fields": ["semantic"],
                    "record": {
                        "prompt_view": {
                            "canonical_prompt": "create a migration plan for lumen routing",
                            "original_prompt": "create a migration plan for lumen routing",
                            "rewritten": False,
                        }
                    },
                }
            ],
        }
    )

    evidence = builder.build(mode="planning", context=context)
    summary = builder.summarize_local_context(context=context)
    grounding_strength = builder.grounding_strength(context=context)

    assert any(
        item.startswith("Closest prior session prompt (semantic): create a migration plan for lumen routing.")
        for item in evidence
    )
    assert summary == "Closest semantically aligned prior session prompt: create a migration plan for lumen routing"
    assert grounding_strength == "medium"


def test_evidence_builder_prefers_cross_source_semantic_coherence() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Planning cues matched with score 4",
                "source": "heuristic_planning",
                "confidence": 0.72,
            },
            "top_matches": [
                {
                    "score": 6,
                    "matched_fields": ["semantic"],
                    "record": {
                        "tool_id": "anh",
                        "capability": "spectral_dip_scan",
                        "summary": "Lumen routing migration confirmation run",
                    },
                }
            ],
            "top_interaction_matches": [
                {
                    "score": 6,
                    "matched_fields": ["semantic"],
                    "record": {
                        "prompt_view": {
                            "canonical_prompt": "create a routing migration plan for lumen",
                            "original_prompt": "create a routing migration plan for lumen",
                            "rewritten": False,
                        }
                    },
                }
            ],
        }
    )

    evidence = builder.build(mode="planning", context=context)
    frame = builder.build_reasoning_frame(context=context)
    grounding_strength = builder.grounding_strength(context=context)

    assert any(
        item.startswith("Archive and prior session context are semantically coherent around")
        for item in evidence
    )
    assert "lumen" in frame["coherence_topic"]
    assert "migration" in frame["coherence_topic"]
    assert "routing" in frame["coherence_topic"]
    assert grounding_strength == "high"


def test_evidence_builder_surfaces_same_capability_target_trend() -> None:
    builder = EvidenceBuilder(limit=6)
    context = AssistantContext.from_mapping(
        {
            "route": {
                "reason": "Research cues matched with score 4",
                "source": "heuristic_research",
            },
            "top_matches": [
                {
                    "score": 6,
                    "record": {
                        "tool_id": "anh",
                        "capability": "spectral_dip_scan",
                        "run_id": "run_2026_03_16_213045",
                        "target_label": "GA Local Analysis Kit",
                        "result_quality": "scientific_output_present",
                        "summary": "Great Attractor confirmation candidate",
                    },
                }
            ],
            "archive_target_comparison": {
                "target_label": "GA Local Analysis Kit",
                "run_count": 3,
                "trend_summary": "latest run remains scientific output present; trend is steady",
            },
        }
    )

    evidence = builder.build(mode="research", context=context)
    summary = builder.summarize_local_context(context=context)

    assert any(
        item.startswith("Same-capability archive trend for GA Local Analysis Kit:")
        for item in evidence
    )
    assert "Same-capability trend for GA Local Analysis Kit:" in summary

