from pathlib import Path
from dataclasses import FrozenInstanceError

from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult, RetrievedMemory
from lumen.reasoning.pipeline_models import RouteAuthorityDecision
from lumen.runtime_authority import RUNTIME_ARCHITECTURE_AUTHORITY_DOC


def test_runtime_architecture_authority_doc_exists() -> None:
    assert Path(RUNTIME_ARCHITECTURE_AUTHORITY_DOC).exists()


def test_memory_retrieval_result_projects_to_advisory_context_without_route_fields() -> None:
    result = MemoryRetrievalResult(
        query="return to the migration plan",
        selected=[
            RetrievedMemory(
                source="active_thread",
                memory_kind="thread",
                label="Migration plan",
                summary="Planning response for lumen migration",
                relevance=0.91,
                metadata={"topic": "migration"},
            )
        ],
        memory_reply_hint="I can pick that thread back up.",
        project_return_prompt=True,
        diagnostics={"reason": "selected"},
    )

    advisory = result.to_advisory_context().to_dict()

    assert advisory["query"] == "return to the migration plan"
    assert advisory["project_return_prompt"] is True
    assert "mode" not in advisory
    assert "kind" not in advisory
    assert "route" not in advisory


def test_prompt_understanding_router_view_is_available_from_canonical_nlu() -> None:
    understanding = PromptNLU().analyze("go deeper")

    router_view = understanding.router_view().to_dict()

    assert router_view["dominant_intent"] == understanding.intent.label
    assert router_view["canonical_text"] == understanding.canonical_text
    assert "follow_up_shorthand" in router_view["structure_fragmentation_markers"]


def test_route_authority_decision_is_immutable() -> None:
    decision = RouteAuthorityDecision(
        mode="research",
        kind="research.summary",
        normalized_prompt="what is voltage",
        confidence=0.82,
        reason="summary cue",
        source="explicit_summary",
    )

    try:
        decision.mode = "planning"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("RouteAuthorityDecision should be immutable")
