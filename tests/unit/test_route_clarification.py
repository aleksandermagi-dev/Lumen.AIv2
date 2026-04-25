from lumen.routing.domain_router import DomainRoute
from lumen.routing.route_clarification import RouteClarificationPolicy


def test_route_clarification_policy_matches_base_heuristic_threshold() -> None:
    route = DomainRoute(
        mode="planning",
        kind="planning.architecture",
        normalized_prompt="review the migration summary",
        confidence=0.77,
        reason="Planning cues narrowly outranked research cues",
        source="heuristic_planning",
        decision_summary={
            "ambiguous": True,
            "alternatives": [],
        },
    )

    assert RouteClarificationPolicy.base_should_clarify(route) is True


def test_route_clarification_policy_marks_active_thread_bias_as_immediate_base_clarify() -> None:
    route = DomainRoute(
        mode="planning",
        kind="planning.migration",
        normalized_prompt="review the migration summary",
        confidence=0.9,
        reason="Ambiguous prompt was biased toward the active session thread",
        source="active_thread_bias",
        decision_summary={
            "ambiguous": True,
            "alternatives": [],
        },
    )

    assert RouteClarificationPolicy.base_should_clarify(route) is True
    assert RouteClarificationPolicy.in_adaptive_scope(route) is False


def test_route_clarification_policy_selects_expected_adaptive_thresholds() -> None:
    nlu_threshold = RouteClarificationPolicy.select_adaptive_threshold(
        nlu_uncertainty_high=True,
        retrieval_semantic_bias_high=False,
    )
    retrieval_threshold = RouteClarificationPolicy.select_adaptive_threshold(
        nlu_uncertainty_high=False,
        retrieval_semantic_bias_high=True,
    )
    default_threshold = RouteClarificationPolicy.select_adaptive_threshold(
        nlu_uncertainty_high=False,
        retrieval_semantic_bias_high=False,
    )

    assert nlu_threshold.trigger == "nlu_uncertainty"
    assert nlu_threshold.threshold == 0.88
    assert retrieval_threshold.trigger == "retrieval_semantic_bias"
    assert retrieval_threshold.threshold == 0.86
    assert default_threshold.trigger == "adaptive_threshold"
    assert default_threshold.threshold == 0.84
