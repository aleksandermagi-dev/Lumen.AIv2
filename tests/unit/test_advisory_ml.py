from lumen.reasoning.advisory_ml import AdvisoryMLAdapter, build_advisory_ml_request


def test_advisory_ml_adapter_is_noop_and_advisory_only() -> None:
    adapter = AdvisoryMLAdapter()
    request = build_advisory_ml_request(
        surface="route_recommendation_support",
        session_id="default",
        project_id="lumen",
        message_id="default:msg:assistant",
        context={"mode": "planning"},
        features={"route_confidence": 0.81},
    )

    recommendation = adapter.recommend(request)

    assert recommendation.enabled is False
    assert recommendation.recommended_label is None
    assert recommendation.provenance["authority"] == "advisory_only"
    assert recommendation.provenance["deterministic_authority_preserved"] is True
