from lumen.app.context_policy import ContextPolicy
from lumen.app.settings import AppSettings
from lumen.retrieval.context_models import (
    CompactArchiveContextRecord,
    CompactInteractionContextRecord,
)


def test_context_policy_builds_from_settings(tmp_path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)

    policy = ContextPolicy.from_settings(settings)

    assert policy.context_match_limit == settings.context_match_limit
    assert policy.context_prompt_max_length == settings.context_prompt_max_length
    assert policy.context_summary_max_length == settings.context_summary_max_length
    assert policy.session_objective_max_length == settings.session_objective_max_length
    assert policy.session_thread_summary_max_length == settings.session_thread_summary_max_length


def test_context_policy_compacts_records_and_truncates_text() -> None:
    policy = ContextPolicy(
        context_match_limit=2,
        context_prompt_max_length=20,
        context_summary_max_length=24,
        session_objective_max_length=30,
        session_thread_summary_max_length=40,
    )

    archive_record = policy.compact_archive_context_record(
        {
            "session_id": "default",
            "tool_id": "anh",
            "capability": "spectral_dip_scan",
            "status": "ok",
            "run_id": "run_2026_03_16_213045",
            "target_label": "GA Local Analysis Kit",
            "result_quality": "scientific_output_present",
            "summary": "Great Attractor confirmation candidate with a very long summary",
            "created_at": "2026-03-15T00:00:00+00:00",
            "archive_path": "archive.json",
        }
    )
    interaction_record = policy.compact_interaction_context_record(
        {
            "session_id": "default",
            "prompt": "what about the migration plan for lumen after phase two",
            "resolved_prompt": "what about the migration plan for lumen after phase two",
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Planning response for: what about the migration plan for lumen after phase two",
            "created_at": "2026-03-15T00:00:00+00:00",
            "interaction_path": "interaction.json",
            "resolution_strategy": "reference_follow_up",
            "resolution_reason": "Expanded reference-style follow-up",
            "dominant_intent": "planning",
            "extracted_entities": [{"label": "domain", "value": "migration", "confidence": 0.8}],
            "prompt_view": {
                "canonical_prompt": "what about the migration plan for lumen after phase two",
                "original_prompt": "what about that",
                "resolved_prompt": "what about the migration plan for lumen after phase two",
                "rewritten": True,
            },
        }
    )

    assert isinstance(archive_record, CompactArchiveContextRecord)
    assert isinstance(interaction_record, CompactInteractionContextRecord)
    assert len(archive_record.summary) <= 24
    assert archive_record.summary.endswith("...")
    assert archive_record.run_id == "run_2026_03_16_213045"
    assert archive_record.target_label == "GA Local Analysis Kit"
    assert archive_record.result_quality == "scientific_output_present"
    assert len(interaction_record.prompt) <= 20
    assert len(interaction_record.summary) <= 24
    assert len(interaction_record.prompt_view.canonical_prompt) <= 20
    assert interaction_record.dominant_intent == "planning"
    assert interaction_record.extracted_entities == ("migration",)


def test_compact_context_models_serialize_to_expected_dicts() -> None:
    policy = ContextPolicy(
        context_match_limit=2,
        context_prompt_max_length=20,
        context_summary_max_length=24,
        session_objective_max_length=30,
        session_thread_summary_max_length=40,
    )

    archive_payload = policy.compact_archive_context_record(
        {
            "session_id": "default",
            "tool_id": "anh",
            "capability": "spectral_dip_scan",
            "status": "ok",
            "run_id": "run_2026_03_16_213045",
            "target_label": "GA Local Analysis Kit",
            "result_quality": "scientific_output_present",
            "summary": "short summary",
            "created_at": "2026-03-15T00:00:00+00:00",
            "archive_path": "archive.json",
        }
    ).to_dict()
    interaction_payload = policy.compact_interaction_context_record(
        {
            "session_id": "default",
            "prompt": "what about that",
            "resolved_prompt": "what about the migration plan for lumen",
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Planning response for: what about the migration plan for lumen",
            "created_at": "2026-03-15T00:00:00+00:00",
            "interaction_path": "interaction.json",
            "resolution_strategy": "reference_follow_up",
            "resolution_reason": "Expanded reference-style follow-up",
            "dominant_intent": "planning",
            "extracted_entities": [{"label": "domain", "value": "migration", "confidence": 0.8}],
            "prompt_view": {
                "canonical_prompt": "what about the migration plan for lumen",
                "original_prompt": "what about that",
                "resolved_prompt": "what about the migration plan for lumen",
                "rewritten": True,
            },
        }
    ).to_dict()

    assert archive_payload["tool_id"] == "anh"
    assert archive_payload["run_id"] == "run_2026_03_16_213045"
    assert archive_payload["target_label"] == "GA Local Analysis Kit"
    assert interaction_payload["prompt_view"]["rewritten"] is True
    assert interaction_payload["dominant_intent"] == "planning"
    assert interaction_payload["extracted_entities"] == ["migration"]


def test_compact_interaction_context_record_preserves_nlu_fields() -> None:
    policy = ContextPolicy(
        context_match_limit=2,
        context_prompt_max_length=32,
        context_summary_max_length=40,
        session_objective_max_length=30,
        session_thread_summary_max_length=40,
    )

    record = policy.compact_interaction_context_record(
        {
            "session_id": "default",
            "prompt": "create a migration plan for lumen",
            "resolved_prompt": "create a migration plan for lumen",
            "mode": "planning",
            "kind": "planning.migration",
            "summary": "Grounded planning response for: create a migration plan for lumen",
            "created_at": "2026-03-15T00:00:00+00:00",
            "interaction_path": "interaction.json",
            "dominant_intent": "planning",
            "extracted_entities": [
                {"label": "domain", "value": "migration", "confidence": 0.8},
            ],
            "prompt_view": {
                "canonical_prompt": "create a migration plan for lumen",
                "original_prompt": None,
                "resolved_prompt": "create a migration plan for lumen",
                "rewritten": False,
            },
        }
    )

    payload = record.to_dict()

    assert record.dominant_intent == "planning"
    assert record.extracted_entities == ("migration",)
    assert payload["dominant_intent"] == "planning"
    assert payload["extracted_entities"] == ["migration"]

