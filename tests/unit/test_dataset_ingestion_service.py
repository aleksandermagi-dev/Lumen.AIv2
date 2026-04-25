from __future__ import annotations

import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.schemas.interaction_schema import InteractionRecordSchema
from lumen.services.dataset_curation_service import DatasetCurationService
from lumen.services.dataset_ingestion_service import DatasetIngestionService


def _service(tmp_path: Path) -> tuple[DatasetIngestionService, DatasetCurationService, PersistenceManager, AppSettings]:
    settings = AppSettings.from_repo_root(tmp_path)
    persistence = PersistenceManager(settings)
    persistence.bootstrap()
    return DatasetIngestionService(settings, persistence), DatasetCurationService(settings, persistence), persistence, settings


def test_import_jsonl_classification_examples_persists_examples_and_labels(tmp_path: Path) -> None:
    service, _, _, _ = _service(tmp_path)
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": "route this to planning",
                        "label_category": "intent_domain",
                        "label_value": "planning_strategy",
                        "correction_label": "planning_strategy_refined",
                        "labels": [
                            {
                                "label_role": "canonical_label",
                                "label_value": "planning_strategy",
                                "label_category": "intent_domain",
                                "is_canonical": True,
                                "reviewer": "tester",
                                "reason": "verified",
                            }
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    result = service.import_dataset(
        dataset_name="routing-train",
        source_format="jsonl",
        dataset_kind="classification_examples",
        source_path=dataset,
    )

    assert result["status"] == "ok"
    assert result["example_count"] == 1
    import_run = result["import_run"]
    examples = service.list_dataset_examples(import_run_id=str(import_run["id"]))
    labels = service.list_dataset_labels(dataset_example_id=str(examples[0]["id"]))

    assert examples[0]["example_type"] == "classification"
    assert examples[0]["label_value"] == "planning_strategy"
    assert examples[0]["split_assignment"] in {"train", "validation", "test"}
    assert labels
    assert {label["label_role"] for label in labels} >= {"target_label", "correction_label", "canonical_label"}


def test_import_csv_instruction_response_uses_explicit_mapping(tmp_path: Path) -> None:
    service, _, _, _ = _service(tmp_path)
    dataset = tmp_path / "pairs.csv"
    dataset.write_text("instruction,response\nExplain entropy,Entropy is energy dispersal.\n", encoding="utf-8")

    result = service.import_dataset(
        dataset_name="entropy-ir",
        source_format="csv",
        dataset_kind="instruction_response",
        source_path=dataset,
        csv_mapping={"instruction_column": "instruction", "response_column": "response"},
    )

    examples = service.list_dataset_examples(import_run_id=str(result["import_run"]["id"]))
    assert result["example_count"] == 1
    assert examples[0]["example_type"] == "instruction_response"
    assert examples[0]["target_text"] == "Entropy is energy dispersal."


def test_import_runtime_trainability_examples_derives_from_sqlite_interactions(tmp_path: Path) -> None:
    service, _, persistence, _ = _service(tmp_path)
    persistence.record_interaction(
        session_id="default",
        prompt="run anh",
        response={"mode": "tool", "summary": "ANH analyzed 1 file.", "kind": "tool.command_alias"},
        record=InteractionRecordSchema.normalize(
            {
                "schema_type": "interaction_record",
                "schema_version": "5",
                "session_id": "default",
                "prompt": "run anh",
                "mode": "tool",
                "kind": "tool.command_alias",
                "summary": "ANH analyzed 1 file.",
                "created_at": "2026-04-08T00:00:00+00:00",
                "response": {"summary": "ANH analyzed 1 file."},
                "trainability_trace": {
                    "available_training_surfaces": ["tool_use_decision_support"],
                    "deterministic_surfaces": ["system_invariants"],
                },
            }
        ),
        interaction_path=str(tmp_path / "data" / "interactions" / "default" / "20260408T000000Z.json"),
    )

    result = service.import_runtime_examples(
        dataset_name="runtime-decision-seed",
        import_strategy="derived_trainability",
        session_id="default",
    )

    examples = service.list_dataset_examples(import_run_id=str(result["import_run"]["id"]))
    assert result["example_count"] >= 1
    assert examples[0]["source_trace_id"] is not None
    assert examples[0]["label_source"] == "programmatic_evaluation"
    assert examples[0]["label_category"] == "interaction_outcome"


def test_import_runtime_instruction_response_keeps_rows_out_of_messages_table(tmp_path: Path) -> None:
    service, _, persistence, _ = _service(tmp_path)
    before_messages = len(persistence.messages.list_by_session("default"))
    persistence.record_interaction(
        session_id="default",
        prompt="explain entropy",
        response={"mode": "research", "summary": "Entropy is energy dispersal.", "kind": "research.summary"},
        record=InteractionRecordSchema.normalize(
            {
                "schema_type": "interaction_record",
                "schema_version": "5",
                "session_id": "default",
                "prompt": "explain entropy",
                "mode": "research",
                "kind": "research.summary",
                "summary": "Entropy is energy dispersal.",
                "created_at": "2026-04-08T00:00:00+00:00",
                "response": {"summary": "Entropy is energy dispersal."},
            }
        ),
        interaction_path=str(tmp_path / "data" / "interactions" / "default" / "20260408T000000Z.json"),
    )

    result = service.import_runtime_examples(
        dataset_name="runtime-ir-seed",
        import_strategy="derived_instruction_response",
        session_id="default",
    )

    after_messages = len(persistence.messages.list_by_session("default"))
    examples = service.list_dataset_examples(import_run_id=str(result["import_run"]["id"]))

    assert result["example_count"] >= 1
    assert after_messages == before_messages + 2
    assert examples[0]["example_type"] == "instruction_response"
    assert examples[0]["source_message_id"] is not None


def test_dataset_review_and_canonical_label_update_persist(tmp_path: Path) -> None:
    ingestion, curation, _, _ = _service(tmp_path)
    dataset = tmp_path / "routing.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "text": "route this to planning",
                "label_category": "interaction_outcome",
                "label_value": "planning",
                "label_source": "programmatic_evaluation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = ingestion.import_dataset(
        dataset_name="lumen_route_v1",
        source_format="jsonl",
        dataset_kind="classification_examples",
        source_path=dataset,
    )
    example_id = str(imported["examples"][0]["id"])

    review = curation.sample_review_batch(dataset_name="lumen_route_v1", limit=5)
    assert review["review_count"] == 1

    label_update = curation.label_dataset_example(
        dataset_example_id=example_id,
        label_role="canonical_label",
        label_value="planning",
        label_category="interaction_outcome",
        reviewer="tester",
        reason="verified",
        is_canonical=True,
    )
    example_update = curation.update_dataset_example(
        example_id=example_id,
        trainable=False,
        ingestion_state="archived",
        review_note="held out for review",
    )

    assert label_update["label"]["label_role"] == "canonical_label"
    assert example_update["example"]["trainable"] is False
    assert example_update["example"]["ingestion_state"] == "archived"


def test_dataset_jsonl_export_supports_canonical_only_and_quality_checks(tmp_path: Path) -> None:
    ingestion, curation, _, settings = _service(tmp_path)
    dataset = tmp_path / "conversation.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "Explain entropy simply.", "response": "Entropy is energy spreading."}),
                json.dumps({"instruction": "Explain entropy simply.", "response": "Entropy is energy spreading."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    imported = ingestion.import_dataset(
        dataset_name="lumen_conversation_v1",
        source_format="jsonl",
        dataset_kind="instruction_response",
        source_path=dataset,
    )
    for example in imported["examples"]:
        curation.label_dataset_example(
            dataset_example_id=str(example["id"]),
            label_role="canonical_label",
            label_value="approved",
            label_category="quality",
            reviewer="tester",
            reason="approved for export",
            is_canonical=True,
            sync_example_fields=False,
        )

    export = curation.export_dataset_jsonl(
        dataset_name="lumen_conversation_v1",
        canonical_only=True,
        output_root=settings.labeled_datasets_root,
    )

    assert export["status"] == "ok"
    assert Path(export["manifest_path"]).exists()
    assert Path(export["summary_path"]).exists()
    assert export["quality_report"]["duplicate_group_count"] >= 1
    assert export["example_count"] == 2


def test_compare_dataset_runs_reports_overlap_and_counts(tmp_path: Path) -> None:
    ingestion, curation, _, _ = _service(tmp_path)
    dataset = tmp_path / "route.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"text": "route to tool", "label_category": "interaction_outcome", "label_value": "tool"}),
                json.dumps({"text": "route to research", "label_category": "interaction_outcome", "label_value": "research"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    left = ingestion.import_dataset(
        dataset_name="lumen_route_v1",
        source_format="jsonl",
        dataset_kind="classification_examples",
        source_path=dataset,
        dataset_version="v1",
    )
    right = ingestion.import_dataset(
        dataset_name="lumen_route_v1",
        source_format="jsonl",
        dataset_kind="classification_examples",
        source_path=dataset,
        dataset_version="v2",
    )

    comparison = curation.compare_dataset_runs(
        left_import_run_id=str(left["import_run"]["id"]),
        right_import_run_id=str(right["import_run"]["id"]),
    )

    assert comparison["status"] == "ok"
    assert comparison["left_run"]["example_count"] == 2
    assert comparison["right_run"]["example_count"] == 2
    assert comparison["left_run"]["dataset_version"] == "v1"
    assert comparison["right_run"]["dataset_version"] == "v2"
    assert comparison["overlap"]["shared_example_signature_count"] >= 2
