from __future__ import annotations

import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.evaluation.decision_evaluation import DecisionEvaluation
from lumen.labeling.dataset_exporter import DatasetExporter
from lumen.labeling.labeling_support import LabelingSupport


def _record() -> dict[str, object]:
    return {
        "session_id": "default",
        "interaction_path": "data/interactions/default/record.json",
        "created_at": "2026-04-08T00:00:00+00:00",
        "prompt": "create a migration plan for lumen routing",
        "mode": "planning",
        "kind": "planning.migration",
        "summary": "Roadmap for routing work.",
        "route": {"mode": "planning", "kind": "planning.migration"},
        "response": {
            "reasoning_state": {
                "confidence": 0.82,
                "confidence_tier": "high",
                "intent_domain": "planning_strategy",
                "route_decision": {"mode": "planning", "kind": "planning.migration"},
                "memory_context_used": [{"label": "routing roadmap", "source": "research_note"}],
            }
        },
        "trainability_trace": {
            "available_training_surfaces": [
                "intent_domain_classification",
                "route_recommendation_support",
            ],
            "deterministic_surfaces": ["system_invariants"],
            "route_recommendation_support": {
                "mode": "planning",
                "kind": "planning.migration",
                "route_confidence": 0.81,
                "route_status": "grounded",
                "support_status": "supported",
            },
            "intent_domain_classification": {
                "intent_domain": "planning_strategy",
                "intent_domain_confidence": 0.86,
                "route_mode": "planning",
            },
            "memory_relevance_ranking": {
                "selected_count": 1,
                "rejected_count": 0,
                "selected_labels": ["routing roadmap"],
                "memory_context_used_labels": ["routing roadmap"],
                "memory_context_used_count": 1,
            },
            "tool_use_decision_support": {
                "should_use_tool": False,
                "selected_tool": "workspace",
                "execution_attempted": False,
            },
            "response_style_selection": {"intent_domain": "planning_strategy"},
            "confidence_calibration_support": {
                "confidence_tier": "high",
                "confidence_score": 0.82,
                "confidence_posture": "supported",
                "support_status": "supported",
                "route_status": "grounded",
                "memory_signal_present": True,
                "tool_verified": False,
            },
            "supervised_decision_support": {
                "enabled": True,
                "recommended_surfaces": ["intent_domain_classification"],
                "applied_surfaces": [],
                "deterministic_authority_preserved": True,
            },
        },
        "supervised_support_trace": {
            "enabled": True,
            "recommendations": {
                "intent_domain_classification": {
                    "surface": "intent_domain_classification",
                    "recommended_label": "planning_strategy",
                    "confidence": 0.88,
                }
            },
            "applied_surfaces": [],
            "deterministic_authority_preserved": True,
        },
    }


def test_labeling_support_builds_provenance_rich_examples() -> None:
    evaluation = DecisionEvaluation().evaluate_record(_record())

    examples = LabelingSupport().examples_from_evaluation(record=_record(), evaluation=evaluation)

    assert examples
    assert examples[0].label_source == "programmatic_evaluation"
    assert examples[0].trainable is True
    assert examples[0].provenance["evaluation_overall_judgment"] == evaluation.overall_judgment
    assert examples[0].provenance["source_message_id"] is not None
    assert examples[0].provenance["evaluation_surface"] is not None
    assert "available_training_surfaces" in examples[0].metadata


def test_dataset_exporter_writes_local_dataset_file(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    exporter = DatasetExporter(settings)
    evaluation = DecisionEvaluation().evaluate_record(_record())
    examples = LabelingSupport().examples_from_evaluation(record=_record(), evaluation=evaluation)

    result = exporter.export_examples(examples=examples, session_id="default", project_id="lumen")

    dataset_path = Path(result.dataset_path)
    manifest_path = Path(result.manifest_path)
    assert dataset_path.exists()
    assert manifest_path.exists()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_type"] == "labeled_example_dataset"
    assert payload["example_count"] == len(examples)
    assert payload["project_id"] == "lumen"
    assert payload["export_batch_id"] == result.export_batch_id
    assert payload["examples"][0]["split_assignment"] in {"train", "validation", "test"}
    assert payload["examples"][0]["label_category"]
    assert manifest["schema_type"] == "labeled_example_dataset_manifest"
    assert manifest["split_counts"] == result.split_counts


def test_dataset_exporter_assigns_deterministic_splits(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    exporter = DatasetExporter(settings)
    evaluation = DecisionEvaluation().evaluate_record(_record())
    examples = LabelingSupport().examples_from_evaluation(record=_record(), evaluation=evaluation)

    first = exporter.export_examples(examples=examples, session_id="default")
    second = exporter.export_examples(examples=examples, session_id="default")

    first_payload = json.loads(Path(first.dataset_path).read_text(encoding="utf-8"))
    second_payload = json.loads(Path(second.dataset_path).read_text(encoding="utf-8"))

    first_splits = {item["example_id"]: item["split_assignment"] for item in first_payload["examples"]}
    second_splits = {item["example_id"]: item["split_assignment"] for item in second_payload["examples"]}

    assert first_splits == second_splits
