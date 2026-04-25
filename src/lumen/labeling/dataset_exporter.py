from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.labeling.labeling_models import DatasetExportResult, LabeledExample
from lumen.labeling.split_assignment import stable_split_assignment


class DatasetExporter:
    """Writes local labeled datasets derived from evaluated interaction traces."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def export_examples(
        self,
        *,
        examples: list[LabeledExample],
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> DatasetExportResult:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        export_batch_id = f"dataset_export_{timestamp}"
        dataset_dir = self.settings.labeled_datasets_root / (session_id or "all_sessions")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / f"labeled_examples_{timestamp}.json"
        manifest_path = dataset_dir / f"labeled_examples_{timestamp}.manifest.json"
        enriched_examples = [
            self._enrich_example(example, export_batch_id=export_batch_id)
            for example in examples
        ]
        split_counts = Counter(example.split_assignment or "unknown" for example in enriched_examples)
        payload = {
            "schema_type": "labeled_example_dataset",
            "schema_version": "1",
            "session_id": session_id,
            "project_id": project_id,
            "export_batch_id": export_batch_id,
            "created_at": datetime.now(UTC).isoformat(),
            "example_count": len(enriched_examples),
            "split_counts": dict(split_counts),
            "examples": [example.to_dict() for example in enriched_examples],
        }
        dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        category_counts = Counter(example.label_category for example in enriched_examples)
        value_counts = Counter(example.label_value for example in enriched_examples)
        manifest_payload = {
            "schema_type": "labeled_example_dataset_manifest",
            "schema_version": "1",
            "export_batch_id": export_batch_id,
            "created_at": payload["created_at"],
            "session_id": session_id,
            "project_id": project_id,
            "dataset_path": str(dataset_path),
            "example_count": len(enriched_examples),
            "split_counts": dict(split_counts),
            "label_category_counts": dict(category_counts),
            "label_value_counts": dict(value_counts),
            "filters": {
                "session_id": session_id,
                "project_id": project_id,
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        return DatasetExportResult(
            repo_root=str(self.settings.repo_root),
            session_id=session_id,
            project_id=project_id,
            dataset_path=str(dataset_path),
            manifest_path=str(manifest_path),
            export_batch_id=export_batch_id,
            example_count=len(enriched_examples),
            split_counts=dict(split_counts),
            label_category_counts=dict(category_counts),
            label_value_counts=dict(value_counts),
            examples=tuple(enriched_examples),
        )

    @classmethod
    def _enrich_example(cls, example: LabeledExample, *, export_batch_id: str) -> LabeledExample:
        return LabeledExample(
            schema_version=example.schema_version,
            example_id=example.example_id,
            session_id=example.session_id,
            project_id=example.project_id,
            interaction_path=example.interaction_path,
            message_id=example.message_id,
            created_at=example.created_at,
            source_prompt=example.source_prompt,
            source_summary=example.source_summary,
            label_category=example.label_category,
            label_value=example.label_value,
            trainable=example.trainable,
            split_assignment=cls._stable_split_assignment(example),
            export_batch_id=export_batch_id,
            label_source=example.label_source,
            correction_label=example.correction_label,
            provenance=dict(example.provenance),
            metadata=dict(example.metadata),
        )

    @staticmethod
    def _stable_split_assignment(example: LabeledExample) -> str:
        return stable_split_assignment(
            example.example_id,
            example.session_id or "",
            example.project_id or "",
            example.message_id or "",
            example.label_category,
        )
