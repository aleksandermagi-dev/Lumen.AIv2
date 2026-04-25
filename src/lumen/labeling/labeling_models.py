from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class LabeledExample:
    schema_version: str
    example_id: str
    session_id: str | None
    project_id: str | None
    interaction_path: str | None
    message_id: str | None
    created_at: str | None
    source_prompt: str | None
    source_summary: str | None
    label_category: str
    label_value: str
    trainable: bool
    split_assignment: str | None = None
    export_batch_id: str | None = None
    label_source: str = "programmatic_evaluation"
    correction_label: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "interaction_path": self.interaction_path,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "source_prompt": self.source_prompt,
            "source_summary": self.source_summary,
            "label_category": self.label_category,
            "label_value": self.label_value,
            "trainable": self.trainable,
            "split_assignment": self.split_assignment,
            "export_batch_id": self.export_batch_id,
            "label_source": self.label_source,
            "correction_label": self.correction_label,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class DatasetExportResult:
    repo_root: str
    session_id: str | None
    project_id: str | None
    dataset_path: str
    manifest_path: str
    export_batch_id: str
    example_count: int
    split_counts: dict[str, int]
    label_category_counts: dict[str, int]
    label_value_counts: dict[str, int]
    examples: tuple[LabeledExample, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "dataset_path": self.dataset_path,
            "manifest_path": self.manifest_path,
            "export_batch_id": self.export_batch_id,
            "example_count": self.example_count,
            "split_counts": dict(self.split_counts),
            "label_category_counts": dict(self.label_category_counts),
            "label_value_counts": dict(self.label_value_counts),
            "examples": [example.to_dict() for example in self.examples],
        }
