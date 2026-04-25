from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager


SUPPORTED_EXPORT_TYPES = {
    "instruction_response",
    "classification",
    "explanation_pair",
    "qa_pair",
}
INGESTION_STATES = {"staged", "ready", "rejected", "archived"}


class DatasetCurationService:
    """Curates, compares, and exports supervised dataset rows stored in SQLite."""

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager):
        self.settings = settings
        self.persistence_manager = persistence_manager

    def list_examples_for_review(
        self,
        *,
        dataset_name: str | None = None,
        import_run_id: str | None = None,
        example_type: str | None = None,
        split_assignment: str | None = None,
        ingestion_state: str | None = None,
        label_source: str | None = None,
        trainable: bool | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        self.persistence_manager.bootstrap()
        return self.persistence_manager.dataset_examples.list_examples(
            dataset_name=dataset_name,
            import_run_id=import_run_id,
            example_type=example_type,
            split_assignment=split_assignment,
            ingestion_state=ingestion_state,
            label_source=label_source,
            trainable=trainable,
            limit=limit,
        )

    def sample_review_batch(
        self,
        *,
        dataset_name: str | None = None,
        import_run_id: str | None = None,
        example_type: str | None = None,
        limit: int = 50,
        prioritize: str = "programmatic_first",
        output_path: Path | None = None,
    ) -> dict[str, object]:
        examples = self.list_examples_for_review(
            dataset_name=dataset_name,
            import_run_id=import_run_id,
            example_type=example_type,
            ingestion_state="ready",
            limit=max(limit * 4, limit),
        )
        ranked = sorted(examples, key=lambda item: self._review_priority(item, prioritize=prioritize), reverse=True)
        selected = ranked[: max(int(limit), 1)]
        payload = {
            "status": "ok",
            "schema_type": "dataset_review_batch",
            "generated_at": self._timestamp_now(),
            "dataset_name": dataset_name,
            "import_run_id": import_run_id,
            "example_type": example_type,
            "review_count": len(selected),
            "prioritize": prioritize,
            "examples": selected,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["output_path"] = str(output_path)
        return payload

    def update_dataset_example(
        self,
        *,
        example_id: str,
        trainable: bool | None = None,
        ingestion_state: str | None = None,
        split_assignment: str | None = None,
        label_source: str | None = None,
        review_note: str | None = None,
    ) -> dict[str, object]:
        self.persistence_manager.bootstrap()
        if ingestion_state is not None and ingestion_state not in INGESTION_STATES:
            raise ValueError(f"Unsupported ingestion state: {ingestion_state}")
        updated = self.persistence_manager.dataset_examples.update_example(
            example_id=example_id,
            trainable=trainable,
            ingestion_state=ingestion_state,
            split_assignment=split_assignment,
            label_source=label_source,
            metadata_patch={"review_note": review_note} if review_note else None,
            updated_at=self._timestamp_now(),
        )
        if updated is None:
            raise ValueError(f"Unknown dataset example: {example_id}")
        return {
            "status": "ok",
            "schema_type": "dataset_example_update",
            "example": updated,
        }

    def label_dataset_example(
        self,
        *,
        dataset_example_id: str,
        label_role: str,
        label_value: str,
        label_category: str | None = None,
        reviewer: str | None = None,
        reason: str | None = None,
        is_canonical: bool = False,
        metadata: dict[str, object] | None = None,
        sync_example_fields: bool = True,
    ) -> dict[str, object]:
        self.persistence_manager.bootstrap()
        if is_canonical:
            self.persistence_manager.dataset_example_labels.clear_canonical_for_role(
                dataset_example_id=dataset_example_id,
                label_role=label_role,
            )
        created_at = self._timestamp_now()
        label = self.persistence_manager.dataset_example_labels.upsert(
            label_id=self._make_id(
                prefix="dataset_label",
                seed=f"{dataset_example_id}|{label_role}|{label_value}|{created_at}",
            ),
            dataset_example_id=dataset_example_id,
            label_role=label_role,
            label_value=label_value,
            label_category=label_category,
            is_canonical=is_canonical,
            reviewer=reviewer,
            reason=reason,
            created_at=created_at,
            metadata=metadata or {},
        )
        example = self.persistence_manager.dataset_examples.get(dataset_example_id)
        if example is None:
            raise ValueError(f"Unknown dataset example: {dataset_example_id}")
        if sync_example_fields and label_role in {"target_label", "canonical_label"}:
            example = self.persistence_manager.dataset_examples.update_example(
                example_id=dataset_example_id,
                label_source="canonical_override" if is_canonical else "human_review",
                updated_at=self._timestamp_now(),
                metadata_patch=None,
            )
            if example is not None:
                self.persistence_manager.dataset_examples.upsert(
                    example_id=str(example["id"]),
                    import_run_id=str(example["import_run_id"]),
                    example_type=str(example["example_type"]),
                    source_format=str(example["source_format"]),
                    split_assignment=str(example["split_assignment"]),
                    ingestion_state=str(example["ingestion_state"]),
                    input_text=str(example["input_text"]),
                    target_text=str(example.get("target_text") or "") or None,
                    label_category=label_category or str(example.get("label_category") or "") or None,
                    label_value=label_value,
                    explanation_text=str(example.get("explanation_text") or "") or None,
                    source_session_id=str(example.get("source_session_id") or "") or None,
                    source_message_id=str(example.get("source_message_id") or "") or None,
                    source_interaction_path=str(example.get("source_interaction_path") or "") or None,
                    source_trace_id=str(example.get("source_trace_id") or "") or None,
                    source_tool_run_id=str(example.get("source_tool_run_id") or "") or None,
                    label_source="canonical_override" if is_canonical else "human_review",
                    trainable=bool(example.get("trainable")),
                    provenance=dict(example.get("provenance_json") or {}),
                    metadata=dict(example.get("metadata_json") or {}),
                    created_at=str(example["created_at"]),
                    updated_at=self._timestamp_now(),
                )
        return {
            "status": "ok",
            "schema_type": "dataset_example_label_update",
            "label": label,
            "dataset_example_id": dataset_example_id,
        }

    def export_dataset_jsonl(
        self,
        *,
        dataset_name: str,
        import_run_ids: list[str] | None = None,
        split_assignments: list[str] | None = None,
        example_types: list[str] | None = None,
        label_sources: list[str] | None = None,
        canonical_only: bool = False,
        trainable_only: bool = True,
        evaluation_only: bool = False,
        export_name: str | None = None,
        output_root: Path | None = None,
    ) -> dict[str, object]:
        self.persistence_manager.bootstrap()
        export_root = (output_root or self.settings.labeled_datasets_root).resolve()
        export_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        export_id = export_name or f"{dataset_name}_{'eval' if evaluation_only else 'train'}_{export_timestamp}"
        export_dir = export_root / dataset_name / export_id
        export_dir.mkdir(parents=True, exist_ok=True)

        effective_splits = [item.lower() for item in (split_assignments or []) if str(item).strip()]
        if evaluation_only and not effective_splits:
            effective_splits = ["validation", "test"]
        elif not effective_splits:
            effective_splits = ["train", "validation", "test"]

        selected_examples = self._select_examples_for_export(
            dataset_name=dataset_name,
            import_run_ids=import_run_ids or [],
            split_assignments=effective_splits,
            example_types=example_types or [],
            label_sources=label_sources or [],
            trainable_only=trainable_only,
        )
        labels_by_example = self.persistence_manager.dataset_example_labels.canonical_labels_for_examples(
            [str(item["id"]) for item in selected_examples]
        )
        quality_report = self._quality_checks(
            examples=selected_examples,
            labels_by_example=labels_by_example,
            canonical_only=canonical_only,
        )

        per_split_lines: dict[str, list[str]] = {split: [] for split in effective_splits}
        exported_examples = 0
        exported_ids: list[str] = []
        type_counts: Counter[str] = Counter()
        label_source_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()

        for example in selected_examples:
            split = str(example.get("split_assignment") or "train")
            if split not in per_split_lines:
                continue
            record = self._export_record(
                example=example,
                labels_by_example=labels_by_example,
                canonical_only=canonical_only,
            )
            if record is None:
                continue
            per_split_lines[split].append(json.dumps(record, ensure_ascii=True))
            exported_examples += 1
            exported_ids.append(str(example["id"]))
            type_counts[str(example.get("example_type") or "unknown")] += 1
            label_source_counts[str(example.get("label_source") or "unknown")] += 1
            split_counts[split] += 1

        split_files: dict[str, str] = {}
        for split, lines in per_split_lines.items():
            if not lines:
                continue
            path = export_dir / f"{split}.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            split_files[split] = str(path)

        manifest = {
            "schema_type": "curated_dataset_export",
            "schema_version": "1",
            "dataset_name": dataset_name,
            "export_name": export_id,
            "evaluation_only": evaluation_only,
            "canonical_only": canonical_only,
            "created_at": self._timestamp_now(),
            "example_count": exported_examples,
            "split_counts": dict(split_counts),
            "example_type_counts": dict(type_counts),
            "label_source_counts": dict(label_source_counts),
            "source_import_runs": list(import_run_ids or sorted({str(item.get("import_run_id")) for item in selected_examples})),
            "split_files": split_files,
            "quality_report": quality_report,
            "exported_example_ids": exported_ids,
        }
        manifest_path = export_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary_path = export_dir / "summary.txt"
        summary_path.write_text(self._render_export_summary(manifest), encoding="utf-8")

        return {
            "status": "ok",
            "schema_type": "dataset_jsonl_export",
            "dataset_name": dataset_name,
            "export_name": export_id,
            "export_dir": str(export_dir),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "example_count": exported_examples,
            "split_counts": dict(split_counts),
            "label_source_counts": dict(label_source_counts),
            "example_type_counts": dict(type_counts),
            "quality_report": quality_report,
            "split_files": split_files,
        }

    def compare_dataset_runs(
        self,
        *,
        left_import_run_id: str,
        right_import_run_id: str,
    ) -> dict[str, object]:
        self.persistence_manager.bootstrap()
        left_run = self.persistence_manager.dataset_import_runs.get(left_import_run_id)
        right_run = self.persistence_manager.dataset_import_runs.get(right_import_run_id)
        if left_run is None or right_run is None:
            raise ValueError("Both dataset import runs must exist")
        left_examples = self.persistence_manager.dataset_examples.list_examples(import_run_id=left_import_run_id, limit=5000)
        right_examples = self.persistence_manager.dataset_examples.list_examples(import_run_id=right_import_run_id, limit=5000)
        left_hashes = {self._overlap_key(item) for item in left_examples}
        right_hashes = {self._overlap_key(item) for item in right_examples}
        overlap = left_hashes.intersection(right_hashes)
        return {
            "status": "ok",
            "schema_type": "dataset_run_comparison",
            "left_run": self._run_summary(left_run, left_examples),
            "right_run": self._run_summary(right_run, right_examples),
            "overlap": {
                "shared_example_signature_count": len(overlap),
                "left_only_signature_count": len(left_hashes - right_hashes),
                "right_only_signature_count": len(right_hashes - left_hashes),
                "shared_ratio_vs_smaller_run": round(len(overlap) / max(1, min(len(left_hashes), len(right_hashes))), 4),
            },
        }

    def _select_examples_for_export(
        self,
        *,
        dataset_name: str,
        import_run_ids: list[str],
        split_assignments: list[str],
        example_types: list[str],
        label_sources: list[str],
        trainable_only: bool,
    ) -> list[dict[str, object]]:
        filters = {
            "dataset_name": dataset_name,
            "limit": 5000,
            "trainable": True if trainable_only else None,
        }
        examples = self.persistence_manager.dataset_examples.list_examples(**filters)
        selected: list[dict[str, object]] = []
        for example in examples:
            if import_run_ids and str(example.get("import_run_id")) not in import_run_ids:
                continue
            if split_assignments and str(example.get("split_assignment")) not in split_assignments:
                continue
            if example_types and str(example.get("example_type")) not in example_types:
                continue
            if label_sources and str(example.get("label_source")) not in label_sources:
                continue
            if str(example.get("ingestion_state") or "ready") != "ready":
                continue
            selected.append(example)
        return selected

    def _quality_checks(
        self,
        *,
        examples: list[dict[str, object]],
        labels_by_example: dict[str, list[dict[str, object]]],
        canonical_only: bool,
    ) -> dict[str, object]:
        empty_inputs = 0
        empty_targets = 0
        duplicate_groups: dict[str, list[str]] = {}
        split_leakage: dict[str, set[str]] = {}
        missing_canonical_labels: list[str] = []
        for example in examples:
            example_id = str(example.get("id") or "")
            input_text = str(example.get("input_text") or "").strip()
            target_text = str(example.get("target_text") or "").strip()
            if not input_text:
                empty_inputs += 1
            if str(example.get("example_type")) in {"instruction_response", "qa_pair", "explanation_pair"} and not target_text:
                empty_targets += 1
            signature = self._overlap_key(example)
            duplicate_groups.setdefault(signature, []).append(example_id)
            split_key = self._normalized_text(input_text)
            split_leakage.setdefault(split_key, set()).add(str(example.get("split_assignment") or "train"))
            if canonical_only and not labels_by_example.get(example_id):
                missing_canonical_labels.append(example_id)
        duplicate_examples = [ids for ids in duplicate_groups.values() if len(ids) > 1]
        split_leakage_hits = [
            {"text_hash": self._hash_text(text), "splits": sorted(splits)}
            for text, splits in split_leakage.items()
            if len(splits) > 1
        ]
        return {
            "empty_input_count": empty_inputs,
            "empty_target_count": empty_targets,
            "duplicate_group_count": len(duplicate_examples),
            "duplicate_example_groups": duplicate_examples[:25],
            "split_leakage_count": len(split_leakage_hits),
            "split_leakage_examples": split_leakage_hits[:25],
            "missing_canonical_label_count": len(missing_canonical_labels),
            "missing_canonical_label_examples": missing_canonical_labels[:50],
        }

    def _export_record(
        self,
        *,
        example: dict[str, object],
        labels_by_example: dict[str, list[dict[str, object]]],
        canonical_only: bool,
    ) -> dict[str, object] | None:
        example_id = str(example.get("id") or "")
        canonical_labels = labels_by_example.get(example_id) or []
        canonical_target = next(
            (item for item in canonical_labels if str(item.get("label_role")) in {"target_label", "canonical_label"}),
            None,
        )
        if canonical_only and not canonical_labels:
            return None
        example_type = str(example.get("example_type") or "")
        if example_type not in SUPPORTED_EXPORT_TYPES:
            return None
        if example_type == "instruction_response":
            target_text = str(example.get("target_text") or "").strip()
            if not target_text:
                return None
            return {
                "instruction": str(example.get("input_text") or ""),
                "response": target_text,
                "metadata": self._base_export_metadata(example=example, canonical_labels=canonical_labels),
            }
        if example_type == "classification":
            label_value = str((canonical_target or {}).get("label_value") or example.get("label_value") or "").strip()
            if not label_value:
                return None
            return {
                "text": str(example.get("input_text") or ""),
                "label_category": str((canonical_target or {}).get("label_category") or example.get("label_category") or ""),
                "label_value": label_value,
                "label_source": "canonical_override" if canonical_target else str(example.get("label_source") or ""),
                "metadata": self._base_export_metadata(example=example, canonical_labels=canonical_labels),
            }
        if example_type == "qa_pair":
            target_text = str(example.get("target_text") or "").strip()
            if not target_text:
                return None
            return {
                "question": str(example.get("input_text") or ""),
                "answer": target_text,
                "metadata": self._base_export_metadata(example=example, canonical_labels=canonical_labels),
            }
        if example_type == "explanation_pair":
            target_text = str(example.get("target_text") or "").strip()
            if not target_text:
                return None
            return {
                "prompt": str(example.get("input_text") or ""),
                "answer": target_text,
                "explanation": str(example.get("explanation_text") or ""),
                "metadata": {
                    **self._base_export_metadata(example=example, canonical_labels=canonical_labels),
                    "bounded_explanation": True,
                },
            }
        return None

    def _base_export_metadata(
        self,
        *,
        example: dict[str, object],
        canonical_labels: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "dataset_name": example.get("dataset_name"),
            "import_run_id": example.get("import_run_id"),
            "example_id": example.get("id"),
            "example_type": example.get("example_type"),
            "split_assignment": example.get("split_assignment"),
            "label_source": example.get("label_source"),
            "source_session_id": example.get("source_session_id"),
            "source_message_id": example.get("source_message_id"),
            "source_trace_id": example.get("source_trace_id"),
            "source_tool_run_id": example.get("source_tool_run_id"),
            "canonical_labels": [
                {
                    "label_role": item.get("label_role"),
                    "label_value": item.get("label_value"),
                    "label_category": item.get("label_category"),
                    "reviewer": item.get("reviewer"),
                }
                for item in canonical_labels
            ],
        }

    def _render_export_summary(self, manifest: dict[str, object]) -> str:
        lines = [
            f"dataset_name: {manifest.get('dataset_name')}",
            f"export_name: {manifest.get('export_name')}",
            f"example_count: {manifest.get('example_count')}",
        ]
        split_counts = manifest.get("split_counts") or {}
        if split_counts:
            lines.append("split_counts:")
            for key, value in split_counts.items():
                lines.append(f"- {key}: {value}")
        type_counts = manifest.get("example_type_counts") or {}
        if type_counts:
            lines.append("example_type_counts:")
            for key, value in type_counts.items():
                lines.append(f"- {key}: {value}")
        quality = manifest.get("quality_report") or {}
        if quality:
            lines.append(f"duplicate_group_count: {quality.get('duplicate_group_count')}")
            lines.append(f"split_leakage_count: {quality.get('split_leakage_count')}")
            lines.append(f"missing_canonical_label_count: {quality.get('missing_canonical_label_count')}")
        return "\n".join(lines) + "\n"

    def _run_summary(self, run: dict[str, object], examples: list[dict[str, object]]) -> dict[str, object]:
        split_counts = Counter(str(item.get("split_assignment") or "unknown") for item in examples)
        label_source_counts = Counter(str(item.get("label_source") or "unknown") for item in examples)
        example_type_counts = Counter(str(item.get("example_type") or "unknown") for item in examples)
        return {
            "id": run.get("id"),
            "dataset_name": run.get("dataset_name"),
            "dataset_version": run.get("dataset_version"),
            "dataset_kind": run.get("dataset_kind"),
            "import_strategy": run.get("import_strategy"),
            "example_count": len(examples),
            "split_counts": dict(split_counts),
            "label_source_counts": dict(label_source_counts),
            "example_type_counts": dict(example_type_counts),
        }

    @staticmethod
    def _review_priority(example: dict[str, object], *, prioritize: str) -> tuple[int, float, str]:
        metadata = dict(example.get("metadata_json") or {})
        score = float(metadata.get("evaluation_score") or 0.0)
        label_source = str(example.get("label_source") or "")
        source_priority = 1 if prioritize == "programmatic_first" and label_source == "programmatic_evaluation" else 0
        low_score_priority = 1 if score and score < 0.75 else 0
        return (source_priority, low_score_priority, str(example.get("created_at") or ""))

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip().lower()

    @classmethod
    def _overlap_key(cls, example: dict[str, object]) -> str:
        signature = "|".join(
            [
                str(example.get("example_type") or ""),
                cls._normalized_text(str(example.get("input_text") or "")),
                cls._normalized_text(str(example.get("target_text") or "")),
                str(example.get("label_value") or ""),
            ]
        )
        return cls._hash_text(signature)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _make_id(*, prefix: str, seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}:{digest}"

    @staticmethod
    def _timestamp_now() -> str:
        return datetime.now(UTC).isoformat()
