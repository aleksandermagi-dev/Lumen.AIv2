from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.labeling.split_assignment import stable_split_assignment


SUPPORTED_SOURCE_FORMATS = {
    "json",
    "jsonl",
    "csv",
    "qa_pairs",
    "instruction_response",
    "reasoning_explanation_pairs",
    "classification_examples",
    "runtime_sqlite",
}

INGESTION_STATES = {"staged", "ready", "rejected", "archived"}
LABEL_SOURCES = {
    "programmatic_evaluation",
    "imported_external",
    "human_review",
    "canonical_override",
}


class DatasetIngestionService:
    """Stores imported and derived supervised examples in the unified SQLite DB."""

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager):
        self.settings = settings
        self.persistence_manager = persistence_manager

    def create_import_run(
        self,
        *,
        dataset_name: str,
        source_format: str,
        dataset_kind: str,
        import_strategy: str,
        dataset_version: str | None = None,
        source_path: str | None = None,
        ingestion_status: str = "staged",
        schema_version: str = "1",
        notes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.persistence_manager.bootstrap()
        created_at = datetime.now(UTC).isoformat()
        import_run_id = self._make_id(
            prefix="dataset_import",
            seed="|".join(
                [
                    dataset_name,
                    dataset_version or "",
                    source_format,
                    dataset_kind,
                    import_strategy,
                    created_at,
                ]
            ),
        )
        return self.persistence_manager.dataset_import_runs.upsert(
            import_run_id=import_run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            source_path=source_path,
            source_format=source_format,
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            ingestion_status=ingestion_status,
            example_count=0,
            train_count=0,
            validation_count=0,
            test_count=0,
            schema_version=schema_version,
            notes=notes or {},
            created_at=created_at,
            completed_at=None,
        )

    def import_dataset(
        self,
        *,
        dataset_name: str,
        source_format: str,
        dataset_kind: str,
        source_path: Path | None = None,
        dataset_version: str | None = None,
        import_strategy: str = "external_file",
        csv_mapping: dict[str, str] | None = None,
        notes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_source_format = str(source_format or "").strip().lower()
        if normalized_source_format not in SUPPORTED_SOURCE_FORMATS - {"runtime_sqlite"}:
            raise ValueError(f"Unsupported dataset source_format: {source_format}")
        if source_path is None:
            raise ValueError("source_path is required for external dataset import")
        import_run = self.create_import_run(
            dataset_name=dataset_name,
            source_format=normalized_source_format,
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            dataset_version=dataset_version,
            source_path=str(source_path),
            notes=notes,
        )
        records = self._load_external_records(
            source_format=normalized_source_format,
            dataset_kind=dataset_kind,
            source_path=source_path,
            csv_mapping=csv_mapping or {},
        )
        return self._persist_normalized_examples(
            import_run_id=str(import_run["id"]),
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            source_format=normalized_source_format,
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            source_path=str(source_path),
            normalized_examples=records,
        )

    def import_runtime_examples(
        self,
        *,
        dataset_name: str,
        import_strategy: str,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        if import_strategy not in {"derived_trainability", "derived_instruction_response", "derived_tool_runs"}:
            raise ValueError(f"Unsupported runtime import_strategy: {import_strategy}")
        dataset_kind = {
            "derived_trainability": "classification_examples",
            "derived_instruction_response": "instruction_response",
            "derived_tool_runs": "classification_examples",
        }[import_strategy]
        import_run = self.create_import_run(
            dataset_name=dataset_name,
            source_format="runtime_sqlite",
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            source_path=str(self.settings.persistence_db_path),
            notes={"session_id": session_id, "project_id": project_id, "limit": limit},
        )
        if import_strategy == "derived_trainability":
            examples = self._derive_trainability_examples(session_id=session_id, project_id=project_id, limit=limit)
        elif import_strategy == "derived_instruction_response":
            examples = self._derive_instruction_response_examples(session_id=session_id, project_id=project_id, limit=limit)
        else:
            examples = self._derive_tool_examples(session_id=session_id, project_id=project_id, limit=limit)
        return self._persist_normalized_examples(
            import_run_id=str(import_run["id"]),
            dataset_name=dataset_name,
            dataset_version=None,
            source_format="runtime_sqlite",
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            source_path=str(self.settings.persistence_db_path),
            normalized_examples=examples,
        )

    def list_dataset_import_runs(self, *, dataset_name: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        self.persistence_manager.bootstrap()
        return self.persistence_manager.dataset_import_runs.list_runs(dataset_name=dataset_name, limit=limit)

    def list_dataset_examples(
        self,
        *,
        import_run_id: str | None = None,
        example_type: str | None = None,
        split_assignment: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        self.persistence_manager.bootstrap()
        return self.persistence_manager.dataset_examples.list_examples(
            import_run_id=import_run_id,
            example_type=example_type,
            split_assignment=split_assignment,
            limit=limit,
        )

    def list_dataset_labels(self, *, dataset_example_id: str | None = None, limit: int = 200) -> list[dict[str, object]]:
        self.persistence_manager.bootstrap()
        return self.persistence_manager.dataset_example_labels.list_labels(
            dataset_example_id=dataset_example_id,
            limit=limit,
        )

    def _persist_normalized_examples(
        self,
        *,
        import_run_id: str,
        dataset_name: str,
        dataset_version: str | None,
        source_format: str,
        dataset_kind: str,
        import_strategy: str,
        source_path: str | None,
        normalized_examples: list[dict[str, Any]],
    ) -> dict[str, object]:
        created_at = datetime.now(UTC).isoformat()
        split_counts = {"train": 0, "validation": 0, "test": 0}
        stored_examples: list[dict[str, object]] = []
        label_rows = 0
        created_seed = self.persistence_manager.dataset_import_runs.get(import_run_id)
        run_created_at = str(created_seed.get("created_at") or created_at) if isinstance(created_seed, dict) else created_at
        for position, example in enumerate(normalized_examples):
            split_assignment = str(example.get("split_assignment") or "").strip().lower()
            if split_assignment not in {"train", "validation", "test"}:
                split_assignment = stable_split_assignment(
                    str(example.get("id") or ""),
                    str(example.get("source_session_id") or ""),
                    str(example.get("source_message_id") or ""),
                    str(example.get("label_category") or ""),
                    str(position),
                )
            split_counts[split_assignment] += 1
            row_created_at = str(example.get("created_at") or created_at)
            provenance = dict(example.get("provenance") or {})
            provenance.setdefault("source_format", source_format)
            provenance.setdefault("source_path", source_path)
            provenance.setdefault("import_run_id", import_run_id)
            example_id = str(
                example.get("id")
                or self._make_id(
                    prefix="dataset_example",
                    seed=f"{import_run_id}|{position}|{example.get('input_text')}|{example.get('label_value')}",
                )
            )
            stored = self.persistence_manager.dataset_examples.upsert(
                example_id=example_id,
                import_run_id=import_run_id,
                example_type=str(example["example_type"]),
                source_format=source_format,
                split_assignment=split_assignment,
                ingestion_state=str(example.get("ingestion_state") or "ready"),
                input_text=str(example.get("input_text") or ""),
                target_text=self._opt_text(example.get("target_text")),
                label_category=self._opt_text(example.get("label_category")),
                label_value=self._opt_text(example.get("label_value")),
                explanation_text=self._opt_text(example.get("explanation_text")),
                source_session_id=self._opt_text(example.get("source_session_id")),
                source_message_id=self._opt_text(example.get("source_message_id")),
                source_interaction_path=self._opt_text(example.get("source_interaction_path")),
                source_trace_id=self._opt_text(example.get("source_trace_id")),
                source_tool_run_id=self._opt_text(example.get("source_tool_run_id")),
                label_source=self._normalize_label_source(example.get("label_source")),
                trainable=bool(example.get("trainable", True)),
                provenance=provenance,
                metadata=dict(example.get("metadata") or {}),
                created_at=row_created_at,
                updated_at=created_at,
            )
            stored_examples.append(stored)
            label_rows += self._persist_labels_for_example(example_id=example_id, example=example, created_at=created_at)

        completed_at = datetime.now(UTC).isoformat()
        run = self.persistence_manager.dataset_import_runs.upsert(
            import_run_id=import_run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            source_path=source_path,
            source_format=source_format,
            dataset_kind=dataset_kind,
            import_strategy=import_strategy,
            ingestion_status="ready",
            example_count=len(stored_examples),
            train_count=split_counts["train"],
            validation_count=split_counts["validation"],
            test_count=split_counts["test"],
            schema_version="1",
            notes={"label_row_count": label_rows},
            created_at=run_created_at,
            completed_at=completed_at,
        )
        return {
            "status": "ok",
            "import_run": run,
            "example_count": len(stored_examples),
            "split_counts": split_counts,
            "label_row_count": label_rows,
            "examples": stored_examples,
        }

    def _persist_labels_for_example(self, *, example_id: str, example: dict[str, Any], created_at: str) -> int:
        labels: list[dict[str, Any]] = []
        if example.get("label_value") is not None:
            labels.append(
                {
                    "label_role": "target_label",
                    "label_value": str(example.get("label_value")),
                    "label_category": self._opt_text(example.get("label_category")),
                    "is_canonical": True,
                    "reviewer": None,
                    "reason": "initial import target label",
                    "metadata": {"label_source": example.get("label_source")},
                }
            )
        correction_label = self._opt_text(example.get("correction_label"))
        if correction_label is not None:
            labels.append(
                {
                    "label_role": "correction_label",
                    "label_value": correction_label,
                    "label_category": self._opt_text(example.get("label_category")),
                    "is_canonical": False,
                    "reviewer": None,
                    "reason": "imported correction label",
                    "metadata": {},
                }
            )
        for item in example.get("labels") or []:
            if isinstance(item, dict):
                labels.append(item)
        inserted = 0
        for index, item in enumerate(labels):
            label_role = str(item.get("label_role") or "target_label").strip() or "target_label"
            is_canonical = bool(item.get("is_canonical"))
            if is_canonical:
                self.persistence_manager.dataset_example_labels.clear_canonical_for_role(
                    dataset_example_id=example_id,
                    label_role=label_role,
                )
            self.persistence_manager.dataset_example_labels.upsert(
                label_id=self._make_id(prefix="dataset_label", seed=f"{example_id}|{label_role}|{index}|{item.get('label_value')}"),
                dataset_example_id=example_id,
                label_role=label_role,
                label_value=str(item.get("label_value") or ""),
                label_category=self._opt_text(item.get("label_category")),
                is_canonical=is_canonical,
                reviewer=self._opt_text(item.get("reviewer")),
                reason=self._opt_text(item.get("reason")),
                created_at=created_at,
                metadata=dict(item.get("metadata") or {}),
            )
            inserted += 1
        return inserted

    def _load_external_records(
        self,
        *,
        source_format: str,
        dataset_kind: str,
        source_path: Path,
        csv_mapping: dict[str, str],
    ) -> list[dict[str, Any]]:
        if source_format == "json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
                raw_records = list(payload["examples"])
            elif isinstance(payload, list):
                raw_records = payload
            else:
                raw_records = [payload]
        elif source_format == "jsonl":
            raw_records = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif source_format == "csv":
            with source_path.open("r", encoding="utf-8", newline="") as handle:
                raw_records = list(csv.DictReader(handle))
        else:
            raise ValueError(f"Unsupported external dataset format: {source_format}")
        return [
            self._normalize_record(
                record,
                source_format=source_format,
                dataset_kind=dataset_kind,
                csv_mapping=csv_mapping,
                source_path=str(source_path),
            )
            for record in raw_records
        ]

    def _normalize_record(
        self,
        record: dict[str, Any],
        *,
        source_format: str,
        dataset_kind: str,
        csv_mapping: dict[str, str],
        source_path: str,
    ) -> dict[str, Any]:
        kind = str(dataset_kind or source_format).strip().lower()
        if kind == "qa_pairs":
            return self._normalized_base(
                example_type="qa_pair",
                input_text=record.get("question"),
                target_text=record.get("answer"),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if kind == "instruction_response":
            return self._normalized_base(
                example_type="instruction_response",
                input_text=record.get("instruction"),
                target_text=record.get("response"),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if kind == "reasoning_explanation_pairs":
            return self._normalized_base(
                example_type="explanation_pair",
                input_text=record.get("prompt"),
                target_text=record.get("answer"),
                explanation_text=record.get("explanation"),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if kind == "classification_examples":
            return self._normalized_base(
                example_type="classification",
                input_text=record.get("text"),
                label_category=record.get("label_category"),
                label_value=record.get("label_value") or record.get("label"),
                correction_label=record.get("correction_label"),
                labels=record.get("labels"),
                label_source=record.get("label_source") or "imported_external",
                source_path=source_path,
                record=record,
            )
        if source_format == "csv":
            return self._normalize_csv_record(record, dataset_kind=kind, csv_mapping=csv_mapping, source_path=source_path)
        raise ValueError(f"Unsupported dataset kind for normalization: {dataset_kind}")

    def _normalize_csv_record(
        self,
        record: dict[str, Any],
        *,
        dataset_kind: str,
        csv_mapping: dict[str, str],
        source_path: str,
    ) -> dict[str, Any]:
        if dataset_kind == "qa_pairs":
            return self._normalized_base(
                example_type="qa_pair",
                input_text=record.get(csv_mapping.get("question_column", "question")),
                target_text=record.get(csv_mapping.get("answer_column", "answer")),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if dataset_kind == "instruction_response":
            return self._normalized_base(
                example_type="instruction_response",
                input_text=record.get(csv_mapping.get("instruction_column", "instruction")),
                target_text=record.get(csv_mapping.get("response_column", "response")),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if dataset_kind == "reasoning_explanation_pairs":
            return self._normalized_base(
                example_type="explanation_pair",
                input_text=record.get(csv_mapping.get("prompt_column", "prompt")),
                target_text=record.get(csv_mapping.get("answer_column", "answer")),
                explanation_text=record.get(csv_mapping.get("explanation_column", "explanation")),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        if dataset_kind == "classification_examples":
            return self._normalized_base(
                example_type="classification",
                input_text=record.get(csv_mapping.get("text_column", "text")) or record.get(csv_mapping.get("input_column", "input")),
                label_category=csv_mapping.get("label_category") or record.get("label_category"),
                label_value=record.get(csv_mapping.get("label_column", "label")),
                label_source="imported_external",
                source_path=source_path,
                record=record,
            )
        raise ValueError("CSV imports require an explicit dataset_kind mapping")

    def _normalized_base(
        self,
        *,
        example_type: str,
        input_text: object,
        target_text: object | None = None,
        label_category: object | None = None,
        label_value: object | None = None,
        explanation_text: object | None = None,
        correction_label: object | None = None,
        labels: object | None = None,
        label_source: object | None = None,
        source_path: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "example_type": example_type,
            "input_text": str(input_text or ""),
            "target_text": self._opt_text(target_text),
            "label_category": self._opt_text(label_category),
            "label_value": self._opt_text(label_value),
            "explanation_text": self._opt_text(explanation_text),
            "correction_label": self._opt_text(correction_label),
            "labels": labels if isinstance(labels, list) else [],
            "label_source": self._normalize_label_source(label_source),
            "trainable": True,
            "ingestion_state": "ready",
            "provenance": dict(record.get("provenance") or {"source_path": source_path}),
            "metadata": {"raw_record": record},
        }

    def _derive_trainability_examples(
        self,
        *,
        session_id: str | None,
        project_id: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        params: list[object] = []
        clauses: list[str] = []
        if session_id is not None:
            clauses.append("t.session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.persistence_manager.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    t.*,
                    m.content AS assistant_content,
                    m.message_metadata_json
                FROM trainability_traces t
                JOIN sessions s ON s.id = t.session_id
                LEFT JOIN messages m ON m.id = t.message_id
                {where}
                ORDER BY t.created_at DESC, t.id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        examples: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(str(row["metadata_json"] or "{}")) if row["metadata_json"] else {}
            prompt = ""
            interaction_path = None
            if row["message_metadata_json"]:
                try:
                    message_metadata = json.loads(str(row["message_metadata_json"]))
                    interaction_record = message_metadata.get("interaction_record") or {}
                    prompt = str(interaction_record.get("prompt") or message_metadata.get("original_prompt") or "")
                    interaction_path = interaction_record.get("interaction_path") or message_metadata.get("interaction_path")
                except json.JSONDecodeError:
                    pass
            examples.append(
                {
                    "id": str(row["id"]),
                    "example_type": "classification",
                    "input_text": prompt or str(row["input_context_summary"] or row["assistant_content"] or ""),
                    "label_category": "interaction_outcome",
                    "label_value": self._opt_text(row["outcome"]),
                    "label_source": "programmatic_evaluation",
                    "trainable": True,
                    "ingestion_state": "ready",
                    "source_session_id": self._opt_text(row["session_id"]),
                    "source_message_id": self._opt_text(row["message_id"]),
                    "source_interaction_path": self._opt_text(interaction_path),
                    "source_trace_id": self._opt_text(row["id"]),
                    "provenance": {
                        "available_training_surfaces": list(metadata.get("available_training_surfaces") or []),
                        "deterministic_surfaces": list(metadata.get("deterministic_surfaces") or []),
                        "chosen_action": row["chosen_action"],
                        "decision_type": row["decision_type"],
                    },
                    "metadata": {
                        "evaluation_score": row["evaluation_score"],
                        "confidence_tier": row["confidence_tier"],
                    },
                    "created_at": str(row["created_at"]),
                }
            )
        return examples

    def _derive_instruction_response_examples(
        self,
        *,
        session_id: str | None,
        project_id: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        params: list[object] = []
        clauses: list[str] = []
        if session_id is not None:
            clauses.append("u.session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        filters = f"AND {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.persistence_manager.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    u.session_id,
                    a.id AS assistant_message_id,
                    u.content AS user_content,
                    a.content AS assistant_content,
                    a.intent_domain,
                    a.confidence_tier,
                    a.message_metadata_json,
                    a.created_at
                FROM messages u
                JOIN messages a
                    ON a.session_id = u.session_id
                    AND a.turn_key = u.turn_key
                    AND a.role = 'assistant'
                JOIN sessions s ON s.id = u.session_id
                WHERE u.role = 'user'
                {filters}
                ORDER BY a.created_at DESC, a.id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        examples: list[dict[str, Any]] = []
        for row in rows:
            interaction_path = None
            if row["message_metadata_json"]:
                try:
                    metadata = json.loads(str(row["message_metadata_json"]))
                    interaction_record = metadata.get("interaction_record") or {}
                    interaction_path = interaction_record.get("interaction_path") or metadata.get("interaction_path")
                except json.JSONDecodeError:
                    pass
            examples.append(
                {
                    "id": str(row["assistant_message_id"]),
                    "example_type": "instruction_response",
                    "input_text": str(row["user_content"] or ""),
                    "target_text": self._opt_text(row["assistant_content"]),
                    "label_source": "programmatic_evaluation",
                    "trainable": True,
                    "ingestion_state": "ready",
                    "source_session_id": self._opt_text(row["session_id"]),
                    "source_message_id": self._opt_text(row["assistant_message_id"]),
                    "source_interaction_path": self._opt_text(interaction_path),
                    "provenance": {
                        "intent_domain": row["intent_domain"],
                        "confidence_tier": row["confidence_tier"],
                    },
                    "metadata": {},
                    "created_at": str(row["created_at"]),
                }
            )
        return examples

    def _derive_tool_examples(
        self,
        *,
        session_id: str | None,
        project_id: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        params: list[object] = []
        clauses: list[str] = []
        if session_id is not None:
            clauses.append("t.session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.persistence_manager.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*
                FROM tool_runs t
                JOIN sessions s ON s.id = t.session_id
                {where}
                ORDER BY t.created_at DESC, t.id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        examples: list[dict[str, Any]] = []
        for row in rows:
            examples.append(
                {
                    "id": str(row["id"]),
                    "example_type": "classification",
                    "input_text": str(row["input_summary"] or row["output_summary"] or row["tool_name"] or ""),
                    "label_category": "tool_run_status",
                    "label_value": "success" if bool(row["success"]) else "error",
                    "label_source": "programmatic_evaluation",
                    "trainable": True,
                    "ingestion_state": "ready",
                    "source_session_id": self._opt_text(row["session_id"]),
                    "source_message_id": self._opt_text(row["message_id"]),
                    "source_tool_run_id": self._opt_text(row["id"]),
                    "provenance": {
                        "tool_name": row["tool_name"],
                        "capability": row["capability"],
                        "tool_bundle": row["tool_bundle"],
                    },
                    "metadata": {
                        "archive_path": row["archive_path"],
                        "run_dir": row["run_dir"],
                    },
                    "created_at": str(row["created_at"]),
                }
            )
        return examples

    @staticmethod
    def _normalize_label_source(value: object) -> str:
        candidate = str(value or "imported_external").strip()
        if candidate not in LABEL_SOURCES:
            return "imported_external"
        return candidate

    @staticmethod
    def _make_id(*, prefix: str, seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}:{digest}"

    @staticmethod
    def _opt_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None
