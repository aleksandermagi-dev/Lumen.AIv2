from __future__ import annotations

from datetime import UTC, datetime
from heapq import nlargest
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.schemas.research_artifact_schema import ResearchArtifactSchema
from lumen.schemas.research_note_schema import ResearchNoteSchema


class ResearchArtifactManager:
    """Promotes chronological research notes into traceable structured artifacts."""

    ALLOWED_TYPES = {"hypothesis", "finding", "experiment", "decision", "milestone"}

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager | None = None):
        self.settings = settings
        self.research_notes_root = settings.research_notes_root
        self.research_artifacts_root = settings.research_artifacts_root
        self.persistence_manager = persistence_manager or PersistenceManager(settings)

    def list_notes(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.persistence_manager.bootstrap()
        normalized_limit = max(int(limit), 1) if limit is not None else None
        db_rows = self.persistence_manager.memory_items.list_by_filters(
            session_id=session_id,
            project_id=project_id,
            source_type="research_note",
            include_archived=include_archived,
            archived_only=archived_only,
            limit=normalized_limit,
        )
        if db_rows:
            return [self._hydrate_db_record(row, path_key="note_path") for row in db_rows]
        root = self.research_notes_root / session_id if session_id else self.research_notes_root
        self.persistence_manager.record_fallback_read("research note structured listing fallback")
        return self._load_records(
            root,
            ResearchNoteSchema,
            include_archived=include_archived,
            archived_only=archived_only,
            limit=normalized_limit,
        )

    def list_artifacts(self, *, session_id: str | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        self.persistence_manager.bootstrap()
        db_rows = self.persistence_manager.memory_items.list_by_filters(
            session_id=session_id,
            project_id=project_id,
            source_type="research_artifact",
            include_archived=True,
        )
        if db_rows:
            return [self._hydrate_db_record(row, path_key="artifact_path") for row in db_rows]
        root = self.research_artifacts_root / session_id if session_id else self.research_artifacts_root
        self.persistence_manager.record_fallback_read("research artifact structured listing fallback")
        return self._load_records(root, ResearchArtifactSchema)

    def load_artifact(self, artifact_path: str | Path) -> dict[str, Any]:
        path = Path(artifact_path)
        if not path.is_absolute():
            path = self.research_artifacts_root / path.name
        self.persistence_manager.record_fallback_read("research artifact raw hydration")
        payload = ResearchArtifactSchema.normalize(json.loads(path.read_text(encoding="utf-8")))
        ResearchArtifactSchema.validate(payload)
        payload["artifact_path"] = str(path)
        return payload

    def archive_note(self, *, note_path: str) -> dict[str, Any]:
        path = Path(note_path)
        if not path.exists():
            row = self.persistence_manager.memory_items.update_status_by_identity(note_path, status="archived")
            if row is not None:
                return self._hydrate_db_record(row, path_key="note_path")
            raise FileNotFoundError(f"Research note not found: {note_path}")
        payload = ResearchNoteSchema.normalize(json.loads(path.read_text(encoding="utf-8")))
        ResearchNoteSchema.validate(payload)
        payload["archived"] = True
        payload["archived_at"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.persistence_manager.memory_items.update_status_by_identity(str(path), status="archived")
        payload["note_path"] = str(path)
        return payload

    def delete_note(self, *, note_path: str) -> dict[str, Any]:
        path = Path(note_path)
        existed = path.exists()
        if existed:
            path.unlink()
            self.persistence_manager.memory_items.delete_by_identity(str(path))
        elif self.persistence_manager.memory_items.delete_by_identity(note_path):
            return {
                "note_path": note_path,
                "memory_item_id": note_path,
                "deleted": True,
            }
        return {
            "note_path": str(path),
            "deleted": existed,
        }

    def promote_note(
        self,
        *,
        note_path: Path,
        artifact_type: str,
        title: str | None = None,
        promotion_reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_type = artifact_type.strip().lower()
        if normalized_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f"Unsupported artifact_type '{artifact_type}'. Expected one of: {', '.join(sorted(self.ALLOWED_TYPES))}"
            )
        note_payload = ResearchNoteSchema.normalize(
            json.loads(note_path.read_text(encoding="utf-8"))
        )
        ResearchNoteSchema.validate(note_payload)
        session_id = str(note_payload["session_id"])
        timestamp = datetime.now(UTC)
        target_dir = self.research_artifacts_root / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        artifact_path = target_dir / f"{stamp}.json"
        artifact_payload = ResearchArtifactSchema.normalize(
            {
                "session_id": session_id,
                "created_at": timestamp.isoformat(),
                "artifact_type": normalized_type,
                "title": title or str(note_payload.get("title") or "").strip() or normalized_type,
                "content": note_payload.get("content"),
                "normalized_topic": note_payload.get("normalized_topic"),
                "source_note_path": str(note_path),
                "source_interaction_prompt": note_payload.get("source_interaction_prompt"),
                "source_interaction_path": note_payload.get("source_interaction_path"),
                "promotion_reason": promotion_reason,
            }
        )
        ResearchArtifactSchema.validate(artifact_payload)
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

        promoted = list(note_payload.get("promoted_artifacts") or [])
        promoted.append(
            {
                "artifact_type": normalized_type,
                "artifact_path": str(artifact_path),
                "promoted_at": timestamp.isoformat(),
            }
        )
        note_payload["promoted_artifacts"] = promoted
        note_path.write_text(json.dumps(note_payload, indent=2), encoding="utf-8")
        note_payload["note_path"] = str(note_path)
        self.persistence_manager.record_memory_item(source_type="research_note", payload=note_payload)
        source_interaction_path = str(note_payload.get("source_interaction_path") or "").strip()
        if source_interaction_path:
            self.persistence_manager.update_interaction_memory_links(
                session_id=session_id,
                interaction_path=source_interaction_path,
                research_note=note_payload,
            )

        artifact_payload["artifact_path"] = str(artifact_path)
        self.persistence_manager.record_memory_item(source_type="research_artifact", payload=artifact_payload)
        return artifact_payload

    @staticmethod
    def _load_records(
        root: Path,
        schema,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not root.exists():
            return []
        normalized_limit = max(int(limit), 1) if limit is not None else None
        records: list[dict[str, Any]] = []
        seen_paths: set[Path] = set()
        for path in ResearchArtifactManager._candidate_fallback_paths(root=root, normalized_limit=normalized_limit):
            seen_paths.add(path)
            payload = schema.normalize(json.loads(path.read_text(encoding="utf-8")))
            schema.validate(payload)
            is_archived = bool(payload.get("archived", False))
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            key = "note_path" if schema is ResearchNoteSchema else "artifact_path"
            payload[key] = str(path)
            records.append(payload)
            if normalized_limit is not None and len(records) >= normalized_limit:
                return records
        if normalized_limit is not None and len(records) < normalized_limit:
            for path in sorted(root.rglob("*.json"), reverse=True):
                if path in seen_paths:
                    continue
                payload = schema.normalize(json.loads(path.read_text(encoding="utf-8")))
                schema.validate(payload)
                is_archived = bool(payload.get("archived", False))
                if archived_only and not is_archived:
                    continue
                if not include_archived and is_archived:
                    continue
                key = "note_path" if schema is ResearchNoteSchema else "artifact_path"
                payload[key] = str(path)
                records.append(payload)
                if len(records) >= normalized_limit:
                    break
        return records

    @staticmethod
    def _candidate_fallback_paths(*, root: Path, normalized_limit: int | None) -> list[Path]:
        if normalized_limit is None:
            return sorted(root.rglob("*.json"), reverse=True)
        candidate_count = max(normalized_limit * 8, normalized_limit + 8, 32)
        candidates = nlargest(candidate_count, root.rglob("*.json"), key=lambda path: str(path))
        candidates.sort(key=lambda path: str(path), reverse=True)
        return candidates

    @staticmethod
    def _hydrate_db_record(row: dict[str, Any], *, path_key: str) -> dict[str, Any]:
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if isinstance(metadata, dict) and metadata:
            record_path = str(metadata.get(path_key) or "").strip()
            hydrated = dict(metadata)
            hydrated[path_key] = record_path or str(row.get("source_id") or "")
            hydrated["memory_item_id"] = row.get("id")
            hydrated["id"] = row.get("id")
            hydrated["source_id"] = row.get("source_id")
            return hydrated
        return {
            "memory_item_id": row.get("id"),
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "session_id": row.get("session_id"),
            "created_at": row.get("created_at"),
            "title": row.get("relevance_hint"),
            "content": row.get("content"),
            "normalized_topic": row.get("domain"),
            path_key: str(row.get("source_id") or ""),
            "archived": str(row.get("status") or "") == "archived",
        }
