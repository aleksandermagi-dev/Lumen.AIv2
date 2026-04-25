from __future__ import annotations

from datetime import datetime
from heapq import nlargest
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.schemas.personal_memory_schema import PersonalMemorySchema


class PersonalMemoryManager:
    """Persists explicitly user-saved personal memory entries."""

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager | None = None):
        self.settings = settings
        self.personal_memory_root = settings.personal_memory_root
        self.persistence_manager = persistence_manager or PersistenceManager(settings)

    def maybe_record_entry(
        self,
        *,
        session_id: str,
        timestamp: datetime,
        record: dict[str, Any],
        client_surface: str = "main",
        source_interaction_path: str | None = None,
    ) -> dict[str, Any] | None:
        memory_classification = dict(record.get("memory_classification") or {})
        if str(memory_classification.get("candidate_type") or "") != "personal_context_candidate":
            return None
        if not bool(memory_classification.get("explicit_save_requested")):
            return None
        return self.record_entry(
            session_id=session_id,
            timestamp=timestamp,
            record=record,
            client_surface=client_surface,
            source_interaction_path=source_interaction_path,
        )

    def record_entry(
        self,
        *,
        session_id: str,
        timestamp: datetime,
        record: dict[str, Any],
        client_surface: str = "main",
        source_interaction_path: str | None = None,
        title_override: str | None = None,
        content_override: str | None = None,
        memory_origin: str = "user",
    ) -> dict[str, Any]:
        memory_classification = dict(record.get("memory_classification") or {})

        target_dir = self.personal_memory_root / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        entry_path = target_dir / f"{stamp}.json"
        payload = PersonalMemorySchema.normalize(
            {
                "session_id": session_id,
                "created_at": timestamp.isoformat(),
                "title": title_override or self._title_for_record(record),
                "content": content_override or self._content_for_record(record),
                "normalized_topic": record.get("normalized_topic"),
                "memory_classification": memory_classification,
                "source_interaction_prompt": record.get("prompt"),
                "source_interaction_mode": record.get("mode"),
                "source_interaction_kind": record.get("kind"),
                "source_interaction_path": source_interaction_path,
                "source_interaction_summary": record.get("summary"),
                "client_surface": client_surface,
                "memory_origin": str(memory_origin or "user").strip() or "user",
            }
        )
        PersonalMemorySchema.validate(payload)
        entry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["entry_path"] = str(entry_path)
        self.persistence_manager.record_memory_item(source_type="personal_memory", payload=payload)
        return payload

    def list_entries(
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
            source_type="personal_memory",
            include_archived=include_archived,
            archived_only=archived_only,
            limit=normalized_limit,
        )
        if db_rows:
            return [self._hydrate_db_entry(row) for row in db_rows]

        root = self.personal_memory_root / session_id if session_id else self.personal_memory_root
        if not root.exists():
            return []
        self.persistence_manager.record_fallback_read("personal memory structured listing fallback")
        records: list[dict[str, Any]] = []
        seen_paths: set[Path] = set()
        for entry_path in self._candidate_fallback_paths(root=root, normalized_limit=normalized_limit):
            seen_paths.add(entry_path)
            payload = self._load_entry_payload(entry_path)
            if payload is None:
                continue
            payload["entry_path"] = str(entry_path)
            is_archived = bool(payload.get("archived", False))
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            records.append(payload)
            if normalized_limit is not None and len(records) >= normalized_limit:
                break
        if normalized_limit is not None and len(records) < normalized_limit:
            for entry_path in sorted(root.rglob("*.json"), reverse=True):
                if entry_path in seen_paths:
                    continue
                payload = self._load_entry_payload(entry_path)
                if payload is None:
                    continue
                payload["entry_path"] = str(entry_path)
                is_archived = bool(payload.get("archived", False))
                if archived_only and not is_archived:
                    continue
                if not include_archived and is_archived:
                    continue
                records.append(payload)
                if len(records) >= normalized_limit:
                    break
        return records

    def load_entry(self, entry_path: str | Path) -> dict[str, Any]:
        path = self._resolve_entry_path(str(entry_path))
        self.persistence_manager.record_fallback_read("personal memory raw hydration")
        payload = self._load_entry_payload(path)
        if payload is None:
            raise FileNotFoundError(f"Personal memory entry not found: {entry_path}")
        payload["entry_path"] = str(path)
        return payload

    def _load_entry_payload(self, entry_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _hydrate_db_entry(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if isinstance(metadata, dict) and metadata:
            entry_path = str(metadata.get("entry_path") or "").strip()
            hydrated = dict(metadata)
            hydrated["entry_path"] = entry_path or str(row.get("source_id") or "")
            hydrated["memory_item_id"] = row.get("id")
            hydrated["id"] = row.get("id")
            hydrated["source_id"] = row.get("source_id")
            return hydrated
        return {
            "memory_item_id": row.get("id"),
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "session_id": row.get("session_id"),
            "title": row.get("relevance_hint"),
            "content": row.get("content"),
            "normalized_topic": row.get("domain"),
            "created_at": row.get("created_at"),
            "entry_path": str(row.get("source_id") or ""),
            "archived": str(row.get("status") or "") == "archived",
        }

    def archive_entry(self, *, entry_path: str) -> dict[str, Any]:
        try:
            path = self._resolve_entry_path(entry_path)
        except FileNotFoundError:
            row = self.persistence_manager.memory_items.update_status_by_identity(entry_path, status="archived")
            if row is not None:
                return self._hydrate_db_entry(row)
            raise
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid personal memory payload.")
        payload["archived"] = True
        payload["archived_at"] = datetime.now().isoformat()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.persistence_manager.memory_items.update_status_by_identity(str(path), status="archived")
        payload["entry_path"] = str(path)
        return payload

    def delete_entry(self, *, entry_path: str) -> dict[str, Any]:
        try:
            path = self._resolve_entry_path(entry_path)
        except FileNotFoundError:
            deleted = self.persistence_manager.memory_items.delete_by_identity(entry_path)
            if deleted:
                return {
                    "entry_path": entry_path,
                    "memory_item_id": entry_path,
                    "deleted": True,
                }
            raise
        existed = path.exists()
        if existed:
            path.unlink()
        self.persistence_manager.memory_items.delete_by_identity(str(path))
        return {
            "entry_path": str(path),
            "deleted": existed,
        }

    def _resolve_entry_path(self, entry_path: str) -> Path:
        path = self.personal_memory_root / Path(entry_path).name if not Path(entry_path).is_absolute() else Path(entry_path)
        if not path.exists():
            raise FileNotFoundError(f"Personal memory entry not found: {entry_path}")
        return path

    @staticmethod
    def _candidate_fallback_paths(*, root: Path, normalized_limit: int | None) -> list[Path]:
        if normalized_limit is None:
            return sorted(root.rglob("*.json"), reverse=True)
        candidate_count = max(normalized_limit * 8, normalized_limit + 8, 32)
        candidates = nlargest(candidate_count, root.rglob("*.json"), key=lambda path: str(path))
        candidates.sort(key=lambda path: str(path), reverse=True)
        return candidates

    @staticmethod
    def _title_for_record(record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt") or "").strip()
        if ":" in prompt:
            return prompt.split(":", 1)[0].strip()
        return prompt or "personal memory"

    @staticmethod
    def _content_for_record(record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt") or "").strip()
        summary = str(record.get("summary") or "").strip()
        pieces = [
            f"Prompt: {prompt}",
            f"Summary: {summary}",
        ]
        return "\n".join(piece for piece in pieces if piece.rstrip(": ").strip())
