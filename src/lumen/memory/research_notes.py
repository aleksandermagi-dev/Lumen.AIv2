from __future__ import annotations

from datetime import datetime
from heapq import nlargest
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.schemas.research_note_schema import ResearchNoteSchema


class ResearchNoteManager:
    """Persists conservative chronological research notes."""

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager | None = None):
        self.settings = settings
        self.research_notes_root = settings.research_notes_root
        self.persistence_manager = persistence_manager or PersistenceManager(settings)

    def maybe_record_note(
        self,
        *,
        session_id: str,
        timestamp: datetime,
        record: dict[str, Any],
        client_surface: str = "main",
        source_interaction_path: str | None = None,
    ) -> dict[str, Any] | None:
        memory_classification = dict(record.get("memory_classification") or {})
        if not bool(memory_classification.get("save_eligible")):
            return None
        if str(memory_classification.get("candidate_type") or "") != "research_memory_candidate":
            return None
        if client_surface == "mobile" and not self.settings.mobile_research_note_auto_save:
            return None
        return self.record_note(
            session_id=session_id,
            timestamp=timestamp,
            record=record,
            client_surface=client_surface,
            source_interaction_path=source_interaction_path,
        )

    def record_note(
        self,
        *,
        session_id: str,
        timestamp: datetime,
        record: dict[str, Any],
        client_surface: str = "main",
        source_interaction_path: str | None = None,
    ) -> dict[str, Any]:
        memory_classification = dict(record.get("memory_classification") or {})

        target_dir = self.research_notes_root / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        note_path = target_dir / f"{stamp}.json"
        payload = ResearchNoteSchema.normalize(
            {
                "session_id": session_id,
                "created_at": timestamp.isoformat(),
                "note_type": "chronological_research_note",
                "title": self._title_for_record(record),
                "content": self._content_for_record(record),
                "normalized_topic": record.get("normalized_topic"),
                "dominant_intent": record.get("dominant_intent"),
                "memory_classification": memory_classification,
                "interaction_profile": dict(record.get("interaction_profile") or {}),
                "client_surface": client_surface,
                "source_interaction_prompt": record.get("prompt"),
                "source_interaction_mode": record.get("mode"),
                "source_interaction_kind": record.get("kind"),
                "source_interaction_path": source_interaction_path,
            }
        )
        ResearchNoteSchema.validate(payload)
        note_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["note_path"] = str(note_path)
        self.persistence_manager.record_memory_item(source_type="research_note", payload=payload)
        return payload

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
            return [self._hydrate_db_note(row) for row in db_rows]
        root = self.research_notes_root / session_id if session_id else self.research_notes_root
        if not root.exists():
            return []
        self.persistence_manager.record_fallback_read("research note structured listing fallback")
        notes: list[dict[str, Any]] = []
        seen_paths: set[Path] = set()
        for note_path in self._candidate_fallback_paths(root=root, normalized_limit=normalized_limit):
            seen_paths.add(note_path)
            payload = self._load_note_payload(note_path)
            if payload is None:
                continue
            is_archived = bool(payload.get("archived", False))
            if archived_only and not is_archived:
                continue
            if not include_archived and is_archived:
                continue
            payload["note_path"] = str(note_path)
            notes.append(payload)
            if normalized_limit is not None and len(notes) >= normalized_limit:
                break
        if normalized_limit is not None and len(notes) < normalized_limit:
            for note_path in sorted(root.rglob("*.json"), reverse=True):
                if note_path in seen_paths:
                    continue
                payload = self._load_note_payload(note_path)
                if payload is None:
                    continue
                is_archived = bool(payload.get("archived", False))
                if archived_only and not is_archived:
                    continue
                if not include_archived and is_archived:
                    continue
                payload["note_path"] = str(note_path)
                notes.append(payload)
                if len(notes) >= normalized_limit:
                    break
        return notes

    def load_note(self, note_path: str | Path) -> dict[str, Any]:
        path = Path(note_path)
        if not path.is_absolute():
            path = self.research_notes_root / path.name
        self.persistence_manager.record_fallback_read("research note raw hydration")
        payload = self._load_note_payload(path)
        if payload is None:
            raise FileNotFoundError(f"Research note not found: {note_path}")
        payload["note_path"] = str(path)
        return payload

    @staticmethod
    def _load_note_payload(note_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(note_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _hydrate_db_note(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if isinstance(metadata, dict) and metadata:
            note_path = str(metadata.get("note_path") or "").strip()
            hydrated = dict(metadata)
            hydrated["note_path"] = note_path or str(row.get("source_id") or "")
            return hydrated
        return {
            "session_id": row.get("session_id"),
            "created_at": row.get("created_at"),
            "note_type": "chronological_research_note",
            "title": row.get("relevance_hint"),
            "content": row.get("content"),
            "normalized_topic": row.get("domain"),
            "note_path": str(row.get("source_id") or ""),
            "archived": str(row.get("status") or "") == "archived",
        }

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
        topic = str(record.get("normalized_topic") or "").strip()
        summary = str(record.get("summary") or "").strip()
        if topic:
            return topic
        if summary:
            return summary
        return str(record.get("prompt") or "research note").strip()

    @staticmethod
    def _content_for_record(record: dict[str, Any]) -> str:
        prompt = str(record.get("prompt") or "").strip()
        summary = str(record.get("summary") or "").strip()
        mode = str(record.get("mode") or "").strip()
        kind = str(record.get("kind") or "").strip()
        pieces = [
            f"Prompt: {prompt}",
            f"Mode: {mode}",
            f"Kind: {kind}",
            f"Summary: {summary}",
        ]
        return "\n".join(piece for piece in pieces if piece.rstrip(": ").strip())
