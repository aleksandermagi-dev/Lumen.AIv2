from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.memory.classification import MemoryClassifier
from lumen.memory.graph_memory import GraphMemoryManager
from lumen.memory.personal_memory import PersonalMemoryManager
from lumen.memory.research_notes import ResearchNoteManager
from lumen.memory.memory_models import MemoryClassification
from lumen.memory.write_policy import MemoryWritePolicy
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.retrieval.semantic_matcher import SemanticCandidate, SemanticMatcher
from lumen.schemas.interaction_schema import InteractionRecordSchema
from lumen.tools.registry_types import ToolResult


class InteractionLogManager:
    """Persists lightweight records for assistant ask interactions."""

    INDEX_FILENAME = "_index.json"
    TARGET_RECORD_BYTES = 500 * 1024
    def __init__(
        self,
        settings: AppSettings,
        graph_memory_manager: GraphMemoryManager | None = None,
        persistence_manager: PersistenceManager | None = None,
    ):
        self.settings = settings
        self.interactions_root = settings.interactions_root
        self.max_record_bytes = settings.max_interaction_record_bytes
        self.prompt_nlu = PromptNLU()
        self.semantic_matcher = SemanticMatcher()
        self.memory_classifier = MemoryClassifier()
        self.memory_write_policy = MemoryWritePolicy()
        self.persistence_manager = persistence_manager or PersistenceManager(settings)
        self.personal_memory_manager = PersonalMemoryManager(settings, persistence_manager=self.persistence_manager)
        self.research_note_manager = ResearchNoteManager(settings, persistence_manager=self.persistence_manager)
        self.graph_memory_manager = graph_memory_manager

    def record_interaction(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now(UTC)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        target_dir = self.interactions_root / session_id
        target_dir.mkdir(parents=True, exist_ok=True)

        self.persistence_manager.ensure_session(
            session_id=session_id,
            prompt=prompt,
            created_at=timestamp.isoformat(),
            mode=str(response.get("mode") or "").strip() or None,
            title=prompt,
            active_topic=str(response.get("normalized_topic") or "").strip() or None,
            project_id=project_id,
            project_name=project_name,
            status="active",
        )

        record = InteractionRecordSchema.normalize(
            self._build_payload(
                session_id=session_id,
                prompt=prompt,
                response=response,
                timestamp=timestamp,
            )
        )
        InteractionRecordSchema.validate(record)

        interaction_path = target_dir / f"{stamp}.json"
        record, record_json = self._prepare_record_for_storage(record)
        interaction_path.write_text(record_json, encoding="utf-8")
        client_surface = str(response.get("client_surface") or "main")
        memory_write_decision = self.memory_write_policy.decide(
            classification=MemoryClassification.from_mapping(record.get("memory_classification")),
            client_surface=client_surface,
            mobile_research_note_auto_save=self.settings.mobile_research_note_auto_save,
        )
        record["memory_write_decision"] = memory_write_decision.to_dict()
        personal_memory = None
        research_note = None
        if memory_write_decision.save_personal_memory:
            personal_memory = self.personal_memory_manager.record_entry(
                session_id=session_id,
                timestamp=timestamp,
                record=record,
                client_surface=client_surface,
                source_interaction_path=str(interaction_path),
            )
        if memory_write_decision.save_research_note:
            research_note = self.research_note_manager.record_note(
                session_id=session_id,
                timestamp=timestamp,
                record=record,
                client_surface=client_surface,
                source_interaction_path=str(interaction_path),
            )
        if personal_memory is not None:
            record["personal_memory"] = personal_memory
        if research_note is not None:
            record["research_note"] = research_note
        if personal_memory is not None or research_note is not None or record.get("memory_write_decision"):
            record, record_json = self._prepare_record_for_storage(record)
            interaction_path.write_text(record_json, encoding="utf-8")
        if self.graph_memory_manager is not None:
            self.graph_memory_manager.ingest_interaction_memory(record=record)
        self._append_index_entry(record, interaction_path)
        record["interaction_path"] = str(interaction_path)
        self.persistence_manager.record_interaction(
            session_id=session_id,
            prompt=prompt,
            response=response,
            record=record,
            interaction_path=str(interaction_path),
            project_id=project_id,
            project_name=project_name,
        )
        return record

    def list_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            self.persistence_manager.bootstrap()
            records = self.persistence_manager.messages.list_interaction_records(
                session_id=session_id,
                project_id=project_id,
            )
            if records:
                return [self._hydrate_db_record(record) for record in records]
        except Exception:
            records = []

        root = self.interactions_root / session_id if session_id else self.interactions_root
        if not root.exists():
            return []

        self.persistence_manager.record_fallback_read("interaction structured read fallback")
        file_records: list[dict[str, Any]] = []
        for interaction_file in self._iter_record_paths(root):
            record = self._load_record_if_valid(interaction_file)
            if record is not None:
                file_records.append(record)
        return file_records

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        records = self.list_records(session_id=session_id)
        return {
            "session_id": session_id,
            "record_count": len(records),
            "records": records,
        }

    def index_status(self, *, session_id: str | None = None) -> dict[str, Any]:
        root = self.interactions_root / session_id if session_id else self.interactions_root
        if not root.exists():
            return {
                "scope": str(root),
                "record_file_count": 0,
                "indexed_record_count": 0,
                "index_file_count": 0,
                "legacy_record_count": 0,
                "coverage_ratio": 0.0,
            }

        record_paths = {
            path
            for path in root.rglob("*.json")
            if path.name != self.INDEX_FILENAME
        }
        indexed_paths = set(self._indexed_paths(root))
        legacy_record_count = len(record_paths - indexed_paths)
        indexed_record_count = len(record_paths & indexed_paths)
        record_file_count = len(record_paths)
        coverage_ratio = (indexed_record_count / record_file_count) if record_file_count else 0.0
        return {
            "scope": str(root),
            "record_file_count": record_file_count,
            "indexed_record_count": indexed_record_count,
            "index_file_count": len(list(root.rglob(self.INDEX_FILENAME))),
            "legacy_record_count": legacy_record_count,
            "coverage_ratio": round(coverage_ratio, 4),
        }

    def search_records(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        needle = query.strip().lower()
        records = self.list_records(session_id=session_id, project_id=project_id)
        if not needle:
            return records
        query_understanding = self.prompt_nlu.analyze(query)
        matches: list[tuple[int, int, dict[str, int], list[str], dict[str, Any]]] = []
        candidate_records = records

        for record in candidate_records:
            score, semantic_score, score_breakdown, matched_fields = self._score_record(
                record,
                needle,
                query_understanding=query_understanding,
            )
            if score > 0:
                matches.append((score, semantic_score, score_breakdown, matched_fields, record))

        matches.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[4].get("created_at", ""),
            ),
            reverse=True,
        )
        return [
            {
                "score": score,
                "score_breakdown": score_breakdown,
                "matched_fields": matched_fields,
                "record": record,
            }
            for score, _, score_breakdown, matched_fields, record in matches
        ]

    def load_record(self, path: Path) -> dict[str, Any]:
        record = self._load_db_record_for_path(path)
        if record is not None:
            return record
        self.persistence_manager.record_fallback_read("interaction structured read fallback")
        return self._load_file_record(path)

    def _load_record_if_valid(self, path: Path) -> dict[str, Any] | None:
        try:
            if path.stat().st_size > self.max_record_bytes:
                return None
            return self._load_file_record(path)
        except (json.JSONDecodeError, MemoryError, OSError, ValueError):
            return None

    def _hydrate_db_record(self, record: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(record)
        interaction_path = str(hydrated.get("interaction_path") or "").strip()
        if "interaction_path" not in hydrated or not hydrated.get("interaction_path"):
            hydrated["interaction_path"] = interaction_path or None
        return hydrated

    def _load_db_record_for_path(self, path: Path) -> dict[str, Any] | None:
        try:
            self.persistence_manager.bootstrap()
        except Exception:
            return None
        session_id = path.parent.name
        message_id = f"{session_id}:{path.stem}:assistant"
        message = self.persistence_manager.messages.get(message_id)
        if not isinstance(message, dict):
            return None
        metadata = message.get("message_metadata_json")
        if not isinstance(metadata, dict):
            return None
        record = metadata.get("interaction_record")
        if not isinstance(record, dict):
            return None
        return self._hydrate_db_record(record)

    def _load_file_record(self, path: Path) -> dict[str, Any]:
        payload = InteractionRecordSchema.normalize(
            json.loads(path.read_text(encoding="utf-8"))
        )
        payload["context"] = self._sanitize_context(payload.get("context", {}))
        response = payload.get("response")
        if isinstance(response, dict):
            payload["response"] = self._serialize_response(
                response,
                sanitized_context=payload["context"],
            )
            payload["confidence_posture"] = str(
                payload.get("confidence_posture") or response.get("confidence_posture") or ""
            ).strip() or None
        InteractionRecordSchema.validate(payload)
        payload["interaction_path"] = str(path)
        return payload

    def _iter_record_paths(self, root: Path) -> list[Path]:
        indexed_paths = self._indexed_paths(root)
        extras = sorted(
            (
                path
                for path in root.rglob("*.json")
                if path.name != self.INDEX_FILENAME and path not in indexed_paths
            ),
            reverse=True,
        )
        return indexed_paths + extras

    def _indexed_paths(self, root: Path) -> list[Path]:
        entries = self._load_recursive_index_entries(root)
        paths: list[Path] = []
        for entry_root, entry in entries:
            path = entry_root / str(entry.get("path", ""))
            if path.exists():
                paths.append(path)
        paths.sort(reverse=True)
        return paths

    def _append_index_entry(self, record: dict[str, Any], interaction_path: Path) -> None:
        index_path = self._index_path(interaction_path.parent)
        entries = self._load_index(interaction_path.parent)
        relative_path = interaction_path.relative_to(interaction_path.parent).as_posix()
        entries = [entry for entry in entries if entry.get("path") != relative_path]
        semantic_signature = self.semantic_matcher.signature_from_candidate(
            SemanticCandidate(
                prompt=InteractionLogManager._canonical_prompt(record),
                normalized_topic=str(record.get("normalized_topic") or "").strip() or None,
                dominant_intent=str(record.get("dominant_intent") or "").strip() or None,
                extracted_entities=self._entity_values(record.get("extracted_entities")),
            )
        )
        entries.append(
            {
                "path": relative_path,
                "session_id": record.get("session_id"),
                "prompt": record.get("prompt"),
                "resolved_prompt": record.get("resolved_prompt"),
                "summary": record.get("summary"),
                "mode": record.get("mode"),
                "kind": record.get("kind"),
                "confidence_posture": record.get("confidence_posture"),
                "tool_route_origin": record.get("tool_route_origin"),
                "detected_language": record.get("detected_language"),
                "normalized_topic": record.get("normalized_topic"),
                "dominant_intent": record.get("dominant_intent"),
                "extracted_entities": self._entity_values(record.get("extracted_entities")),
                "semantic_signature": semantic_signature.to_dict(),
                "resolution_strategy": record.get("resolution_strategy"),
                "created_at": record.get("created_at"),
            }
        )
        entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def _load_index(self, root: Path) -> list[dict[str, Any]]:
        index_path = self._index_path(root)
        if not index_path.exists():
            return []
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _search_index(
        self,
        needle: str,
        *,
        query_understanding,
        session_id: str | None,
    ) -> list[tuple[int, int, dict[str, int], Path]]:
        root = self.interactions_root / session_id if session_id else self.interactions_root
        entries = self._load_recursive_index_entries(root)
        matches: list[tuple[int, int, dict[str, int], Path]] = []
        for entry_root, entry in entries:
            score, semantic_score, score_breakdown = self._score_index_entry(
                entry,
                needle,
                query_understanding=query_understanding,
            )
            if score <= 0:
                continue
            path = entry_root / str(entry.get("path", ""))
            if path.exists():
                matches.append((score, semantic_score, score_breakdown, path))
        matches.sort(
            key=lambda item: (
                item[0],
                item[1],
                self._indexed_created_at(item[3]),
                str(item[3]),
            ),
            reverse=True,
        )
        return matches[: self.settings.search_candidate_limit]

    def _score_index_entry(
        self,
        entry: dict[str, Any],
        needle: str,
        *,
        query_understanding,
    ) -> tuple[int, int, dict[str, int]]:
        keyword_score = 0
        field_weights = {
            "prompt": 5,
            "resolved_prompt": 5,
            "normalized_topic": 4,
            "dominant_intent": 2,
            "summary": 4,
            "mode": 2,
            "kind": 2,
            "resolution_strategy": 2,
        }
        for field, weight in field_weights.items():
            value = entry.get(field)
            haystack = str(value or "").lower()
            if needle in haystack:
                keyword_score += weight
        semantic_score = self._score_nlu_metadata(
            prompt=InteractionLogManager._canonical_prompt(entry),
            dominant_intent=str(entry.get("dominant_intent") or "").strip() or None,
            normalized_topic=str(entry.get("normalized_topic") or "").strip() or None,
            extracted_entities=self._entity_values(entry.get("extracted_entities")),
            semantic_signature=entry.get("semantic_signature"),
            query_understanding=query_understanding,
        )
        score_breakdown = {
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
        }
        return self._blend_search_score(keyword_score, semantic_score), semantic_score, score_breakdown

    def _index_path(self, root: Path) -> Path:
        return root / self.INDEX_FILENAME

    def _load_recursive_index_entries(self, root: Path) -> list[tuple[Path, dict[str, Any]]]:
        entries: list[tuple[Path, dict[str, Any]]] = []
        if not root.exists():
            return entries
        for index_path in root.rglob(self.INDEX_FILENAME):
            index_root = index_path.parent
            for entry in self._load_index(index_root):
                entries.append((index_root, entry))
        return entries

    @staticmethod
    def _indexed_created_at(path: Path) -> str:
        return path.stem

    def _build_payload(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        timestamp: datetime,
    ) -> dict[str, object]:
        sanitized_context = self._sanitize_context(response.get("context", {}))
        canonical_prompt = str(response.get("resolved_prompt") or prompt)
        understanding = self.prompt_nlu.analyze(canonical_prompt)
        memory_classification = self.memory_classifier.classify(
            prompt=prompt,
            resolved_prompt=str(response.get("resolved_prompt") or "").strip() or None,
            mode=str(response.get("mode", "unknown")),
            dominant_intent=understanding.intent.label,
            summary=str(response.get("summary", "")),
        )
        return {
            "session_id": session_id,
            "prompt": prompt,
            "mode": response.get("mode", "unknown"),
            "kind": response.get("kind", "unknown"),
            "summary": response.get("summary", ""),
            "confidence_posture": response.get("confidence_posture"),
            "route_status": response.get("route_status"),
            "support_status": response.get("support_status"),
            "tension_status": response.get("tension_status"),
            "tool_route_origin": response.get("tool_route_origin"),
            "local_context_assessment": response.get("local_context_assessment"),
            "coherence_topic": (response.get("reasoning_frame") or {}).get("coherence_topic"),
            "interaction_profile": response.get("interaction_profile", {}),
            "profile_advice": response.get("profile_advice"),
            "memory_classification": memory_classification.to_dict(),
            "memory_write_decision": {},
            "personal_memory": None,
            "pipeline_observability": response.get("pipeline_observability", {}),
            "pipeline_trace": response.get("pipeline_trace", {}),
            "trainability_trace": response.get("trainability_trace", {}),
            "supervised_support_trace": response.get("supervised_support_trace", {}),
            "client_surface": str(response.get("client_surface") or "main"),
            "resolved_prompt": response.get("resolved_prompt"),
            "detected_language": understanding.language.code,
            "normalized_topic": understanding.topic.value,
            "dominant_intent": understanding.intent.label,
            "extracted_entities": [entity.to_dict() for entity in understanding.entities],
            "resolution_strategy": response.get("resolution_strategy"),
            "resolution_reason": response.get("resolution_reason"),
            "route": response.get("route", {}),
            "context": sanitized_context,
            "created_at": timestamp.isoformat(),
            "response": self._serialize_response(response, sanitized_context=sanitized_context),
        }

    def _prepare_record_for_storage(self, record: dict[str, object]) -> tuple[dict[str, object], str]:
        prepared = record
        serialized = self._safe_json_dumps(prepared)
        target_bytes = min(self.max_record_bytes, self.TARGET_RECORD_BYTES)
        if serialized is None or len(serialized.encode("utf-8")) > target_bytes:
            prepared = self._compact_record_for_storage(prepared)
            serialized = self._safe_json_dumps(prepared)
        if serialized is None or len(serialized.encode("utf-8")) > target_bytes:
            prepared = self._minimal_record_for_storage(prepared)
            serialized = self._safe_json_dumps(prepared)
        if serialized is None:
            raise MemoryError("Unable to serialize interaction record for storage safely.")
        return prepared, serialized

    @staticmethod
    def _safe_json_dumps(payload: object) -> str | None:
        try:
            return json.dumps(payload, indent=2)
        except (MemoryError, TypeError, ValueError):
            return None

    def _compact_record_for_storage(self, record: dict[str, object]) -> dict[str, object]:
        compact = dict(record)
        compact["context"] = self._compact_serialized_value(compact.get("context"), label="context", max_chars=50_000)
        compact["pipeline_observability"] = self._compact_serialized_value(
            compact.get("pipeline_observability"),
            label="pipeline_observability",
            max_chars=50_000,
        )
        compact["pipeline_trace"] = self._compact_serialized_value(
            compact.get("pipeline_trace"),
            label="pipeline_trace",
            max_chars=50_000,
        )
        compact["trainability_trace"] = self._compact_serialized_value(
            compact.get("trainability_trace"),
            label="trainability_trace",
            max_chars=30_000,
        )
        compact["supervised_support_trace"] = self._compact_serialized_value(
            compact.get("supervised_support_trace"),
            label="supervised_support_trace",
            max_chars=30_000,
        )
        response = compact.get("response")
        if isinstance(response, dict):
            compact_response = dict(response)
            for key in (
                "context",
                "tool_execution",
                "tool_runtime_status",
                "pipeline_trace",
                "pipeline_observability",
                "trainability_trace",
                "supervised_support_trace",
                "reasoning_state",
                "reasoning_frame",
                "route",
            ):
                if key in compact_response:
                    compact_response[key] = self._compact_serialized_value(
                        compact_response.get(key),
                        label=f"response.{key}",
                        max_chars=50_000,
                    )
            compact["response"] = compact_response
        return compact

    def _minimal_record_for_storage(self, record: dict[str, object]) -> dict[str, object]:
        minimal = dict(record)
        minimal["context"] = {}
        minimal["pipeline_observability"] = {}
        minimal["pipeline_trace"] = {}
        minimal["trainability_trace"] = self._compact_serialized_value(
            minimal.get("trainability_trace"),
            label="trainability_trace",
            max_chars=20_000,
        )
        minimal["supervised_support_trace"] = self._compact_serialized_value(
            minimal.get("supervised_support_trace"),
            label="supervised_support_trace",
            max_chars=20_000,
        )
        response = minimal.get("response")
        if isinstance(response, dict):
            minimal["response"] = {
                "schema_type": response.get("schema_type", "assistant_response"),
                "mode": response.get("mode"),
                "kind": response.get("kind"),
                "summary": response.get("summary"),
                "tool_route_origin": response.get("tool_route_origin"),
                "tool_execution": self._compact_serialized_value(
                    response.get("tool_execution"),
                    label="response.tool_execution",
                    max_chars=20_000,
                ),
                "tool_result": self._compact_serialized_value(
                    response.get("tool_result"),
                    label="response.tool_result",
                    max_chars=20_000,
                ),
            }
        return minimal

    def _serialize_response(
        self,
        response: dict[str, object],
        *,
        sanitized_context: dict[str, object],
    ) -> dict[str, object]:
        serialized = dict(response)
        serialized["context"] = sanitized_context
        tool_result = serialized.get("tool_result")
        if isinstance(tool_result, ToolResult):
            serialized["tool_result"] = self._serialize_tool_result(tool_result)
        return serialized

    def _sanitize_context(self, context: object) -> dict[str, object]:
        if not isinstance(context, dict):
            return {}

        sanitized = dict(context)
        top_matches = sanitized.get("top_matches")
        if isinstance(top_matches, list):
            sanitized["top_matches"] = [
                self._sanitize_archive_match(match) for match in top_matches if isinstance(match, dict)
            ]

        top_interaction_matches = sanitized.get("top_interaction_matches")
        if isinstance(top_interaction_matches, list):
            sanitized["top_interaction_matches"] = [
                self._sanitize_interaction_match(match)
                for match in top_interaction_matches
                if isinstance(match, dict)
            ]

        active_thread = sanitized.get("active_thread")
        if isinstance(active_thread, dict):
            sanitized["active_thread"] = self._sanitize_active_thread_reference(active_thread)

        return sanitized

    @staticmethod
    def _sanitize_archive_match(match: dict[str, object]) -> dict[str, object]:
        sanitized = dict(match)
        record = sanitized.get("record")
        if isinstance(record, dict):
            sanitized["record"] = dict(record)
        return sanitized

    def _sanitize_interaction_match(self, match: dict[str, object]) -> dict[str, object]:
        sanitized = dict(match)
        record = sanitized.get("record")
        if isinstance(record, dict):
            sanitized["record"] = self._sanitize_interaction_record_reference(record)
        return sanitized

    def _sanitize_interaction_record_reference(self, record: dict[str, object]) -> dict[str, object]:
        sanitized = dict(record)
        sanitized.pop("response", None)
        sanitized.pop("pipeline_trace", None)
        sanitized.pop("pipeline_observability", None)
        sanitized.pop("trainability_trace", None)
        sanitized.pop("supervised_support_trace", None)
        nested_context = sanitized.get("context")
        if isinstance(nested_context, dict):
            nested_context = dict(nested_context)
            nested_context.pop("top_interaction_matches", None)
            nested_context.pop("active_thread", None)
            sanitized["context"] = nested_context
        return sanitized

    @staticmethod
    def _sanitize_active_thread_reference(active_thread: dict[str, object]) -> dict[str, object]:
        allowed = {
            "session_id",
            "mode",
            "kind",
            "prompt",
            "original_prompt",
            "objective",
            "thread_summary",
            "summary",
            "confidence_posture",
            "tool_route_origin",
            "local_context_assessment",
            "normalized_topic",
            "dominant_intent",
            "intent_domain",
            "conversation_phase",
            "continuation_offer",
            "tool_context",
            "updated_at",
        }
        sanitized = {key: active_thread.get(key) for key in allowed if key in active_thread}
        offer = sanitized.get("continuation_offer")
        if isinstance(offer, dict):
            sanitized["continuation_offer"] = {
                key: offer.get(key)
                for key in ("kind", "topic", "target_prompt", "label", "explanation_mode")
                if offer.get(key)
            }
        tool_context = sanitized.get("tool_context")
        if isinstance(tool_context, dict):
            sanitized["tool_context"] = {
                key: tool_context.get(key)
                for key in ("tool_id", "capability", "input_path", "status")
                if tool_context.get(key)
            }
        return sanitized

    def _score_record(
        self,
        record: dict[str, Any],
        needle: str,
        *,
        query_understanding,
    ) -> tuple[int, int, dict[str, int], list[str]]:
        keyword_score = 0
        matched_fields: list[str] = []
        canonical_prompt = InteractionLogManager._canonical_prompt(record)
        if canonical_prompt and needle in canonical_prompt.lower():
            keyword_score += 6
            matched_fields.append("canonical_prompt")
        field_weights = {
            "prompt": 5,
            "summary": 4,
            "resolved_prompt": 4,
            "resolution_strategy": 2,
            "resolution_reason": 2,
            "mode": 2,
            "kind": 2,
            "response": 1,
        }
        for field, weight in field_weights.items():
            value = record.get(field)
            haystack = value.lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()
            if needle in haystack:
                keyword_score += weight
                matched_fields.append(field)
        semantic_score = self._score_nlu_metadata(
            prompt=canonical_prompt,
            dominant_intent=str(record.get("dominant_intent") or "").strip() or None,
            normalized_topic=str(record.get("normalized_topic") or "").strip() or None,
            extracted_entities=self._entity_values(record.get("extracted_entities")),
            semantic_signature=None,
            query_understanding=query_understanding,
        )
        if semantic_score > 0:
            matched_fields.append("semantic")
        score_breakdown = {
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
        }
        return (
            self._blend_search_score(keyword_score, semantic_score),
            semantic_score,
            score_breakdown,
            matched_fields,
        )

    @staticmethod
    def _blend_search_score(keyword_score: int, semantic_score: int) -> int:
        if keyword_score <= 0:
            return semantic_score
        semantic_bonus = min(semantic_score, max(2, min(6, keyword_score // 2 + 1)))
        return keyword_score + semantic_bonus

    @staticmethod
    def _entity_values(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        extracted: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("value") is not None:
                extracted.append(str(item.get("value")).strip().lower())
            elif isinstance(item, str):
                extracted.append(item.strip().lower())
        return tuple(item for item in extracted if item)

    def _score_nlu_metadata(
        self,
        *,
        prompt: str | None,
        dominant_intent: str | None,
        normalized_topic: str | None,
        extracted_entities: tuple[str, ...],
        semantic_signature: object | None,
        query_understanding,
    ) -> int:
        if isinstance(semantic_signature, dict):
            match = self.semantic_matcher.score_signature(
                query_understanding,
                self._signature_from_index_entry(
                    semantic_signature,
                    dominant_intent=dominant_intent,
                    extracted_entities=extracted_entities,
                ),
            )
        else:
            match = self.semantic_matcher.score(
                query_understanding,
                SemanticCandidate(
                    prompt=prompt,
                    normalized_topic=normalized_topic,
                    dominant_intent=dominant_intent,
                    extracted_entities=extracted_entities,
                ),
            )
        return match.score

    @staticmethod
    def _signature_from_index_entry(
        signature: dict[str, object],
        *,
        dominant_intent: str | None,
        extracted_entities: tuple[str, ...],
    ):
        from lumen.retrieval.semantic_matcher import SemanticSignature

        prompt_tokens = tuple(
            str(item).strip().lower()
            for item in (signature.get("prompt_tokens") or [])
            if str(item).strip()
        )
        topic_tokens = tuple(
            str(item).strip().lower()
            for item in (signature.get("topic_tokens") or [])
            if str(item).strip()
        )
        entities = tuple(
            str(item).strip().lower()
            for item in (signature.get("entities") or [])
            if str(item).strip()
        ) or extracted_entities
        return SemanticSignature(
            prompt_tokens=prompt_tokens,
            topic_tokens=topic_tokens,
            dominant_intent=str(signature.get("dominant_intent") or dominant_intent or "").strip().lower() or None,
            entities=entities,
        )

    @staticmethod
    def _canonical_prompt(record: dict[str, Any]) -> str:
        resolved_prompt = str(record.get("resolved_prompt") or "").strip()
        if resolved_prompt:
            return resolved_prompt
        return str(record.get("prompt") or "").strip()

    @staticmethod
    def _serialize_tool_result(result: ToolResult) -> dict[str, object]:
        return {
            "status": result.status,
            "tool_id": result.tool_id,
            "capability": result.capability,
            "summary": result.summary,
            "run_dir": str(result.run_dir) if result.run_dir else None,
            "archive_path": str(result.archive_path) if result.archive_path else None,
            "error": result.error,
            "structured_data": InteractionLogManager._compact_serialized_value(
                result.structured_data,
                label="structured_data",
            ),
            "artifacts": [
                {
                    "name": artifact.name,
                    "path": str(artifact.path),
                    "media_type": artifact.media_type,
                    "description": artifact.description,
                }
                for artifact in result.artifacts
            ],
            "logs": InteractionLogManager._compact_serialized_value(result.logs, label="logs"),
            "provenance": InteractionLogManager._compact_serialized_value(
                result.provenance,
                label="provenance",
            ),
        }

    @staticmethod
    def _compact_serialized_value(value: object, *, label: str, max_chars: int = 200_000) -> object:
        try:
            serialized = json.dumps(value, sort_keys=True)
        except (MemoryError, TypeError, ValueError):
            serialized = None
        if serialized is not None and len(serialized) <= max_chars:
            return value

        summary: dict[str, object] = {
            "compacted": True,
            "label": label,
            "detail": "Value omitted from interaction-log storage because it was too large to persist safely.",
        }
        if serialized is not None:
            summary["approx_chars"] = len(serialized)
        if isinstance(value, dict):
            summary["top_level_keys"] = list(value.keys())[:20]
        elif isinstance(value, list):
            summary["item_count"] = len(value)
            if value and all(isinstance(item, dict) for item in value[:5]):
                summary["sample_keys"] = list(
                    {
                        key
                        for item in value[:5]
                        for key in item.keys()
                    }
                )[:20]
        elif value is not None:
            summary["value_type"] = type(value).__name__
        return summary
