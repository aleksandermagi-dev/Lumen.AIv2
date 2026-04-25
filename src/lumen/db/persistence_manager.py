from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.db.database_manager import DatabaseManager
from lumen.db.graph_repository import GraphRepository
from lumen.db.migration_runner import MigrationRunner
from lumen.db.project_resolver import ProjectResolver
from lumen.db.repositories import (
    BugLogRepository,
    DatasetExampleLabelRepository,
    DatasetExampleRepository,
    DatasetImportRunRepository,
    MemoryItemEmbeddingRepository,
    MemoryRepository,
    MessageRepository,
    PreferenceRepository,
    ProjectRepository,
    SessionRepository,
    SessionSummaryRepository,
    ToolRunRepository,
    TrainabilityTraceRepository,
)
from lumen.db.schema_manager import SchemaManager
from lumen.knowledge.knowledge_db import KnowledgeDB
from lumen.schemas.archive_schema import ArchiveRecordSchema
from lumen.schemas.interaction_schema import InteractionRecordSchema
from lumen.schemas.personal_memory_schema import PersonalMemorySchema
from lumen.schemas.research_artifact_schema import ResearchArtifactSchema
from lumen.schemas.research_note_schema import ResearchNoteSchema
from lumen.schemas.session_thread_schema import SessionThreadSchema
from lumen.semantic.embedding_service import SemanticEmbeddingService


class PersistenceManager:
    """Unified SQLite persistence facade for Lumen's structured state."""

    STRUCTURED_READ_SURFACES: dict[str, str] = {
        "interaction structured read fallback": "compat_fallback",
        "archive structured metadata fallback": "compat_fallback",
        "session metadata fallback": "compat_fallback",
        "session profile fallback": "compat_fallback",
        "session thread fallback": "compat_fallback",
        "personal memory structured listing fallback": "compat_fallback",
        "research note structured listing fallback": "compat_fallback",
        "research artifact structured listing fallback": "compat_fallback",
        "archive raw artifact hydration": "raw_hydration",
        "personal memory raw hydration": "raw_hydration",
        "research note raw hydration": "raw_hydration",
        "research artifact raw hydration": "raw_hydration",
        "interaction records": "db_primary",
        "session state": "db_primary",
        "archive metadata": "db_primary",
        "memory listings": "db_primary",
    }
    STRUCTURED_SURFACE_READINESS: dict[str, dict[str, object]] = {
        "interaction structured read fallback": {"db_parity_expected": True, "recovery_critical": False},
        "archive structured metadata fallback": {"db_parity_expected": True, "recovery_critical": False},
        "session metadata fallback": {"db_parity_expected": True, "recovery_critical": True},
        "session profile fallback": {"db_parity_expected": True, "recovery_critical": True},
        "session thread fallback": {"db_parity_expected": True, "recovery_critical": True},
        "personal memory structured listing fallback": {"db_parity_expected": True, "recovery_critical": False},
        "research note structured listing fallback": {"db_parity_expected": True, "recovery_critical": False},
        "research artifact structured listing fallback": {"db_parity_expected": True, "recovery_critical": False},
        "archive raw artifact hydration": {"db_parity_expected": False, "recovery_critical": True},
        "personal memory raw hydration": {"db_parity_expected": False, "recovery_critical": True},
        "research note raw hydration": {"db_parity_expected": False, "recovery_critical": True},
        "research artifact raw hydration": {"db_parity_expected": False, "recovery_critical": True},
    }

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.database_manager = DatabaseManager(settings)
        self.schema_manager = SchemaManager(self.database_manager)
        self.migration_runner = MigrationRunner(self.database_manager)
        self.projects = ProjectRepository(self.database_manager)
        self.sessions = SessionRepository(self.database_manager)
        self.messages = MessageRepository(self.database_manager)
        self.session_summaries = SessionSummaryRepository(self.database_manager)
        self.memory_items = MemoryRepository(self.database_manager)
        self.memory_embeddings = MemoryItemEmbeddingRepository(self.database_manager)
        self.tool_runs = ToolRunRepository(self.database_manager)
        self.bug_logs = BugLogRepository(self.database_manager)
        self.trainability_traces = TrainabilityTraceRepository(self.database_manager)
        self.dataset_import_runs = DatasetImportRunRepository(self.database_manager)
        self.dataset_examples = DatasetExampleRepository(self.database_manager)
        self.dataset_example_labels = DatasetExampleLabelRepository(self.database_manager)
        self.preferences = PreferenceRepository(self.database_manager)
        self.graph = GraphRepository(self.database_manager)
        self.project_resolver = ProjectResolver(
            project_repository=self.projects,
            session_repository=self.sessions,
        )
        self.semantic_embedding_service = SemanticEmbeddingService()
        self._initialized = False
        self._fallback_reads: dict[str, int] = {}

    def bootstrap(self, *, run_imports: bool = False) -> None:
        if not self._initialized:
            self.schema_manager.initialize()
            self._initialized = True
        if run_imports:
            self.run_legacy_imports()

    def record_fallback_read(self, surface: str, *, count: int = 1) -> None:
        normalized = " ".join(str(surface or "").strip().split())
        if not normalized:
            return
        self._fallback_reads[normalized] = self._fallback_reads.get(normalized, 0) + max(int(count), 1)

    def fallback_read_report(self) -> dict[str, object]:
        surfaces = dict(sorted(self._fallback_reads.items()))
        categories = {
            surface: self.STRUCTURED_READ_SURFACES.get(surface, "compat_fallback")
            for surface in surfaces
        }
        category_counts: dict[str, int] = {}
        for surface, count in surfaces.items():
            category = categories.get(surface, "compat_fallback")
            category_counts[category] = category_counts.get(category, 0) + int(count)
        retirement_readiness = self._retirement_readiness_report(surfaces=surfaces, categories=categories)
        return {
            "fallback_read_count": int(sum(surfaces.values())),
            "fallback_read_surfaces": list(surfaces.keys()),
            "fallback_read_breakdown": surfaces,
            "fallback_surface_categories": categories,
            "fallback_category_counts": category_counts,
            "persistence_surface_policies": dict(sorted(self.STRUCTURED_READ_SURFACES.items())),
            "fallback_retirement_readiness": retirement_readiness,
            "retire_candidate_surfaces": [
                surface
                for surface, readiness in retirement_readiness.items()
                if bool(readiness.get("retire_candidate"))
            ],
        }

    def _retirement_readiness_report(
        self,
        *,
        surfaces: dict[str, int],
        categories: dict[str, str],
    ) -> dict[str, dict[str, object]]:
        report: dict[str, dict[str, object]] = {}
        for surface, policy in sorted(self.STRUCTURED_SURFACE_READINESS.items()):
            category = categories.get(surface, self.STRUCTURED_READ_SURFACES.get(surface, "compat_fallback"))
            count = int(surfaces.get(surface, 0))
            db_parity_expected = bool(policy.get("db_parity_expected", False))
            recovery_critical = bool(policy.get("recovery_critical", False))
            retire_candidate = (
                category == "compat_fallback"
                and db_parity_expected
                and not recovery_critical
                and count == 0
            )
            report[surface] = {
                "fallback_count": count,
                "last_seen_category": category,
                "db_parity_expected": db_parity_expected,
                "recovery_critical": recovery_critical,
                "retire_candidate": retire_candidate,
            }
        return report

    def ensure_session(
        self,
        *,
        session_id: str,
        prompt: str | None,
        created_at: str | None,
        mode: str | None,
        title: str | None = None,
        active_topic: str | None = None,
        project_id: str | None = None,
        project_name: str | None = None,
        status: str = "active",
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.bootstrap()
        timestamp = created_at or datetime.now(UTC).isoformat()
        session_metadata = dict(metadata or {})
        preserve_existing_title = bool(session_metadata.pop("_preserve_title", False))
        resolved_project_id = self.project_resolver.resolve_project_id(
            session_id=session_id,
            prompt=prompt,
            title=title,
            active_topic=active_topic,
            project_id=project_id,
            project_name=project_name,
        )
        return self.sessions.upsert(
            session_id=session_id,
            project_id=resolved_project_id,
            title=None if preserve_existing_title and title is None else title or self._default_session_title(prompt),
            mode=mode,
            started_at=timestamp,
            updated_at=timestamp,
            status=status,
            metadata=session_metadata or None,
        )

    def record_interaction(
        self,
        *,
        session_id: str,
        prompt: str,
        response: dict[str, object],
        record: dict[str, object],
        interaction_path: str | None,
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        self.bootstrap()
        created_at = str(record.get("created_at") or datetime.now(UTC).isoformat())
        current_session = self.sessions.get(session_id)
        current_metadata = (
            dict(current_session.get("metadata_json") or {})
            if isinstance(current_session, dict)
            else {}
        )
        title_locked = bool(current_metadata.get("title_locked")) or str(current_metadata.get("title_source") or "") == "manual_user"
        session_title = None if title_locked else self._default_session_title(prompt)
        session_metadata = {
            **current_metadata,
            "title_locked": bool(title_locked),
            "title_source": str(current_metadata.get("title_source") or "manual_user")
            if title_locked
            else "auto_user_intent",
        }
        if title_locked:
            session_metadata["_preserve_title"] = True
        session = self.ensure_session(
            session_id=session_id,
            prompt=prompt,
            created_at=created_at,
            mode=str(record.get("mode") or response.get("mode") or "").strip() or None,
            title=session_title,
            active_topic=str(record.get("normalized_topic") or "").strip() or None,
            project_id=project_id,
            project_name=project_name,
            status="active",
            metadata=session_metadata,
        )
        turn_key = self._turn_key(interaction_path=interaction_path, created_at=created_at)
        route_decision = record.get("route") if isinstance(record.get("route"), dict) else None
        metadata = {
            "kind": record.get("kind"),
            "mode": record.get("mode"),
            "summary": record.get("summary"),
            "original_prompt": prompt,
            "resolved_prompt": record.get("resolved_prompt"),
            "resolution_strategy": record.get("resolution_strategy"),
            "resolution_reason": record.get("resolution_reason"),
            "interaction_path": interaction_path,
        }
        user_message_id = f"{session_id}:{turn_key}:user"
        assistant_message_id = f"{session_id}:{turn_key}:assistant"
        self.messages.upsert(
            message_id=user_message_id,
            session_id=session_id,
            turn_key=turn_key,
            role="user",
            content=prompt,
            created_at=created_at,
            intent_domain=str(record.get("intent_domain") or response.get("intent_domain") or "").strip() or None,
            confidence_tier=str(record.get("confidence_posture") or response.get("confidence_posture") or "").strip() or None,
            response_depth=str(record.get("response_depth") or response.get("response_depth") or "").strip() or None,
            conversation_phase=str(record.get("conversation_phase") or response.get("conversation_phase") or "").strip() or None,
            tool_usage_intent=str(record.get("tool_route_origin") or response.get("tool_route_origin") or "").strip() or None,
            route_decision=route_decision,
            metadata=metadata,
        )
        self.messages.upsert(
            message_id=assistant_message_id,
            session_id=session_id,
            turn_key=turn_key,
            role="assistant",
            content=self._assistant_message_content(record=record, response=response),
            created_at=self._assistant_message_created_at(created_at),
            intent_domain=str(record.get("intent_domain") or response.get("intent_domain") or "").strip() or None,
            confidence_tier=str(record.get("confidence_posture") or response.get("confidence_posture") or "").strip() or None,
            response_depth=str(record.get("response_depth") or response.get("response_depth") or "").strip() or None,
            conversation_phase=str(record.get("conversation_phase") or response.get("conversation_phase") or "").strip() or None,
            tool_usage_intent=str(record.get("tool_route_origin") or response.get("tool_route_origin") or "").strip() or None,
            route_decision=route_decision,
            metadata={**metadata, "interaction_record": dict(record)},
        )
        trace = record.get("trainability_trace") if isinstance(record.get("trainability_trace"), dict) else {}
        if trace:
            self.trainability_traces.upsert(
                trace_id=f"{session_id}:{turn_key}:trainability",
                session_id=session_id,
                message_id=assistant_message_id,
                decision_type=str(trace.get("primary_training_surface") or "interaction_decision"),
                input_context_summary=str(record.get("summary") or prompt).strip() or None,
                chosen_action=str(trace.get("selected_action") or record.get("kind") or "").strip() or None,
                outcome=str(trace.get("outcome") or record.get("mode") or "").strip() or None,
                label=str(trace.get("label") or "").strip() or None,
                confidence_tier=str(record.get("confidence_posture") or "").strip() or None,
                created_at=created_at,
                model_assist_used=bool(record.get("supervised_support_trace")),
                metadata={
                    "available_training_surfaces": list(trace.get("available_training_surfaces") or []),
                    "deterministic_surfaces": list(trace.get("deterministic_surfaces") or []),
                    "interaction_path": interaction_path,
                },
            )
        return session

    def record_session_state(
        self,
        *,
        session_id: str,
        payload: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        self.bootstrap()
        updated_at = str(payload.get("updated_at") or datetime.now(UTC).isoformat())
        current = self.sessions.get(session_id)
        current_title = str((current or {}).get("title") or "").strip()
        current_metadata = dict((current or {}).get("metadata_json") or {}) if isinstance(current, dict) else {}
        title_locked = bool(current_metadata.get("title_locked")) or str(current_metadata.get("title_source") or "") == "manual_user"
        prompt_title = self._default_session_title(str(payload.get("prompt") or "").strip())
        session_metadata = {**current_metadata, **dict(payload)}
        if title_locked:
            session_metadata["title_locked"] = True
            session_metadata["title_source"] = "manual_user"
            session_metadata["_preserve_title"] = True
            title_for_upsert = current_title or None
        else:
            session_metadata["title_locked"] = False
            session_metadata["title_source"] = "auto_user_intent"
            title_for_upsert = prompt_title or current_title or None
        session = self.ensure_session(
            session_id=session_id,
            prompt=str(payload.get("prompt") or "").strip() or None,
            created_at=updated_at,
            mode=str(payload.get("mode") or "").strip() or None,
            title=title_for_upsert,
            active_topic=str(payload.get("normalized_topic") or "").strip() or None,
            project_id=project_id,
            project_name=project_name,
            status="active",
            metadata=session_metadata,
        )
        summary_text = str(payload.get("thread_summary") or payload.get("summary") or payload.get("objective") or "").strip()
        if summary_text:
            summary_id = f"{session_id}:summary:{self._turn_key(interaction_path=None, created_at=updated_at)}"
            summary = self.session_summaries.upsert(
                summary_id=summary_id,
                session_id=session_id,
                summary_text=summary_text,
                created_at=updated_at,
                confidence_tier=str(payload.get("confidence_posture") or "").strip() or None,
                tags=[],
                summary_scope="session_state_snapshot",
                metadata={"objective": payload.get("objective"), "prompt": payload.get("prompt")},
            )
            self.sessions.update_summary(session_id=session_id, summary_id=str(summary["id"]), updated_at=updated_at)
        return session

    def record_tool_run(
        self,
        *,
        session_id: str,
        archive_record: dict[str, object],
        message_id: str | None = None,
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        self.bootstrap()
        created_at = str(archive_record.get("created_at") or datetime.now(UTC).isoformat())
        session = self.ensure_session(
            session_id=session_id,
            prompt=str(archive_record.get("summary") or "").strip() or None,
            created_at=created_at,
            mode="tool",
            title=str(archive_record.get("summary") or "").strip() or None,
            active_topic=str(archive_record.get("capability") or "").strip() or None,
            project_id=project_id,
            project_name=project_name,
            status="active",
        )
        tool_run_id = self._stable_id("tool-run", str(archive_record.get("archive_path") or created_at))
        return self.tool_runs.upsert(
            tool_run_id=tool_run_id,
            session_id=session_id,
            message_id=message_id,
            project_id=str(session.get("project_id")) if session.get("project_id") else None,
            tool_name=str(archive_record.get("tool_id") or "unknown"),
            capability=str(archive_record.get("capability") or "").strip() or None,
            input_summary=str(archive_record.get("input_path") or "").strip() or None,
            output_summary=str(archive_record.get("summary") or "").strip() or None,
            success=str(archive_record.get("status") or "").strip().lower() == "ok",
            created_at=created_at,
            tool_bundle=str(archive_record.get("tool_id") or "").strip() or None,
            archive_path=str(archive_record.get("archive_path") or "").strip() or None,
            run_dir=str(archive_record.get("run_dir") or "").strip() or None,
            metadata={
                "archive_record": dict(archive_record),
                "status": archive_record.get("status"),
                "result_quality": archive_record.get("result_quality"),
                "target_label": archive_record.get("target_label"),
            },
        )

    def record_memory_item(
        self,
        *,
        source_type: str,
        payload: dict[str, object],
        project_id: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, object]:
        self.bootstrap()
        session_id = str(payload.get("session_id") or "").strip() or None
        created_at = str(payload.get("created_at") or datetime.now(UTC).isoformat())
        source_path = str(
            payload.get("entry_path")
            or payload.get("note_path")
            or payload.get("artifact_path")
            or payload.get("source_interaction_path")
            or ""
        ).strip()
        session = None
        if session_id:
            session = self.ensure_session(
                session_id=session_id,
                prompt=str(payload.get("source_interaction_prompt") or payload.get("title") or "").strip() or None,
                created_at=created_at,
                mode=str(payload.get("source_interaction_mode") or "").strip() or None,
                title=str(payload.get("title") or "").strip() or None,
                active_topic=str(payload.get("normalized_topic") or "").strip() or None,
                project_id=project_id,
                project_name=project_name,
                status="active",
            )
        resolved_project_id = (
            str(session.get("project_id")) if session and session.get("project_id") else None
        ) or self.project_resolver.resolve_project_id(
            session_id=session_id or "memory",
            prompt=str(payload.get("source_interaction_prompt") or payload.get("title") or "").strip() or None,
            title=str(payload.get("title") or "").strip() or None,
            active_topic=str(payload.get("normalized_topic") or "").strip() or None,
            project_id=project_id,
            project_name=project_name,
        )
        category = {
            "personal_memory": "personal_memory",
            "research_note": "research_note",
            "research_artifact": "research_artifact",
        }.get(source_type, source_type)
        domain = str(payload.get("normalized_topic") or payload.get("dominant_intent") or "").strip() or None
        content = str(payload.get("content") or payload.get("title") or "").strip() or ""
        source_id = source_path or self._stable_id(source_type, f"{session_id}:{created_at}:{content}")
        memory_id = self._stable_id("memory", f"{source_type}:{source_id}")
        row = self.memory_items.upsert(
            memory_id=memory_id,
            source_type=source_type,
            source_id=source_id,
            project_id=resolved_project_id,
            session_id=session_id,
            category=category,
            domain=domain,
            content=content,
            confidence_tier=self._memory_confidence_tier(payload),
            created_at=created_at,
            updated_at=created_at,
            recency_weight=1.0,
            relevance_hint=str(payload.get("title") or "").strip() or None,
            status="archived" if bool(payload.get("archived", False)) else "active",
            source_summary=str(payload.get("source_interaction_summary") or payload.get("title") or "").strip() or None,
            metadata=dict(payload),
        )
        self.ensure_memory_item_embedding(row)
        return row

    def ensure_memory_item_embedding(
        self,
        memory_row: dict[str, object] | None,
        *,
        try_generate: bool = True,
    ) -> dict[str, object] | None:
        self.bootstrap()
        if not isinstance(memory_row, dict):
            return None
        memory_item_id = str(memory_row.get("id") or "").strip()
        if not memory_item_id:
            return None
        content = str(memory_row.get("content") or "")
        content_hash = self.semantic_embedding_service.content_hash(content)
        model_name = self.semantic_embedding_service.model_name
        timestamp = datetime.now(UTC).isoformat()
        existing = self.memory_embeddings.get(memory_item_id)
        if (
            isinstance(existing, dict)
            and str(existing.get("content_hash") or "") == content_hash
            and str(existing.get("model_name") or "") == model_name
            and str(existing.get("status") or "") == "ready"
            and existing.get("embedding_blob") is not None
        ):
            return existing
        pending = self.memory_embeddings.upsert(
            memory_item_id=memory_item_id,
            source_id=str(memory_row.get("source_id") or "").strip() or memory_item_id,
            source_type=str(memory_row.get("source_type") or "memory_item").strip() or "memory_item",
            model_name=model_name,
            embedding_dim=int(existing.get("embedding_dim") or 0) if isinstance(existing, dict) and existing.get("embedding_dim") else None,
            embedding_blob=None,
            content_hash=content_hash,
            status="pending",
            created_at=str(existing.get("created_at") or timestamp) if isinstance(existing, dict) else timestamp,
            updated_at=timestamp,
            error_message=None,
        )
        if not try_generate:
            return pending
        if not self.semantic_embedding_service.is_available():
            return pending
        try:
            vector = self.semantic_embedding_service.embed_text(content)
        except RuntimeError:
            return pending
        except Exception as exc:
            return self.memory_embeddings.upsert(
                memory_item_id=memory_item_id,
                source_id=str(memory_row.get("source_id") or "").strip() or memory_item_id,
                source_type=str(memory_row.get("source_type") or "memory_item").strip() or "memory_item",
                model_name=model_name,
                embedding_dim=None,
                embedding_blob=None,
                content_hash=content_hash,
                status="failed",
                created_at=str(pending.get("created_at") or timestamp),
                updated_at=timestamp,
                error_message=str(exc)[:240],
            )
        return self.memory_embeddings.upsert(
            memory_item_id=memory_item_id,
            source_id=str(memory_row.get("source_id") or "").strip() or memory_item_id,
            source_type=str(memory_row.get("source_type") or "memory_item").strip() or "memory_item",
            model_name=model_name,
            embedding_dim=len(vector),
            embedding_blob=self.semantic_embedding_service.pack_embedding(vector),
            content_hash=content_hash,
            status="ready",
            created_at=str(pending.get("created_at") or timestamp),
            updated_at=timestamp,
            error_message=None,
        )

    def backfill_memory_item_embeddings(self, *, limit: int | None = None) -> dict[str, object]:
        self.bootstrap()
        content_hashes = self._memory_item_content_hashes()
        candidates: list[dict[str, object]] = []
        candidates.extend(self.memory_embeddings.list_missing(limit=limit))
        stale_ids = {str(item.get("memory_item_id") or "") for item in self.memory_embeddings.list_stale(content_hashes=content_hashes)}
        if stale_ids:
            for memory_item_id in stale_ids:
                row = self.memory_items.get(memory_item_id)
                if isinstance(row, dict):
                    candidates.append(row)
        seen: set[str] = set()
        processed = 0
        ready = 0
        failed = 0
        pending = 0
        skipped = 0
        max_items = max(int(limit), 1) if limit is not None else None
        for row in candidates:
            memory_item_id = str(row.get("id") or row.get("memory_item_id") or "").strip()
            if not memory_item_id or memory_item_id in seen:
                continue
            seen.add(memory_item_id)
            if max_items is not None and processed >= max_items:
                break
            result = self.ensure_memory_item_embedding(row, try_generate=True)
            processed += 1
            status = str((result or {}).get("status") or "")
            if status == "ready":
                ready += 1
            elif status == "failed":
                failed += 1
            elif status == "pending":
                pending += 1
            else:
                skipped += 1
        return {
            "db_path": str(self.settings.persistence_db_path),
            "model_name": self.semantic_embedding_service.model_name,
            "runtime_available": self.semantic_embedding_service.is_available(),
            "processed": processed,
            "ready": ready,
            "failed": failed,
            "pending": pending,
            "skipped": skipped,
        }

    def semantic_status_report(self) -> dict[str, object]:
        self.bootstrap()
        with self.database_manager.connect() as conn:
            total_memory_items = int(conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0])
            total_embeddings = int(conn.execute("SELECT COUNT(*) FROM memory_item_embeddings").fetchone()[0])
            ready_embeddings = int(
                conn.execute("SELECT COUNT(*) FROM memory_item_embeddings WHERE status = 'ready'").fetchone()[0]
            )
            pending_embeddings = int(
                conn.execute("SELECT COUNT(*) FROM memory_item_embeddings WHERE status = 'pending'").fetchone()[0]
            )
            failed_embeddings = int(
                conn.execute("SELECT COUNT(*) FROM memory_item_embeddings WHERE status = 'failed'").fetchone()[0]
            )
        stale_embeddings = len(self.memory_embeddings.list_stale(content_hashes=self._memory_item_content_hashes()))
        availability = self.semantic_embedding_service.availability_status()
        return {
            "db_path": str(self.settings.persistence_db_path),
            "semantic_model_name": self.semantic_embedding_service.model_name,
            "semantic_runtime_available": bool(availability.get("available")),
            "semantic_runtime_error": availability.get("error"),
            "total_memory_items": total_memory_items,
            "embedded_memory_items": total_embeddings,
            "ready_embeddings": ready_embeddings,
            "pending_embeddings": pending_embeddings,
            "failed_embeddings": failed_embeddings,
            "stale_embeddings": stale_embeddings,
        }

    def delete_session(self, session_id: str) -> None:
        self.bootstrap()
        self.sessions.delete(session_id)

    def update_interaction_memory_links(
        self,
        *,
        session_id: str,
        interaction_path: str,
        research_note: dict[str, object] | None = None,
        personal_memory: dict[str, object] | None = None,
    ) -> None:
        self.bootstrap()
        turn_key = self._turn_key(interaction_path=interaction_path, created_at="")
        message_id = f"{session_id}:{turn_key}:assistant"
        current = self.messages.get(message_id)
        if current is None:
            return
        metadata = (
            dict(current.get("message_metadata_json") or {})
            if isinstance(current.get("message_metadata_json"), dict)
            else {}
        )
        interaction_record = (
            dict(metadata.get("interaction_record") or {})
            if isinstance(metadata.get("interaction_record"), dict)
            else {}
        )
        if research_note is not None:
            interaction_record["research_note"] = dict(research_note)
        if personal_memory is not None:
            interaction_record["personal_memory"] = dict(personal_memory)
        metadata["interaction_record"] = interaction_record
        self.messages.upsert(
            message_id=message_id,
            session_id=str(current.get("session_id") or session_id),
            turn_key=str(current.get("turn_key") or turn_key),
            role=str(current.get("role") or "assistant"),
            content=str(current.get("content") or ""),
            created_at=str(current.get("created_at") or datetime.now(UTC).isoformat()),
            intent_domain=str(current.get("intent_domain") or "").strip() or None,
            confidence_tier=str(current.get("confidence_tier") or "").strip() or None,
            response_depth=str(current.get("response_depth") or "").strip() or None,
            conversation_phase=str(current.get("conversation_phase") or "").strip() or None,
            tool_usage_intent=str(current.get("tool_usage_intent") or "").strip() or None,
            route_decision=(
                dict(current.get("route_decision_json") or {})
                if isinstance(current.get("route_decision_json"), dict)
                else None
            ),
            metadata=metadata,
        )

    def status_report(self) -> dict[str, object]:
        self.bootstrap()
        table_names = (
            "projects",
            "sessions",
            "messages",
            "session_summaries",
            "memory_items",
            "tool_runs",
            "bug_logs",
            "trainability_traces",
            "preferences",
            "knowledge_entries",
            "knowledge_aliases",
            "knowledge_relationships",
            "knowledge_formulas",
            "memory_item_embeddings",
            "nodes",
            "relations",
            "observations",
        )
        table_counts: dict[str, int] = {}
        with self.database_manager.connect() as conn:
            for table_name in table_names:
                table_counts[table_name] = int(
                    conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                )
        return {
            "repo_root": str(self.settings.repo_root),
            "db_path": str(self.settings.persistence_db_path),
            "migrations": self.migration_runner.list_migrations(),
            "import_runs": self.migration_runner.list_import_runs(),
            "table_counts": table_counts,
            "semantic_status": self.semantic_status_report(),
            **self.fallback_read_report(),
        }

    def coverage_report(self) -> dict[str, object]:
        self.bootstrap()
        sessions_legacy = self._legacy_session_count()
        interaction_files = self._legacy_json_count(self.settings.interactions_root, skip_index=True)
        tool_run_files = self._legacy_json_count(self.settings.archive_root, skip_index=True)
        personal_files = self._legacy_json_count(self.settings.personal_memory_root)
        note_files = self._legacy_json_count(self.settings.research_notes_root)
        artifact_files = self._legacy_json_count(self.settings.research_artifacts_root)
        with self.database_manager.connect() as conn:
            db_counts = {
                "sessions": int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]),
                "messages": int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
                "tool_runs": int(conn.execute("SELECT COUNT(*) FROM tool_runs").fetchone()[0]),
                "memory_items": int(conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]),
                "trainability_traces": int(conn.execute("SELECT COUNT(*) FROM trainability_traces").fetchone()[0]),
                "knowledge_entries": int(conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]),
                "memory_item_embeddings": int(conn.execute("SELECT COUNT(*) FROM memory_item_embeddings").fetchone()[0]),
            }
        legacy_counts = {
            "sessions": sessions_legacy,
            "interaction_files": interaction_files,
            "archive_files": tool_run_files,
            "personal_memory_files": personal_files,
            "research_note_files": note_files,
            "research_artifact_files": artifact_files,
            "knowledge_db_present": int(self.settings.knowledge_db_path.exists()),
        }
        return {
            "repo_root": str(self.settings.repo_root),
            "db_path": str(self.settings.persistence_db_path),
            "db_counts": db_counts,
            "legacy_counts": legacy_counts,
            "semantic_status": self.semantic_status_report(),
            **self.fallback_read_report(),
        }

    def doctor_report(self) -> dict[str, object]:
        self.bootstrap()
        missing_sessions = 0
        for session_dir in self.settings.sessions_root.iterdir() if self.settings.sessions_root.exists() else []:
            if not session_dir.is_dir():
                continue
            if (session_dir / "thread_state.json").exists() and self.sessions.get(session_dir.name) is None:
                missing_sessions += 1

        missing_interactions = 0
        for path in self._iter_legacy_records(self.settings.interactions_root, skip_index=True):
            session_id = path.parent.name
            message_id = f"{session_id}:{path.stem}:assistant"
            if self.messages.get(message_id) is None:
                missing_interactions += 1

        missing_tool_runs = 0
        for path in self._iter_legacy_records(self.settings.archive_root, skip_index=True):
            tool_run_id = self._stable_id("tool-run", str(path))
            if self.tool_runs.get(tool_run_id) is None:
                missing_tool_runs += 1

        missing_personal_memory = 0
        for path in self._iter_legacy_records(self.settings.personal_memory_root):
            memory_id = self._stable_id("memory", f"personal_memory:{path}")
            if self.memory_items.get(memory_id) is None:
                missing_personal_memory += 1

        missing_research_notes = 0
        for path in self._iter_legacy_records(self.settings.research_notes_root):
            memory_id = self._stable_id("memory", f"research_note:{path}")
            if self.memory_items.get(memory_id) is None:
                missing_research_notes += 1

        missing_research_artifacts = 0
        for path in self._iter_legacy_records(self.settings.research_artifacts_root):
            memory_id = self._stable_id("memory", f"research_artifact:{path}")
            if self.memory_items.get(memory_id) is None:
                missing_research_artifacts += 1

        orphan_counts = self._orphan_counts()
        return {
            "repo_root": str(self.settings.repo_root),
            "db_path": str(self.settings.persistence_db_path),
            "missing_rows": {
                "sessions": missing_sessions,
                "interaction_records": missing_interactions,
                "tool_runs": missing_tool_runs,
                "personal_memory": missing_personal_memory,
                "research_notes": missing_research_notes,
                "research_artifacts": missing_research_artifacts,
            },
            "orphan_counts": orphan_counts,
            "duplicate_import_suspicions": {
                "sessions": 0,
                "interaction_records": 0,
                "tool_runs": 0,
                "knowledge_entries": 0,
            },
            "semantic_status": self.semantic_status_report(),
            **self.fallback_read_report(),
        }

    def run_legacy_imports(self) -> None:
        self.bootstrap()
        self._import_legacy_sessions()
        self._import_legacy_interactions()
        self._import_legacy_memory()
        self._import_legacy_tool_runs()
        self._import_legacy_knowledge()
        self.graph.import_legacy_graph(self.settings.graph_memory_db_path)
        self.migration_runner.record_import_run("0003_legacy_import", {"source_data_root": str(self.settings.data_root)})

    def _import_legacy_sessions(self) -> None:
        sessions_root = self.settings.sessions_root
        if not sessions_root.exists():
            return
        for session_dir in sessions_root.iterdir():
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            thread_path = session_dir / "thread_state.json"
            if not thread_path.exists():
                continue
            try:
                payload = SessionThreadSchema.normalize(json.loads(thread_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            self.record_session_state(session_id=session_id, payload=payload)

    def _import_legacy_interactions(self) -> None:
        root = self.settings.interactions_root
        if not root.exists():
            return
        for interaction_path in sorted(root.rglob("*.json")):
            if interaction_path.name == "_index.json":
                continue
            try:
                payload = InteractionRecordSchema.normalize(json.loads(interaction_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            self.record_interaction(
                session_id=str(payload.get("session_id") or interaction_path.parent.name),
                prompt=str(payload.get("prompt") or ""),
                response=dict(payload.get("response") or {}),
                record=payload,
                interaction_path=str(interaction_path),
            )

    def _import_legacy_memory(self) -> None:
        roots_and_schemas = [
            (self.settings.personal_memory_root, PersonalMemorySchema, "personal_memory", "entry_path"),
            (self.settings.research_notes_root, ResearchNoteSchema, "research_note", "note_path"),
            (self.settings.research_artifacts_root, ResearchArtifactSchema, "research_artifact", "artifact_path"),
        ]
        for root, schema, source_type, path_key in roots_and_schemas:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.json")):
                try:
                    payload = schema.normalize(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                payload[path_key] = str(path)
                self.record_memory_item(source_type=source_type, payload=payload)

    def _import_legacy_tool_runs(self) -> None:
        root = self.settings.archive_root
        if not root.exists():
            return
        for path in sorted(root.rglob("*.json")):
            if path.name == "_index.json":
                continue
            try:
                payload = ArchiveRecordSchema.normalize(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            payload["archive_path"] = str(path)
            self.record_tool_run(
                session_id=str(payload.get("session_id") or path.parent.parent.parent.name),
                archive_record=payload,
            )

    def _import_legacy_knowledge(self) -> None:
        knowledge_db = KnowledgeDB(self.settings.persistence_db_path)
        imported = knowledge_db.import_legacy_database(self.settings.knowledge_db_path)
        self.migration_runner.record_import_run(
            "0004_legacy_knowledge_import",
            {
                "legacy_path": str(self.settings.knowledge_db_path),
                "db_path": str(self.settings.persistence_db_path),
                "imported": imported,
            },
        )

    @staticmethod
    def _stable_id(prefix: str, value: str) -> str:
        sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
        sanitized = "_".join(part for part in sanitized.split("_") if part)
        return f"{prefix}:{sanitized[:180] or 'item'}"

    @staticmethod
    def _turn_key(*, interaction_path: str | None, created_at: str) -> str:
        if interaction_path:
            return Path(interaction_path).stem
        sanitized = "".join(char if char.isalnum() else "_" for char in created_at)
        return "_".join(part for part in sanitized.split("_") if part)

    @staticmethod
    def _default_session_title(prompt: str | None) -> str | None:
        text = " ".join(str(prompt or "").split()).strip()
        if not text:
            return None
        return text[:120]

    @staticmethod
    def _assistant_message_created_at(created_at: str) -> str:
        """Keep assistant rows just after their paired user row for raw DB readers."""
        text = str(created_at or "").strip()
        if not text:
            return text
        try:
            normalized = text.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return text
        return (parsed + timedelta(microseconds=1)).isoformat()

    @staticmethod
    def _assistant_message_content(*, record: dict[str, object], response: dict[str, object]) -> str:
        candidates = [
            record.get("user_facing_answer"),
            response.get("user_facing_answer"),
            record.get("reply"),
            response.get("reply"),
            record.get("answer"),
            response.get("answer"),
            record.get("result_text"),
            response.get("result_text"),
            record.get("result_summary"),
            response.get("result_summary"),
            record.get("summary"),
            response.get("summary"),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text
        return str(record.get("kind") or response.get("kind") or "assistant_response")

    @staticmethod
    def _memory_confidence_tier(payload: dict[str, object]) -> str | None:
        classification = payload.get("memory_classification")
        if isinstance(classification, dict):
            score = float(classification.get("classification_confidence") or 0.0)
            if score >= 0.8:
                return "high"
            if score >= 0.5:
                return "medium"
            return "low"
        return None

    def _legacy_session_count(self) -> int:
        if not self.settings.sessions_root.exists():
            return 0
        count = 0
        for session_dir in self.settings.sessions_root.iterdir():
            if session_dir.is_dir() and (session_dir / "thread_state.json").exists():
                count += 1
        return count

    @staticmethod
    def _iter_legacy_records(root: Path, *, skip_index: bool = False):
        if not root.exists():
            return []
        return [
            path
            for path in sorted(root.rglob("*.json"))
            if not (skip_index and path.name == "_index.json")
        ]

    def _legacy_json_count(self, root: Path, *, skip_index: bool = False) -> int:
        return len(self._iter_legacy_records(root, skip_index=skip_index))

    def _orphan_counts(self) -> dict[str, int]:
        with self.database_manager.connect() as conn:
            assistant_without_session = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM messages m
                    LEFT JOIN sessions s ON s.id = m.session_id
                    WHERE s.id IS NULL
                    """
                ).fetchone()[0]
            )
            tool_without_session = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_runs t
                    LEFT JOIN sessions s ON s.id = t.session_id
                    WHERE s.id IS NULL
                    """
                ).fetchone()[0]
            )
            memory_without_session = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM memory_items m
                    LEFT JOIN sessions s ON s.id = m.session_id
                    WHERE m.session_id IS NOT NULL AND s.id IS NULL
                    """
                ).fetchone()[0]
            )
        return {
            "messages_without_session": assistant_without_session,
            "tool_runs_without_session": tool_without_session,
            "memory_without_session": memory_without_session,
        }

    def _memory_item_content_hashes(self) -> dict[str, str]:
        rows = self.memory_items.list_by_filters(include_archived=True)
        return {
            str(row.get("id") or ""): self.semantic_embedding_service.content_hash(str(row.get("content") or ""))
            for row in rows
            if str(row.get("id") or "").strip()
        }
