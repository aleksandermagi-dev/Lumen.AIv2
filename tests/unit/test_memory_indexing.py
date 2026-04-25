from datetime import UTC, datetime, timedelta

from lumen.memory.indexing import MemoryIndexBuilder
from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalLayer
from lumen.semantic.embedding_service import SemanticEmbeddingService


class _NullInteractionHistoryService:
    def retrieve_context(
        self,
        prompt: str,
        *,
        session_id: str,
        project_id: str | None = None,
        limit: int = 3,
    ) -> dict[str, object]:
        return {"top_interaction_matches": []}


class _NullArchiveService:
    def retrieve_context(
        self,
        prompt: str,
        *,
        session_id: str,
        project_id: str | None = None,
        limit: int = 3,
    ) -> dict[str, object]:
        return {"top_matches": []}


class _NullGraphMemoryManager:
    def search_nodes(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        return []


class _ProjectAwareInteractionHistoryService:
    def retrieve_context(
        self,
        prompt: str,
        *,
        session_id: str,
        project_id: str | None = None,
        limit: int = 3,
    ) -> dict[str, object]:
        matches = [
            {
                "score": 5,
                "record": {
                    "session_id": "other",
                    "project_id": "other",
                    "summary": "Routing migration notes from an unrelated project.",
                    "normalized_topic": "routing migration",
                    "created_at": "2026-04-08T10:00:00Z",
                    "mode": "planning",
                    "kind": "planning.migration",
                },
            },
            {
                "score": 5,
                "record": {
                    "session_id": session_id,
                    "project_id": project_id,
                    "summary": "Routing migration notes for the active lumen project.",
                    "normalized_topic": "routing migration",
                    "created_at": "2026-04-08T11:00:00Z",
                    "mode": "planning",
                    "kind": "planning.migration",
                },
            },
        ]
        return {"top_interaction_matches": matches[:limit]}


class _StubPersonalMemoryManager:
    def __init__(self, entries: list[dict[str, object]]) -> None:
        self._entries = entries

    def list_entries(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> list[dict[str, object]]:
        return list(self._entries)


class _StubResearchNoteManager:
    def __init__(self, notes: list[dict[str, object]]) -> None:
        self._notes = notes

    def list_notes(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
    ) -> list[dict[str, object]]:
        return list(self._notes)


class _FakeEmbeddingRepository:
    def __init__(self, rows: dict[tuple[str, str], dict[str, object]]) -> None:
        self._rows = rows

    def get_by_source(self, *, source_type: str, source_id: str) -> dict[str, object] | None:
        return self._rows.get((source_type, source_id))


class _FakeSemanticEmbeddingService:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def is_available(self) -> bool:
        return True

    def embed_text(self, text: str | None) -> list[float]:
        normalized = " ".join(str(text or "").strip().lower().split())
        if "schema" in normalized and "migration" in normalized:
            return [1.0, 0.0]
        if "galaxy" in normalized and "spectral" in normalized:
            return [0.0, 1.0]
        return [0.4, 0.4]

    @staticmethod
    def unpack_embedding(blob):
        return SemanticEmbeddingService.unpack_embedding(blob)

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        return SemanticEmbeddingService.cosine_similarity(left, right)


class _FakePersistenceManager:
    def __init__(
        self,
        rows: dict[tuple[str, str], dict[str, object]],
        *,
        summaries: list[dict[str, object]] | None = None,
        messages: list[dict[str, object]] | None = None,
    ) -> None:
        self.memory_embeddings = _FakeEmbeddingRepository(rows)
        self.semantic_embedding_service = _FakeSemanticEmbeddingService()
        self.session_summaries = _FakeSessionSummaryRepository(summaries or [])
        self.messages = _FakeMessageRepository(messages or [])


class _FakeSessionSummaryRepository:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    def list_recent_by_session(self, session_id: str, *, limit: int = 5) -> list[dict[str, object]]:
        return [row for row in self._rows if row.get("session_id") == session_id][:limit]

    def list_recent_by_project(self, project_id: str, *, limit: int = 5) -> list[dict[str, object]]:
        return [
            row
            for row in self._rows
            if str((row.get("metadata_json") or {}).get("project_id") or "").strip() == project_id
        ][:limit]


class _FakeMessageRepository:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    def list_message_window_by_session(self, session_id: str, *, limit: int = 6) -> list[dict[str, object]]:
        return [row for row in self._rows if row.get("session_id") == session_id][:limit]

    def list_message_window_by_project(self, project_id: str, *, limit: int = 6) -> list[dict[str, object]]:
        return [
            row
            for row in self._rows
            if str((row.get("message_metadata_json") or {}).get("project_id") or "").strip() == project_id
        ][:limit]


def test_memory_index_builder_normalizes_personal_memory_entry() -> None:
    indexed = MemoryIndexBuilder.from_personal_entry(
        {
            "title": "verbosity preference",
            "content": "The user prefers concise replies.",
            "normalized_topic": "preferences",
            "created_at": "2026-04-08T10:00:00Z",
            "entry_path": "data/personal/default/entry.json",
            "memory_classification": {"candidate_type": "personal_context_candidate"},
            "client_surface": "main",
        }
    )

    assert indexed is not None
    assert indexed.storage_layer == "indexed_layer"
    assert indexed.source_category == "personal_memory"
    assert indexed.memory_kind == "durable_user_memory"
    assert "preferences" in indexed.domain_tags


def test_memory_index_builder_normalizes_research_note() -> None:
    indexed = MemoryIndexBuilder.from_research_note(
        {
            "title": "routing migration",
            "content": "We split the routing cleanup into controller and service work.",
            "normalized_topic": "routing",
            "dominant_intent": "planning",
            "note_type": "chronological_research_note",
            "created_at": "2026-04-08T11:00:00Z",
            "note_path": "data/research/default/note.json",
        }
    )

    assert indexed is not None
    assert indexed.storage_layer == "indexed_layer"
    assert indexed.source_category == "research_notes"
    assert indexed.memory_kind == "durable_project_memory"
    assert "routing" in indexed.domain_tags


def test_memory_retrieval_layer_reads_personal_memory_store_directly() -> None:
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=_StubPersonalMemoryManager(
            [
                {
                    "title": "verbosity preference",
                    "content": "The user prefers concise replies.",
                    "normalized_topic": "preferences",
                    "created_at": "2026-04-08T10:00:00Z",
                }
            ]
        ),
        research_note_manager=None,
    )

    result = layer.retrieve(
        prompt="what do you remember about my preferences?",
        session_id="default",
        active_thread=None,
        route_mode="conversation",
    )

    assert result.selected
    lead = result.selected[0]
    assert lead.source == "personal_memory"
    assert lead.metadata["storage_layer"] == "indexed_layer"
    assert lead.metadata["source_category"] == "personal_memory"


def test_memory_retrieval_layer_reads_research_notes_store_directly() -> None:
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=_StubResearchNoteManager(
            [
                {
                    "title": "routing migration",
                    "content": "We split the routing cleanup into controller and service work.",
                    "normalized_topic": "routing",
                    "dominant_intent": "planning",
                    "note_type": "chronological_research_note",
                    "created_at": "2026-04-08T11:00:00Z",
                }
            ]
        ),
    )

    result = layer.retrieve(
        prompt="where were we on routing?",
        session_id="default",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    assert any(item.source == "research_notes" for item in result.selected)
    research_note = next(item for item in result.selected if item.source == "research_notes")
    assert research_note.metadata["storage_layer"] == "indexed_layer"
    assert research_note.metadata["source_category"] == "research_notes"


def test_memory_retrieval_layer_prefers_recent_reaffirmed_project_memory_over_stale_generic_note() -> None:
    now = datetime.now(UTC)
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=_StubResearchNoteManager(
            [
                {
                    "title": "routing migration summary",
                    "content": "We split routing migration into controller and service work.",
                    "normalized_topic": "routing migration",
                    "dominant_intent": "planning",
                    "note_type": "chronological_research_note",
                    "created_at": (now - timedelta(days=3)).isoformat().replace("+00:00", "Z"),
                },
                {
                    "title": "first pass summary",
                    "content": "Older routing notes from the first pass with generic cleanup ideas.",
                    "normalized_topic": "routing",
                    "dominant_intent": "planning",
                    "note_type": "chronological_research_note",
                    "created_at": (now - timedelta(days=190)).isoformat().replace("+00:00", "Z"),
                },
            ]
        ),
    )

    result = layer.retrieve(
        prompt="where were we on routing migration?",
        session_id="default",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    lead = result.selected[0]
    assert lead.source == "research_notes"
    assert lead.label == "routing migration summary"
    assert lead.metadata["age_bucket"] == "recent"
    assert lead.metadata["reaffirmed"] is True


def test_memory_retrieval_layer_penalizes_conflicting_project_memory_for_normal_recall() -> None:
    now = datetime.now(UTC)
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=_StubResearchNoteManager(
            [
                {
                    "title": "routing migration plan",
                    "content": "The routing migration keeps controller cleanup ahead of service extraction.",
                    "normalized_topic": "routing migration",
                    "dominant_intent": "planning",
                    "note_type": "chronological_research_note",
                    "created_at": (now - timedelta(days=5)).isoformat().replace("+00:00", "Z"),
                },
                {
                    "title": "routing contradiction note",
                    "content": "Potential contradiction: service extraction should come first instead.",
                    "normalized_topic": "routing migration",
                    "dominant_intent": "planning",
                    "note_type": "chronological_research_note",
                    "created_at": (now - timedelta(days=5)).isoformat().replace("+00:00", "Z"),
                },
            ]
        ),
    )

    result = layer.retrieve(
        prompt="what do you remember about the routing migration?",
        session_id="default",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    assert result.selected[0].label == "routing migration plan"
    assert all(item.label != "routing contradiction note" for item in result.selected)


def test_memory_retrieval_layer_prefers_same_project_history_when_project_is_known() -> None:
    layer = MemoryRetrievalLayer(
        interaction_history_service=_ProjectAwareInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=None,
    )

    result = layer.retrieve(
        prompt="what do you remember about routing migration?",
        session_id="default",
        project_id="lumen",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    lead = result.selected[0]
    assert lead.source == "interaction_history"
    assert lead.metadata["project_match"] is True
    assert "active lumen project" in lead.summary.lower()


def test_memory_retrieval_layer_reports_continuity_window_and_selection_reasons() -> None:
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=None,
    )

    result = layer.retrieve(
        prompt="what do you remember about routing migration?",
        session_id="default",
        project_id="lumen",
        active_thread=None,
        route_mode="planning",
        recent_interactions=[
            {
                "session_id": "default",
                "project_id": "lumen",
                "prompt": "create a routing migration plan",
                "summary": "Routing migration stays split into controller then service cleanup.",
                "normalized_topic": "routing migration",
                "created_at": "2026-04-08T11:00:00Z",
            },
            {
                "session_id": "other",
                "project_id": "other",
                "prompt": "summarize another project",
                "summary": "Different project context.",
                "normalized_topic": "other project",
                "created_at": "2026-04-08T10:00:00Z",
            },
        ],
    )

    assert result.selected
    assert result.selected[0].source == "recent_interactions"
    assert result.diagnostics["continuity_window_used"] == 2
    assert result.diagnostics["selected_reasons"][0]["reason"] == "recent_interaction_window"
    assert result.diagnostics["selected_reasons"][0]["continuity_bucket"] == "recent_interaction_window"
    assert result.diagnostics["candidate_origins"]["recent_interactions"] >= 1
    assert result.diagnostics["continuity_buckets"]["recent_interaction_window"] >= 1


def test_memory_retrieval_layer_uses_recent_summary_and_message_windows_before_broader_history() -> None:
    persistence_manager = _FakePersistenceManager(
        {},
        summaries=[
            {
                "id": "summary-1",
                "session_id": "default",
                "summary_text": "Routing migration is paused on the SQLite-first read-path cleanup.",
                "created_at": "2026-04-08T11:30:00Z",
                "summary_scope": "session_state_snapshot",
                "metadata_json": {"objective": "routing migration", "project_id": "lumen"},
            }
        ],
        messages=[
            {
                "id": "message-1",
                "session_id": "default",
                "role": "assistant",
                "content": "We left the routing migration at the SQLite-first read-path cleanup stage.",
                "created_at": "2026-04-08T11:20:00Z",
                "message_metadata_json": {"resolved_prompt": "routing migration", "project_id": "lumen"},
            }
        ],
    )
    layer = MemoryRetrievalLayer(
        interaction_history_service=_ProjectAwareInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=None,
        persistence_manager=persistence_manager,
    )

    result = layer.retrieve(
        prompt="where were we on routing migration?",
        session_id="default",
        project_id="lumen",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    assert result.selected[0].source in {"session_summaries", "message_window"}
    assert result.diagnostics["candidate_origins"]["session_summaries"] >= 1
    assert result.diagnostics["candidate_origins"]["message_window"] >= 1
    assert (
        result.diagnostics["selected_reasons"][0]["continuity_bucket"]
        in {"recent_summary_window", "recent_message_window"}
    )


def test_memory_retrieval_layer_uses_semantic_bonus_for_same_project_memory() -> None:
    schema_note = {
        "title": "schema evolution",
        "content": "SQLite schema migration planning for the Lumen persistence layer.",
        "normalized_topic": "persistence schema",
        "dominant_intent": "planning",
        "note_type": "chronological_research_note",
        "created_at": "2026-04-08T11:00:00Z",
        "note_path": "data/research/default/schema.json",
        "project_id": "lumen",
        "session_id": "default",
    }
    galaxy_note = {
        "title": "galaxy spectrum pass",
        "content": "Spectral inspection notes for a galaxy absorption workflow.",
        "normalized_topic": "galaxy spectrum",
        "dominant_intent": "research",
        "note_type": "chronological_research_note",
        "created_at": "2026-04-08T11:00:01Z",
        "note_path": "data/research/default/galaxy.json",
        "project_id": "lumen",
        "session_id": "default",
    }
    rows = {
        ("research_note", "data/research/default/schema.json"): {
            "status": "ready",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_blob": SemanticEmbeddingService.pack_embedding([1.0, 0.0]),
        },
        ("research_note", "data/research/default/galaxy.json"): {
            "status": "ready",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_blob": SemanticEmbeddingService.pack_embedding([0.0, 1.0]),
        },
    }
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=_StubResearchNoteManager([galaxy_note, schema_note]),
        persistence_manager=_FakePersistenceManager(rows),
        semantic_embedding_service=_FakeSemanticEmbeddingService(),
    )

    result = layer.retrieve(
        prompt="where were we on the sqlite schema migration?",
        session_id="default",
        project_id="lumen",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    lead = result.selected[0]
    assert lead.label == "schema evolution"
    assert float(lead.metadata["semantic_bonus"]) > 0
    assert float(lead.metadata["semantic_similarity"]) > 0.7
    assert lead.metadata["semantic_status"] in {"applied", "applied:capped"}


def test_memory_retrieval_layer_dampens_cross_project_semantic_match() -> None:
    same_project = {
        "title": "lumen migration note",
        "content": "We planned a phased persistence cleanup for Lumen.",
        "normalized_topic": "persistence cleanup",
        "dominant_intent": "planning",
        "note_type": "chronological_research_note",
        "created_at": "2026-04-08T11:00:00Z",
        "note_path": "data/research/default/lumen.json",
        "project_id": "lumen",
        "session_id": "default",
    }
    cross_project = {
        "title": "schema evolution",
        "content": "SQLite schema migration planning for the Lumen persistence layer.",
        "normalized_topic": "persistence schema",
        "dominant_intent": "planning",
        "note_type": "chronological_research_note",
        "created_at": "2026-04-08T11:00:01Z",
        "note_path": "data/research/default/schema.json",
        "project_id": "other",
        "session_id": "other",
    }
    rows = {
        ("research_note", "data/research/default/lumen.json"): {
            "status": "ready",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_blob": SemanticEmbeddingService.pack_embedding([0.4, 0.4]),
        },
        ("research_note", "data/research/default/schema.json"): {
            "status": "ready",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_blob": SemanticEmbeddingService.pack_embedding([1.0, 0.0]),
        },
    }
    layer = MemoryRetrievalLayer(
        interaction_history_service=_NullInteractionHistoryService(),
        archive_service=_NullArchiveService(),
        graph_memory_manager=_NullGraphMemoryManager(),
        personal_memory_manager=None,
        research_note_manager=_StubResearchNoteManager([cross_project, same_project]),
        persistence_manager=_FakePersistenceManager(rows),
        semantic_embedding_service=_FakeSemanticEmbeddingService(),
    )

    result = layer.retrieve(
        prompt="where were we on the sqlite schema migration?",
        session_id="default",
        project_id="lumen",
        active_thread=None,
        route_mode="planning",
    )

    assert result.selected
    lead = result.selected[0]
    assert lead.label == "lumen migration note"
    cross_project_item = next(item for item in result.selected if item.label == "schema evolution")
    assert float(cross_project_item.metadata["semantic_bonus"] or 0.0) < 0.05
    assert str(cross_project_item.metadata.get("semantic_status") or "") in {"applied", "suppressed:cross_project_dampened"}
