import json
import sqlite3
from datetime import datetime
from pathlib import Path

from lumen.app.controller import AppController
from lumen.app.models import InteractionProfile
from lumen.app.settings import AppSettings
from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.memory.archive_manager import ArchiveManager
from lumen.memory.personal_memory import PersonalMemoryManager
from lumen.memory.research_artifacts import ResearchArtifactManager
from lumen.memory.research_notes import ResearchNoteManager
from lumen.db.persistence_manager import PersistenceManager
from lumen.semantic.embedding_service import SemanticEmbeddingService
from lumen.schemas.archive_schema import ArchiveRecordSchema
from lumen.schemas.interaction_schema import InteractionRecordSchema
from lumen.schemas.personal_memory_schema import PersonalMemorySchema
from lumen.schemas.session_thread_schema import SessionThreadSchema
from lumen.tools.registry_types import ToolResult


class _FakeSemanticEmbeddingService:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def is_available(self) -> bool:
        return True

    def availability_status(self) -> dict[str, object]:
        return {"available": True, "model_name": self.model_name, "error": None}

    def normalize_text(self, text: str | None) -> str:
        return " ".join(str(text or "").strip().split())

    def content_hash(self, text: str | None) -> str:
        return SemanticEmbeddingService().content_hash(text)

    def embed_text(self, text: str | None) -> list[float]:
        normalized = self.normalize_text(text).lower()
        if "routing" in normalized and "migration" in normalized:
            return [1.0, 0.0, 0.0]
        if "preference" in normalized or "phased" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def pack_embedding(self, vector: list[float]) -> bytes:
        return SemanticEmbeddingService.pack_embedding(vector)


def test_persistence_manager_initializes_core_and_secondary_schema(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)

    manager.bootstrap()

    assert settings.persistence_db_path.exists()
    with sqlite3.connect(settings.persistence_db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    tables = {row[0] for row in rows}
    assert {"projects", "sessions", "messages", "session_summaries", "memory_items", "tool_runs"} <= tables
    assert {"bug_logs", "trainability_traces", "preferences", "nodes", "relations", "observations"} <= tables
    assert {"knowledge_entries", "knowledge_aliases", "knowledge_relationships", "knowledge_formulas"} <= tables
    assert {"dataset_import_runs", "dataset_examples", "dataset_example_labels"} <= tables


def test_persistence_manager_records_core_entities_and_secondary_support(tmp_path: Path) -> None:
    bug_dir = tmp_path / "bug ref txt"
    bug_dir.mkdir(parents=True, exist_ok=True)
    (bug_dir / "error codes.txt").write_text("Turtle\nKraken\n", encoding="utf-8")

    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)

    manager.bootstrap()
    manager.record_interaction(
        session_id="default",
        prompt="create a migration plan for lumen",
        response={"mode": "planning", "summary": "A plan outline", "confidence_posture": "supported"},
        record=InteractionRecordSchema.normalize(
            {
                "schema_type": "interaction_record",
                "schema_version": "5",
                "session_id": "default",
                "prompt": "create a migration plan for lumen",
                "mode": "planning",
                "kind": "planning.migration",
                "summary": "A plan outline",
                "created_at": "2026-04-08T00:00:00+00:00",
                "response": {"summary": "A plan outline"},
                "trainability_trace": {"primary_training_surface": "route_decision"},
            }
        ),
        interaction_path=str(tmp_path / "data" / "interactions" / "default" / "20260408T000000Z.json"),
        project_name="Lumen",
    )
    manager.record_session_state(
        session_id="default",
        payload=SessionThreadSchema.normalize(
            {
                "schema_type": "session_thread_state",
                "schema_version": "1",
                "session_id": "default",
                "mode": "planning",
                "kind": "planning.migration",
                "prompt": "create a migration plan for lumen",
                "objective": "Plan work for: create a migration plan for lumen",
                "thread_summary": "migration plan for lumen",
                "summary": "A plan outline",
                "updated_at": "2026-04-08T00:00:01+00:00",
            }
        ),
        project_name="Lumen",
    )
    manager.record_memory_item(
        source_type="personal_memory",
        payload=PersonalMemorySchema.normalize(
            {
                "session_id": "default",
                "created_at": "2026-04-08T00:00:02+00:00",
                "title": "Migration preference",
                "content": "Prefer phased stabilization work.",
            }
        ),
        project_name="Lumen",
    )
    manager.record_tool_run(
        session_id="default",
        archive_record=ArchiveRecordSchema.normalize(
            {
                "schema_type": "archive_record",
                "schema_version": "1",
                "session_id": "default",
                "tool_id": "anh",
                "capability": "spectral_dip_scan",
                "status": "ok",
                "summary": "ANH run completed",
                "created_at": "2026-04-08T00:00:03+00:00",
            }
        ),
        project_name="Lumen",
    )
    manager.bug_logs.upsert(
        bug_log_id="bug:1",
        project_id="general",
        session_id="default",
        bug_type="performance",
        severity="medium",
        title="Slow persistence write",
        description="The first write takes too long.",
        status="open",
        created_at="2026-04-08T00:00:04+00:00",
        taxonomy_label="Turtle",
    )

    session = manager.sessions.get("default")
    messages = manager.messages.list_by_session("default")
    memory_items = manager.memory_items.list_by_filters(project_id=str(session["project_id"]))
    traces = manager.trainability_traces.list_by_session("default")
    bug_pref = manager.preferences.get(key="bug_taxonomy_catalog", scope="system")

    assert session is not None
    assert session["project_id"] in {"lumen", "general"}
    assert len(messages) == 2
    assert memory_items
    assert traces
    assert bug_pref is not None


def test_persistence_manager_auto_title_uses_latest_user_intent_not_assistant_summary(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)

    manager.record_interaction(
        session_id="desktop-title",
        prompt="what is a galaxy?",
        response={"mode": "research", "summary": "A galaxy is a large system of stars."},
        record=InteractionRecordSchema.normalize(
            {
                "session_id": "desktop-title",
                "prompt": "what is a galaxy?",
                "mode": "research",
                "kind": "research.summary",
                "summary": "Here’s the best first read. Best next check: stale scaffold",
                "created_at": "2026-04-23T12:00:00+00:00",
                "response": {"summary": "A galaxy is a large system of stars."},
            }
        ),
        interaction_path=None,
    )
    manager.record_interaction(
        session_id="desktop-title",
        prompt="tell me about the zodiac",
        response={"mode": "research", "summary": "The zodiac is a cultural symbol system."},
        record=InteractionRecordSchema.normalize(
            {
                "session_id": "desktop-title",
                "prompt": "tell me about the zodiac",
                "mode": "research",
                "kind": "research.summary",
                "summary": "The zodiac is a cultural symbol system.",
                "created_at": "2026-04-23T12:01:00+00:00",
                "response": {"summary": "The zodiac is a cultural symbol system."},
            }
        ),
        interaction_path=None,
    )

    session = manager.sessions.get("desktop-title")

    assert session is not None
    assert session["title"] == "tell me about the zodiac"
    metadata = session["metadata_json"]
    assert isinstance(metadata, dict)
    assert metadata["title_source"] == "auto_user_intent"
    assert metadata["title_locked"] is False


def test_persistence_manager_lists_saved_turns_user_before_assistant(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)
    created_at = "2026-04-08T00:00:00+00:00"

    manager.record_interaction(
        session_id="default",
        prompt="hi lumen",
        response={"mode": "conversation", "summary": "Hi. I'm here."},
        record=InteractionRecordSchema.normalize(
            {
                "schema_type": "interaction_record",
                "schema_version": "5",
                "session_id": "default",
                "prompt": "hi lumen",
                "mode": "conversation",
                "kind": "conversation.social",
                "summary": "Hi. I'm here.",
                "created_at": created_at,
                "response": {"summary": "Hi. I'm here."},
            }
        ),
        interaction_path=str(tmp_path / "data" / "interactions" / "default" / "20260408T000000Z.json"),
    )

    messages = manager.messages.list_by_session("default")

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hi lumen"
    assert messages[1]["content"] == "Hi. I'm here."


def test_persistence_manager_creates_and_reports_memory_item_embeddings(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)
    manager.semantic_embedding_service = _FakeSemanticEmbeddingService()

    memory_row = manager.record_memory_item(
        source_type="research_note",
        payload={
            "session_id": "default",
            "created_at": "2026-04-08T00:00:02+00:00",
            "title": "routing migration",
            "content": "Routing migration stays split into controller and service work.",
            "note_path": str(tmp_path / "data" / "research_notes" / "default" / "routing.json"),
            "normalized_topic": "routing migration",
            "dominant_intent": "planning",
        },
        project_name="Lumen",
    )

    embedding = manager.memory_embeddings.get(str(memory_row["id"]))
    status = manager.semantic_status_report()

    assert embedding is not None
    assert embedding["status"] == "ready"
    assert embedding["embedding_dim"] == 3
    assert status["embedded_memory_items"] == 1
    assert status["ready_embeddings"] == 1
    assert "fallback_retirement_readiness" in manager.status_report()


def test_persistence_manager_backfill_embeddings_is_idempotent(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersistenceManager(settings)
    manager.semantic_embedding_service = _FakeSemanticEmbeddingService()

    row = manager.record_memory_item(
        source_type="personal_memory",
        payload={
            "session_id": "default",
            "created_at": "2026-04-08T00:00:02+00:00",
            "title": "Migration preference",
            "content": "Prefer phased stabilization work.",
            "entry_path": str(tmp_path / "data" / "personal_memory" / "default" / "pref.json"),
            "normalized_topic": "preferences",
        },
        project_name="Lumen",
    )

    manager.memory_embeddings.upsert(
        memory_item_id=str(row["id"]),
        source_id=str(row["source_id"]),
        source_type=str(row["source_type"]),
        model_name=manager.semantic_embedding_service.model_name,
        embedding_dim=None,
        embedding_blob=None,
        content_hash="stale",
        status="pending",
        created_at="2026-04-08T00:00:02+00:00",
        updated_at="2026-04-08T00:00:02+00:00",
    )

    first = manager.backfill_memory_item_embeddings()
    second = manager.backfill_memory_item_embeddings()
    embedding = manager.memory_embeddings.get(str(row["id"]))

    assert first["processed"] >= 1
    assert second["ready"] >= 0
    assert embedding is not None
    assert embedding["status"] == "ready"
    assert embedding["content_hash"] == manager.semantic_embedding_service.content_hash(str(row["content"]))


def test_persistence_manager_imports_legacy_json_and_graph_once_without_duplication(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    interaction_dir = settings.interactions_root / "default"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    interaction_path = interaction_dir / "20260408T000000Z.json"
    interaction_path.write_text(
        json.dumps(
            InteractionRecordSchema.normalize(
                {
                    "schema_type": "interaction_record",
                    "schema_version": "5",
                    "session_id": "default",
                    "prompt": "summarize lumen",
                    "mode": "research",
                    "kind": "research.general",
                    "summary": "Lumen summary",
                    "created_at": "2026-04-08T00:00:00+00:00",
                    "response": {"summary": "Lumen summary"},
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    settings.sessions_root.mkdir(parents=True, exist_ok=True)
    session_dir = settings.sessions_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "thread_state.json").write_text(
        json.dumps(
            SessionThreadSchema.normalize(
                {
                    "schema_type": "session_thread_state",
                    "schema_version": "1",
                    "session_id": "default",
                    "mode": "research",
                    "kind": "research.general",
                    "prompt": "summarize lumen",
                    "objective": "Research topic: summarize lumen",
                    "thread_summary": "summarize lumen",
                    "summary": "Lumen summary",
                    "updated_at": "2026-04-08T00:00:01+00:00",
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    legacy_graph = settings.graph_memory_db_path
    legacy_graph.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(legacy_graph) as conn:
        conn.executescript(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(normalized_name, entity_type)
            );
            CREATE TABLE relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                target_node_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_node_id, relation_type, target_node_id)
            );
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT,
                source_type TEXT,
                source_path TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO nodes(name, normalized_name, entity_type) VALUES ('Lumen', 'lumen', 'project');
            INSERT INTO observations(node_id, content, session_id, source_type) VALUES (1, 'Lumen note', 'default', 'legacy');
            """
        )

    manager = PersistenceManager(settings)
    manager.bootstrap(run_imports=True)
    manager.bootstrap(run_imports=True)

    messages = manager.messages.list_by_session("default")
    graph = manager.graph.read_graph(limit=10)

    assert len(messages) == 2
    assert graph["node_count"] >= 1


def test_persistence_manager_imports_legacy_knowledge_into_unified_db(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    settings.knowledge_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.knowledge_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE knowledge_entries (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                summary_short TEXT NOT NULL,
                summary_medium TEXT NOT NULL,
                summary_deep TEXT,
                key_points_json TEXT NOT NULL,
                common_questions_json TEXT NOT NULL,
                related_topics_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                difficulty_level TEXT NOT NULL,
                examples_json TEXT NOT NULL,
                source_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                last_updated TEXT NOT NULL
            );
            CREATE TABLE knowledge_aliases (
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                PRIMARY KEY (normalized_alias, entry_id)
            );
            CREATE TABLE knowledge_relationships (
                source_entry_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entry_id TEXT NOT NULL,
                PRIMARY KEY (source_entry_id, relation_type, target_entry_id)
            );
            CREATE TABLE knowledge_formulas (
                entry_id TEXT PRIMARY KEY,
                formula_text TEXT NOT NULL,
                variable_meanings_json TEXT NOT NULL,
                units_json TEXT NOT NULL,
                interpretation TEXT,
                example_usage TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO knowledge_entries (
                id, title, entry_type, category, subcategory, summary_short, summary_medium,
                summary_deep, key_points_json, common_questions_json, related_topics_json,
                tags_json, difficulty_level, examples_json, source_type, confidence, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy.custom_flux",
                "Legacy Flux",
                "concept",
                "physics",
                None,
                "Legacy flux summary",
                "Legacy flux medium summary",
                None,
                json.dumps(["Flux carries rate information."]),
                json.dumps([]),
                json.dumps([]),
                json.dumps(["legacy"]),
                "medium",
                json.dumps([]),
                "legacy_seed",
                0.8,
                "2026-04-08T00:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO knowledge_aliases(alias, normalized_alias, entry_id) VALUES (?, ?, ?)",
            ("Legacy Flux", "legacy flux", "legacy.custom_flux"),
        )

    manager = PersistenceManager(settings)
    manager.bootstrap(run_imports=True)

    service = KnowledgeService.from_path(settings.persistence_db_path)
    lookup = service.lookup("legacy flux")
    status = manager.status_report()

    assert lookup is not None
    assert lookup.primary.title == "Legacy Flux"
    assert status["table_counts"]["knowledge_entries"] >= 1
    assert any(item["name"] == "0004_legacy_knowledge_import" for item in status["import_runs"])


def test_controller_ask_accepts_project_name_and_persists_session_rows(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="create a migration plan for lumen",
        session_id="default",
        project_name="Lumen",
    )

    manager = PersistenceManager(controller.settings)
    manager.bootstrap()
    session = manager.sessions.get("default")
    project = manager.projects.get_by_name("Lumen")

    assert session is not None
    assert project is not None
    assert session["project_id"] == project["id"]


def test_db_backed_interaction_reads_survive_missing_legacy_json(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(
        prompt="create a migration plan for lumen routing",
        session_id="default",
        project_name="Lumen",
    )

    interaction_files = list((controller.settings.interactions_root / "default").glob("*.json"))
    for path in interaction_files:
        if path.name != "_index.json":
            path.unlink()

    listing = controller.list_interactions(session_id="default")

    assert listing["interaction_count"] == 1
    assert listing["interaction_records"][0]["prompt"] == "create a migration plan for lumen routing"


def test_db_backed_active_thread_reads_survive_missing_thread_state(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a roadmap for developing lumen further", session_id="default")

    thread_state = controller.settings.sessions_root / "default" / "thread_state.json"
    if thread_state.exists():
        thread_state.unlink()

    report = controller.current_session_thread("default")

    assert report["active_thread"] is not None
    assert report["active_thread"]["prompt"] == "create a roadmap for developing lumen further"


def test_db_backed_research_note_reads_survive_missing_legacy_file(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)
    note_manager = ResearchNoteManager(controller.settings, persistence_manager=controller.persistence_manager)
    note = note_manager.record_note(
        session_id="default",
        timestamp=datetime.fromisoformat("2026-04-08T00:00:00+00:00"),
        record={
            "prompt": "Summarize the routing migration plan",
            "summary": "Routing migration is staged around controller then service cleanup.",
            "mode": "research",
            "kind": "research.general",
            "normalized_topic": "routing migration",
            "dominant_intent": "planning",
            "memory_classification": {"save_eligible": True, "candidate_type": "research_memory_candidate"},
        },
    )

    Path(str(note["note_path"])).unlink()

    report = controller.list_research_notes(session_id="default")

    assert report["note_count"] == 1
    assert report["research_notes"][0]["title"] == "routing migration"


def test_db_backed_archive_reads_survive_missing_legacy_json(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    persistence_manager = PersistenceManager(settings)
    manager = ArchiveManager(settings=settings, persistence_manager=persistence_manager)
    run_dir = tmp_path / "data" / "tool_runs" / "default" / "anh" / "spectral_dip_scan" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    record = manager.record_tool_run(
        session_id="default",
        result=ToolResult(
            status="ok",
            tool_id="anh",
            capability="spectral_dip_scan",
            summary="GA Local Analysis Kit run completed",
            structured_data={
                "run_id": "run_2026_03_16_213045",
                "target_label": "GA Local Analysis Kit",
                "analysis_status": {"result_quality": "scientific_output_present"},
            },
            run_dir=run_dir,
        ),
    )

    record.archive_path.unlink()

    records = manager.list_records(session_id="default")

    assert len(records) == 1
    assert records[0]["target_label"] == "GA Local Analysis Kit"
    assert records[0]["result_quality"] == "scientific_output_present"


def test_db_backed_session_metadata_and_profile_survive_missing_legacy_files(tmp_path: Path) -> None:
    controller = AppController(repo_root=tmp_path)

    controller.ask(prompt="create a roadmap for lumen persistence", session_id="default")
    controller.rename_session("default", title="Persistence roadmap")
    controller.session_context_service.set_interaction_profile(
        "default",
        InteractionProfile(interaction_style="direct"),
    )

    session_dir = controller.settings.sessions_root / "default"
    for name in ("session_metadata.json", "interaction_profile.json"):
        path = session_dir / name
        if path.exists():
            path.unlink()

    thread = controller.current_session_thread("default")
    metadata = controller.session_context_service.get_session_metadata("default")

    assert thread["interaction_profile"]["interaction_style"] == "direct"
    assert metadata["title"] == "Persistence roadmap"


def test_persistence_doctor_reports_fallback_reads_after_legacy_structured_read(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ResearchNoteManager(settings)
    session_dir = settings.research_notes_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    note_path = session_dir / "20260408T000000000000Z.json"
    note_path.write_text(
        json.dumps(
            {
                "session_id": "default",
                "created_at": "2026-04-08T00:00:00+00:00",
                "note_type": "chronological_research_note",
                "title": "routing migration",
                "content": "Routing migration remains staged.",
                "normalized_topic": "routing migration",
            }
        ),
        encoding="utf-8",
    )

    notes = manager.list_notes(session_id="default")
    report = manager.persistence_manager.doctor_report()

    assert len(notes) == 1
    assert report["fallback_read_count"] >= 1
    assert "research note structured listing fallback" in report["fallback_read_surfaces"]
    assert report["fallback_surface_categories"]["research note structured listing fallback"] == "compat_fallback"
    readiness = report["fallback_retirement_readiness"]["research note structured listing fallback"]
    assert readiness["db_parity_expected"] is True
    assert readiness["recovery_critical"] is False
    assert readiness["retire_candidate"] is False


def test_personal_memory_fallback_listing_respects_limit_without_full_sort_requirement(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = PersonalMemoryManager(settings)
    session_dir = settings.personal_memory_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        stamp = f"20260408T00000000000{index}Z"
        (session_dir / f"{stamp}.json").write_text(
            json.dumps(
                {
                    "session_id": "default",
                    "created_at": f"2026-04-08T00:00:0{index}+00:00",
                    "title": f"memory-{index}",
                    "content": "Remember this.",
                }
            ),
            encoding="utf-8",
        )

    entries = manager.list_entries(session_id="default", limit=4)

    assert [entry["title"] for entry in entries] == ["memory-9", "memory-8", "memory-7", "memory-6"]


def test_research_note_fallback_listing_respects_limit_without_full_sort_requirement(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ResearchNoteManager(settings)
    session_dir = settings.research_notes_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        stamp = f"20260408T00000000000{index}Z"
        (session_dir / f"{stamp}.json").write_text(
            json.dumps(
                {
                    "session_id": "default",
                    "created_at": f"2026-04-08T00:00:0{index}+00:00",
                    "note_type": "chronological_research_note",
                    "title": f"note-{index}",
                    "content": "Track this.",
                }
            ),
            encoding="utf-8",
        )

    notes = manager.list_notes(session_id="default", limit=4)

    assert [note["title"] for note in notes] == ["note-9", "note-8", "note-7", "note-6"]


def test_research_artifact_manager_note_listing_honors_limit(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ResearchArtifactManager(settings)
    session_dir = settings.research_notes_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        stamp = f"20260408T00000000000{index}Z"
        (session_dir / f"{stamp}.json").write_text(
            json.dumps(
                {
                    "session_id": "default",
                    "created_at": f"2026-04-08T00:00:0{index}+00:00",
                    "note_type": "chronological_research_note",
                    "title": f"artifact-note-{index}",
                    "content": "Track this.",
                    "source_interaction_prompt": "Track this.",
                    "source_interaction_mode": "research",
                }
            ),
            encoding="utf-8",
        )

    notes = manager.list_notes(session_id="default", limit=4)

    assert [note["title"] for note in notes] == [
        "artifact-note-9",
        "artifact-note-8",
        "artifact-note-7",
        "artifact-note-6",
    ]


def test_research_artifact_manager_archived_note_listing_honors_limit(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)
    manager = ResearchArtifactManager(settings)
    session_dir = settings.research_notes_root / "default"
    session_dir.mkdir(parents=True, exist_ok=True)
    for index in range(8):
        stamp = f"20260408T00000000000{index}Z"
        (session_dir / f"{stamp}.json").write_text(
            json.dumps(
                {
                    "session_id": "default",
                    "created_at": f"2026-04-08T00:00:0{index}+00:00",
                    "note_type": "chronological_research_note",
                    "title": f"archived-note-{index}",
                    "content": "Track this.",
                    "source_interaction_prompt": "Track this.",
                    "source_interaction_mode": "research",
                    "archived": index >= 4,
                }
            ),
            encoding="utf-8",
        )

    notes = manager.list_notes(session_id="default", include_archived=True, archived_only=True, limit=2)

    assert [note["title"] for note in notes] == ["archived-note-7", "archived-note-6"]
