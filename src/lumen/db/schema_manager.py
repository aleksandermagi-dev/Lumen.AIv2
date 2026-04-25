from __future__ import annotations

import sqlite3

from lumen.db.database_manager import DatabaseManager
from lumen.db.migration_runner import MigrationRunner
from lumen.db.repositories import PreferenceRepository, ProjectRepository


class SchemaManager:
    """Bootstraps the unified persistence schema and explicit migrations."""

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.migration_runner = MigrationRunner(database_manager)
        self.project_repository = ProjectRepository(database_manager)
        self.preference_repository = PreferenceRepository(database_manager)

    def initialize(self) -> None:
        self.migration_runner.apply("0001", "core_foundation", self._migration_0001_core_foundation)
        self.migration_runner.apply("0002", "secondary_support", self._migration_0002_secondary_support)
        self.migration_runner.apply("0003", "knowledge_catalog", self._migration_0003_knowledge_catalog)
        self.migration_runner.apply("0004", "memory_item_embeddings", self._migration_0004_memory_item_embeddings)
        self.migration_runner.apply("0005", "dataset_ingestion_layer", self._migration_0005_dataset_ingestion_layer)
        self.project_repository.ensure_general_project()
        self._seed_bug_taxonomy()

    def _migration_0001_core_foundation(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                description TEXT,
                status TEXT NOT NULL,
                tags_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT,
                mode TEXT,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                summary_id TEXT,
                metadata_json TEXT,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                intent_domain TEXT,
                confidence_tier TEXT,
                response_depth TEXT,
                conversation_phase TEXT,
                tool_usage_intent TEXT,
                route_decision_json TEXT,
                message_metadata_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                confidence_tier TEXT,
                tags_json TEXT,
                summary_scope TEXT,
                source_message_start_id TEXT,
                source_message_end_id TEXT,
                metadata_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(source_message_start_id) REFERENCES messages(id) ON DELETE SET NULL,
                FOREIGN KEY(source_message_end_id) REFERENCES messages(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                project_id TEXT,
                session_id TEXT,
                category TEXT NOT NULL,
                domain TEXT,
                content TEXT NOT NULL,
                confidence_tier TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                recency_weight REAL,
                relevance_hint TEXT,
                status TEXT,
                source_summary TEXT,
                metadata_json TEXT,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS tool_runs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT,
                project_id TEXT,
                tool_name TEXT NOT NULL,
                capability TEXT,
                input_summary TEXT,
                output_summary TEXT,
                success INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                confidence_impact TEXT,
                latency_ms INTEGER,
                tool_bundle TEXT,
                archive_path TEXT,
                run_dir TEXT,
                metadata_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(normalized_name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                target_node_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_node_id, relation_type, target_node_id),
                FOREIGN KEY(source_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target_node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT,
                source_type TEXT,
                source_path TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions(project_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
            CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_intent_domain ON messages(intent_domain);
            CREATE INDEX IF NOT EXISTS idx_session_summaries_session_created ON session_summaries(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_items_project_id ON memory_items(project_id);
            CREATE INDEX IF NOT EXISTS idx_memory_items_domain ON memory_items(domain);
            CREATE INDEX IF NOT EXISTS idx_memory_items_category ON memory_items(category);
            CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_items_confidence_tier ON memory_items(confidence_tier);
            CREATE INDEX IF NOT EXISTS idx_tool_runs_session_created ON tool_runs(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(entity_type);
            CREATE INDEX IF NOT EXISTS idx_observations_node ON observations(node_id);
            CREATE INDEX IF NOT EXISTS idx_observations_source ON observations(source_type);
            """
        )

    def _migration_0002_secondary_support(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS bug_logs (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                session_id TEXT,
                bug_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                source_component TEXT,
                taxonomy_label TEXT,
                metadata_json TEXT,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS trainability_traces (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT,
                decision_type TEXT NOT NULL,
                input_context_summary TEXT,
                chosen_action TEXT,
                outcome TEXT,
                label TEXT,
                confidence_tier TEXT,
                created_at TEXT NOT NULL,
                model_assist_used INTEGER,
                evaluation_score REAL,
                metadata_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                scope TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bug_logs_project_created ON bug_logs(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_trainability_traces_session_created ON trainability_traces(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_trainability_traces_decision_type ON trainability_traces(decision_type);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_scope_key ON preferences(scope, key);
            """
        )

    def _migration_0003_knowledge_catalog(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
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

            CREATE TABLE IF NOT EXISTS knowledge_aliases (
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                PRIMARY KEY (normalized_alias, entry_id),
                FOREIGN KEY(entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                source_entry_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entry_id TEXT NOT NULL,
                PRIMARY KEY (source_entry_id, relation_type, target_entry_id),
                FOREIGN KEY(source_entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
                FOREIGN KEY(target_entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS knowledge_formulas (
                entry_id TEXT PRIMARY KEY,
                formula_text TEXT NOT NULL,
                variable_meanings_json TEXT NOT NULL,
                units_json TEXT NOT NULL,
                interpretation TEXT,
                example_usage TEXT,
                FOREIGN KEY(entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_aliases_norm ON knowledge_aliases(normalized_alias);
            CREATE INDEX IF NOT EXISTS idx_knowledge_entries_title ON knowledge_entries(title);
            """
        )

    def _migration_0004_memory_item_embeddings(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_item_embeddings (
                memory_item_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER,
                embedding_blob BLOB,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT,
                FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_item_embeddings_status ON memory_item_embeddings(status);
            CREATE INDEX IF NOT EXISTS idx_memory_item_embeddings_source ON memory_item_embeddings(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_memory_item_embeddings_hash ON memory_item_embeddings(content_hash);
            """
        )

    def _migration_0005_dataset_ingestion_layer(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dataset_import_runs (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                dataset_version TEXT,
                source_path TEXT,
                source_format TEXT NOT NULL,
                dataset_kind TEXT NOT NULL,
                import_strategy TEXT NOT NULL,
                ingestion_status TEXT NOT NULL,
                example_count INTEGER NOT NULL,
                train_count INTEGER NOT NULL DEFAULT 0,
                validation_count INTEGER NOT NULL DEFAULT 0,
                test_count INTEGER NOT NULL DEFAULT 0,
                schema_version TEXT NOT NULL,
                notes_json TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS dataset_examples (
                id TEXT PRIMARY KEY,
                import_run_id TEXT NOT NULL,
                example_type TEXT NOT NULL,
                source_format TEXT NOT NULL,
                split_assignment TEXT NOT NULL,
                ingestion_state TEXT NOT NULL,
                input_text TEXT NOT NULL,
                target_text TEXT,
                label_category TEXT,
                label_value TEXT,
                explanation_text TEXT,
                source_session_id TEXT,
                source_message_id TEXT,
                source_interaction_path TEXT,
                source_trace_id TEXT,
                source_tool_run_id TEXT,
                label_source TEXT NOT NULL,
                trainable INTEGER NOT NULL,
                provenance_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(import_run_id) REFERENCES dataset_import_runs(id) ON DELETE CASCADE,
                FOREIGN KEY(source_session_id) REFERENCES sessions(id) ON DELETE SET NULL,
                FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE SET NULL,
                FOREIGN KEY(source_trace_id) REFERENCES trainability_traces(id) ON DELETE SET NULL,
                FOREIGN KEY(source_tool_run_id) REFERENCES tool_runs(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_example_labels (
                id TEXT PRIMARY KEY,
                dataset_example_id TEXT NOT NULL,
                label_role TEXT NOT NULL,
                label_value TEXT NOT NULL,
                label_category TEXT,
                is_canonical INTEGER NOT NULL,
                reviewer TEXT,
                reason TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY(dataset_example_id) REFERENCES dataset_examples(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_dataset_import_runs_name ON dataset_import_runs(dataset_name);
            CREATE INDEX IF NOT EXISTS idx_dataset_import_runs_created_at ON dataset_import_runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_dataset_import_runs_status ON dataset_import_runs(ingestion_status);

            CREATE INDEX IF NOT EXISTS idx_dataset_examples_import_run ON dataset_examples(import_run_id);
            CREATE INDEX IF NOT EXISTS idx_dataset_examples_type ON dataset_examples(example_type);
            CREATE INDEX IF NOT EXISTS idx_dataset_examples_split ON dataset_examples(split_assignment);
            CREATE INDEX IF NOT EXISTS idx_dataset_examples_label_category ON dataset_examples(label_category);
            CREATE INDEX IF NOT EXISTS idx_dataset_examples_label_source ON dataset_examples(label_source);
            CREATE INDEX IF NOT EXISTS idx_dataset_examples_ingestion_state ON dataset_examples(ingestion_state);

            CREATE INDEX IF NOT EXISTS idx_dataset_labels_example_id ON dataset_example_labels(dataset_example_id);
            CREATE INDEX IF NOT EXISTS idx_dataset_labels_role ON dataset_example_labels(label_role);
            CREATE INDEX IF NOT EXISTS idx_dataset_labels_canonical ON dataset_example_labels(dataset_example_id, label_role, is_canonical);
            """
        )

    def _seed_bug_taxonomy(self) -> None:
        bug_reference_path = self.database_manager.settings.repo_root / "bug ref txt" / "error codes.txt"
        if not bug_reference_path.exists():
            return
        value = bug_reference_path.read_text(encoding="utf-8")
        self.preference_repository.set(
            preference_id="system:bug_taxonomy_catalog",
            key="bug_taxonomy_catalog",
            value=value,
            scope="system",
        )
