from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from lumen.knowledge.models import KnowledgeEntry, KnowledgeRelationship


def initialize_schema(conn: sqlite3.Connection) -> None:
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


def seed_entries(
    conn: sqlite3.Connection,
    *,
    entries: tuple[KnowledgeEntry, ...],
    relationships: tuple[KnowledgeRelationship, ...],
    normalize: callable,
) -> None:
    for entry in entries:
        conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_entries (
                id, title, entry_type, category, subcategory,
                summary_short, summary_medium, summary_deep,
                key_points_json, common_questions_json, related_topics_json, tags_json,
                difficulty_level, examples_json, source_type, confidence, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.title,
                entry.entry_type,
                entry.category,
                entry.subcategory,
                entry.summary_short,
                entry.summary_medium,
                entry.summary_deep,
                json.dumps(list(entry.key_points)),
                json.dumps(list(entry.common_questions)),
                json.dumps(list(entry.related_topics)),
                json.dumps(list(entry.tags)),
                entry.difficulty_level,
                json.dumps(list(entry.examples)),
                entry.source_type,
                float(entry.confidence),
                entry.last_updated,
            ),
        )
        conn.execute("DELETE FROM knowledge_aliases WHERE entry_id = ?", (entry.id,))
        aliases = {entry.title, *entry.aliases}
        for alias in aliases:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_aliases(alias, normalized_alias, entry_id) VALUES (?, ?, ?)",
                (alias, normalize(alias), entry.id),
            )
        if entry.formula is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_formulas (
                    entry_id, formula_text, variable_meanings_json, units_json, interpretation, example_usage
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.formula.formula_text,
                    json.dumps(entry.formula.variable_meanings),
                    json.dumps(entry.formula.units),
                    entry.formula.interpretation,
                    entry.formula.example_usage,
                ),
            )
    conn.execute("DELETE FROM knowledge_relationships")
    for relation in relationships:
        conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_relationships(source_entry_id, relation_type, target_entry_id)
            VALUES (?, ?, ?)
            """,
            (relation.source_entry_id, relation.relation_type, relation.target_entry_id),
        )


def import_legacy_database(conn: sqlite3.Connection, *, legacy_db_path: Path | str) -> dict[str, int]:
    path = Path(legacy_db_path)
    if not path.exists():
        return {"entries": 0, "aliases": 0, "relationships": 0, "formulas": 0}
    imported = {"entries": 0, "aliases": 0, "relationships": 0, "formulas": 0}
    with sqlite3.connect(path) as legacy:
        legacy.row_factory = sqlite3.Row
        for row in legacy.execute("SELECT * FROM knowledge_entries").fetchall():
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_entries (
                    id, title, entry_type, category, subcategory,
                    summary_short, summary_medium, summary_deep,
                    key_points_json, common_questions_json, related_topics_json, tags_json,
                    difficulty_level, examples_json, source_type, confidence, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row),
            )
            imported["entries"] += 1
        for row in legacy.execute("SELECT * FROM knowledge_aliases").fetchall():
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_aliases(alias, normalized_alias, entry_id)
                VALUES (?, ?, ?)
                """,
                tuple(row),
            )
            imported["aliases"] += 1
        for row in legacy.execute("SELECT * FROM knowledge_relationships").fetchall():
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_relationships(source_entry_id, relation_type, target_entry_id)
                VALUES (?, ?, ?)
                """,
                tuple(row),
            )
            imported["relationships"] += 1
        for row in legacy.execute("SELECT * FROM knowledge_formulas").fetchall():
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_formulas(
                    entry_id, formula_text, variable_meanings_json, units_json, interpretation, example_usage
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                tuple(row),
            )
            imported["formulas"] += 1
    return imported
