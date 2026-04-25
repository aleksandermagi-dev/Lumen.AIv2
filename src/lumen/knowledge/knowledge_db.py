from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from lumen.db.knowledge_store import import_legacy_database, initialize_schema, seed_entries
from lumen.knowledge.models import KnowledgeEntry, KnowledgeFormula, KnowledgeRelationship


class KnowledgeDB:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        initialize_schema(connection)
        if self.db_path != ":memory:":
            connection.commit()

    def entry_count(self) -> int:
        connection = self.connect()
        row = connection.execute("SELECT COUNT(*) AS count FROM knowledge_entries").fetchone()
        return int(row["count"]) if row is not None else 0

    def seed(self, *, entries: tuple[KnowledgeEntry, ...], relationships: tuple[KnowledgeRelationship, ...]) -> None:
        connection = self.connect()
        seed_entries(
            connection,
            entries=entries,
            relationships=relationships,
            normalize=self.normalize,
        )
        if self.db_path != ":memory:":
            connection.commit()

    def import_legacy_database(self, legacy_db_path: Path | str) -> dict[str, int]:
        connection = self.connect()
        imported = import_legacy_database(connection, legacy_db_path=legacy_db_path)
        if self.db_path != ":memory:":
            connection.commit()
        return imported

    @staticmethod
    def normalize(text: str) -> str:
        normalized = str(text or "").strip().lower()
        for dash in ("-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
            normalized = normalized.replace(dash, " ")
        return " ".join(normalized.split())

    @staticmethod
    def entry_from_row(row: sqlite3.Row, formula_row: sqlite3.Row | None = None) -> KnowledgeEntry:
        formula = None
        if formula_row is not None:
            formula = KnowledgeFormula(
                formula_text=str(formula_row["formula_text"]),
                variable_meanings=json.loads(str(formula_row["variable_meanings_json"])),
                units=json.loads(str(formula_row["units_json"])),
                interpretation=formula_row["interpretation"],
                example_usage=formula_row["example_usage"],
            )
        return KnowledgeEntry(
            id=str(row["id"]),
            title=str(row["title"]),
            entry_type=str(row["entry_type"]),
            category=str(row["category"]),
            subcategory=row["subcategory"],
            aliases=(),
            summary_short=str(row["summary_short"]),
            summary_medium=str(row["summary_medium"]),
            summary_deep=row["summary_deep"],
            key_points=tuple(json.loads(str(row["key_points_json"]))),
            common_questions=tuple(json.loads(str(row["common_questions_json"]))),
            related_topics=tuple(json.loads(str(row["related_topics_json"]))),
            tags=tuple(json.loads(str(row["tags_json"]))),
            difficulty_level=str(row["difficulty_level"]),
            examples=tuple(json.loads(str(row["examples_json"]))),
            source_type=str(row["source_type"]),
            confidence=float(row["confidence"]),
            last_updated=str(row["last_updated"]),
            formula=formula,
        )
