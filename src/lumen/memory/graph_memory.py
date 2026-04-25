from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager


@dataclass(slots=True)
class GraphNode:
    id: int
    name: str
    entity_type: str
    observations: list[dict[str, object]]
    relations_out: list[dict[str, object]]
    relations_in: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "observations": list(self.observations),
            "relations_out": list(self.relations_out),
            "relations_in": list(self.relations_in),
        }


class GraphMemoryManager:
    """SQLite-backed durable graph memory for discrete user/project knowledge."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.persistence_manager = PersistenceManager(settings)
        self.persistence_manager.bootstrap()
        self.graph_repository = self.persistence_manager.graph
        self.db_path = settings.persistence_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def create_entities(self, entities: list[dict[str, object]]) -> list[dict[str, object]]:
        return self.graph_repository.create_entities(entities)

    def create_relations(self, relations: list[dict[str, object]]) -> list[dict[str, object]]:
        return self.graph_repository.create_relations(relations)

    def add_observations(self, observations: list[dict[str, object]]) -> list[dict[str, object]]:
        return self.graph_repository.add_observations(observations)

    def delete_entities(self, *, names: list[str] | None = None, ids: list[int] | None = None) -> int:
        return self.graph_repository.delete_entities(names=names, ids=ids)

    def delete_relations(self, *, ids: list[int]) -> int:
        return self.graph_repository.delete_relations(ids=ids)

    def delete_observations(self, *, ids: list[int]) -> int:
        return self.graph_repository.delete_observations(ids=ids)

    def search_nodes(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return self.graph_repository.search_nodes(query, limit=limit)

    def open_nodes(self, *, names: list[str] | None = None, ids: list[int] | None = None) -> list[dict[str, object]]:
        return [node.to_dict() for node in self.graph_repository.open_nodes(names=names, ids=ids)]

    def read_graph(self, *, limit: int = 50) -> dict[str, object]:
        return self.graph_repository.read_graph(limit=limit)

    def ingest_interaction_memory(self, *, record: dict[str, Any]) -> None:
        decision = dict(record.get("memory_write_decision") or {})
        action = str(decision.get("action") or "").strip()
        if action == "save_personal_memory":
            self._ingest_personal_record(record)
        elif action == "save_research_note":
            self._ingest_research_record(record)

    def ingest_research_artifact(self, artifact: dict[str, Any]) -> None:
        title = str(artifact.get("title") or "").strip()
        if not title:
            return
        artifact_type = str(artifact.get("artifact_type") or "finding").strip() or "finding"
        topic = str(artifact.get("normalized_topic") or "").strip()
        content = str(artifact.get("content") or "").strip() or title
        session_id = str(artifact.get("session_id") or "").strip() or None
        with self._connect() as conn:
            artifact_id = self._upsert_node(conn, name=title, entity_type=artifact_type)
            self._insert_observation(
                conn,
                node_id=artifact_id,
                content=content,
                session_id=session_id,
                source_type="research_artifact",
                source_path=str(artifact.get("artifact_path") or "").strip() or None,
                metadata={"source_note_path": str(artifact.get("source_note_path") or "").strip() or None},
            )
            if topic:
                topic_id = self._upsert_node(conn, name=topic, entity_type="project")
                self._upsert_relation(
                    conn,
                    source_id=artifact_id,
                    relation_type="linked_to",
                    target_id=topic_id,
                )

    def _ingest_personal_record(self, record: dict[str, Any]) -> None:
        prompt = str(record.get("prompt") or "").strip()
        summary = str(record.get("summary") or "").strip()
        observation = self._personal_observation(prompt=prompt, summary=summary)
        if not observation:
            return
        preference_name = self._preference_name(prompt=prompt, summary=summary)
        session_id = str(record.get("session_id") or "").strip() or None
        with self._connect() as conn:
            user_id = self._upsert_node(conn, name="user", entity_type="person")
            preference_id = self._upsert_node(conn, name=preference_name, entity_type="preference")
            self._upsert_relation(
                conn,
                source_id=user_id,
                relation_type="prefers",
                target_id=preference_id,
            )
            self._insert_observation(
                conn,
                node_id=preference_id,
                content=observation,
                session_id=session_id,
                source_type="personal_memory",
                source_path=str((record.get("personal_memory") or {}).get("entry_path") or "").strip() or None,
                metadata={"prompt": prompt, "summary": summary},
            )

    def _ingest_research_record(self, record: dict[str, Any]) -> None:
        topic = str(record.get("normalized_topic") or "").strip()
        prompt = str(record.get("prompt") or "").strip()
        summary = str(record.get("summary") or "").strip()
        node_name = topic or prompt or summary
        if not node_name:
            return
        entity_type = self._entity_type_for_record(record)
        session_name = f"project::{str(record.get('session_id') or 'default').strip() or 'default'}"
        session_id = str(record.get("session_id") or "").strip() or None
        with self._connect() as conn:
            topic_id = self._upsert_node(conn, name=node_name, entity_type=entity_type)
            session_node_id = self._upsert_node(conn, name=session_name, entity_type="project")
            self._upsert_relation(
                conn,
                source_id=topic_id,
                relation_type="discovered_in",
                target_id=session_node_id,
            )
            self._insert_observation(
                conn,
                node_id=topic_id,
                content=summary or prompt,
                session_id=session_id,
                source_type="research_note",
                source_path=str((record.get("research_note") or {}).get("note_path") or "").strip() or None,
                metadata={
                    "mode": str(record.get("mode") or "").strip(),
                    "kind": str(record.get("kind") or "").strip(),
                },
            )

    @staticmethod
    def _entity_type_for_record(record: dict[str, Any]) -> str:
        topic = str(record.get("normalized_topic") or "").lower()
        summary = str(record.get("summary") or "").lower()
        if any(token in topic or token in summary for token in ("tool", "adapter", "bundle")):
            return "tool"
        if any(token in topic or token in summary for token in ("system", "routing", "pipeline", "architecture")):
            return "system"
        if any(token in topic or token in summary for token in ("theory", "concept")):
            return "concept"
        if str(record.get("mode") or "").strip() == "planning":
            return "project"
        return "project"

    @staticmethod
    def _preference_name(*, prompt: str, summary: str) -> str:
        lowered = " ".join(prompt.lower().split())
        if "prefer" in lowered:
            return "response_preference"
        if "call me" in lowered:
            return "name_preference"
        return summary[:80].strip() or "saved_preference"

    @staticmethod
    def _personal_observation(*, prompt: str, summary: str) -> str:
        cleaned_prompt = " ".join(prompt.strip().split())
        lowered = cleaned_prompt.lower()
        for prefix in (
            "remember this about me:",
            "save this about me:",
            "remember that i prefer",
            "save that i prefer",
        ):
            if lowered.startswith(prefix):
                return cleaned_prompt[len(prefix):].strip() or summary or cleaned_prompt
        return cleaned_prompt or summary

    def _ensure_schema(self) -> None:
        self.persistence_manager.bootstrap()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.strip().lower().split())

    def _upsert_node(self, conn: sqlite3.Connection, *, name: str, entity_type: str) -> int:
        normalized_name = self._normalize_name(name)
        conn.execute(
            """
            INSERT INTO nodes (name, normalized_name, entity_type, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(normalized_name, entity_type)
            DO UPDATE SET name = excluded.name, updated_at = CURRENT_TIMESTAMP
            """,
            (name, normalized_name, entity_type),
        )
        row = conn.execute(
            "SELECT id FROM nodes WHERE normalized_name = ? AND entity_type = ?",
            (normalized_name, entity_type),
        ).fetchone()
        return int(row["id"])

    def _upsert_relation(
        self,
        conn: sqlite3.Connection,
        *,
        source_id: int,
        relation_type: str,
        target_id: int,
    ) -> int:
        conn.execute(
            """
            INSERT INTO relations (source_node_id, relation_type, target_node_id)
            VALUES (?, ?, ?)
            ON CONFLICT(source_node_id, relation_type, target_node_id)
            DO NOTHING
            """,
            (source_id, relation_type, target_id),
        )
        row = conn.execute(
            """
            SELECT id FROM relations
            WHERE source_node_id = ? AND relation_type = ? AND target_node_id = ?
            """,
            (source_id, relation_type, target_id),
        ).fetchone()
        return int(row["id"])

    def _insert_observation(
        self,
        conn: sqlite3.Connection,
        *,
        node_id: int,
        content: str,
        session_id: str | None,
        source_type: str | None,
        source_path: str | None,
        metadata: dict[str, object] | None,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO observations (node_id, content, session_id, source_type, source_path, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                content,
                session_id,
                source_type,
                source_path,
                json.dumps(metadata, ensure_ascii=True) if metadata else None,
            ),
        )
        return int(cursor.lastrowid)

    def _node_row(self, conn: sqlite3.Connection, *, node_id: int) -> dict[str, object]:
        row = conn.execute(
            "SELECT id, name, entity_type FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "entity_type": str(row["entity_type"]),
        }

    def _relation_row(self, conn: sqlite3.Connection, *, relation_id: int) -> dict[str, object]:
        row = conn.execute(
            """
            SELECT
                r.id,
                s.name AS source_name,
                s.entity_type AS source_type,
                r.relation_type,
                t.name AS target_name,
                t.entity_type AS target_type
            FROM relations r
            JOIN nodes s ON s.id = r.source_node_id
            JOIN nodes t ON t.id = r.target_node_id
            WHERE r.id = ?
            """,
            (relation_id,),
        ).fetchone()
        return {
            "id": int(row["id"]),
            "source_name": str(row["source_name"]),
            "source_type": str(row["source_type"]),
            "relation_type": str(row["relation_type"]),
            "target_name": str(row["target_name"]),
            "target_type": str(row["target_type"]),
        }

    def _observation_row(self, conn: sqlite3.Connection, *, observation_id: int) -> dict[str, object]:
        row = conn.execute(
            """
            SELECT id, node_id, content, session_id, source_type, source_path, created_at
            FROM observations
            WHERE id = ?
            """,
            (observation_id,),
        ).fetchone()
        return {
            "id": int(row["id"]),
            "node_id": int(row["node_id"]),
            "content": str(row["content"]),
            "session_id": str(row["session_id"]) if row["session_id"] is not None else None,
            "source_type": str(row["source_type"]) if row["source_type"] is not None else None,
            "source_path": str(row["source_path"]) if row["source_path"] is not None else None,
            "created_at": str(row["created_at"]),
        }

    def _open_node(self, conn: sqlite3.Connection, *, node_id: int) -> GraphNode:
        node = conn.execute(
            "SELECT id, name, entity_type FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        observations = [
            {
                "id": int(row["id"]),
                "content": str(row["content"]),
                "session_id": str(row["session_id"]) if row["session_id"] is not None else None,
                "source_type": str(row["source_type"]) if row["source_type"] is not None else None,
                "source_path": str(row["source_path"]) if row["source_path"] is not None else None,
                "created_at": str(row["created_at"]),
            }
            for row in conn.execute(
                """
                SELECT id, content, session_id, source_type, source_path, created_at
                FROM observations
                WHERE node_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 10
                """,
                (node_id,),
            ).fetchall()
        ]
        relations_out = [
            {
                "id": int(row["id"]),
                "relation_type": str(row["relation_type"]),
                "target_name": str(row["target_name"]),
                "target_type": str(row["target_type"]),
            }
            for row in conn.execute(
                """
                SELECT r.id, r.relation_type, t.name AS target_name, t.entity_type AS target_type
                FROM relations r
                JOIN nodes t ON t.id = r.target_node_id
                WHERE r.source_node_id = ?
                ORDER BY r.id DESC
                LIMIT 10
                """,
                (node_id,),
            ).fetchall()
        ]
        relations_in = [
            {
                "id": int(row["id"]),
                "relation_type": str(row["relation_type"]),
                "source_name": str(row["source_name"]),
                "source_type": str(row["source_type"]),
            }
            for row in conn.execute(
                """
                SELECT r.id, r.relation_type, s.name AS source_name, s.entity_type AS source_type
                FROM relations r
                JOIN nodes s ON s.id = r.source_node_id
                WHERE r.target_node_id = ?
                ORDER BY r.id DESC
                LIMIT 10
                """,
                (node_id,),
            ).fetchall()
        ]
        return GraphNode(
            id=int(node["id"]),
            name=str(node["name"]),
            entity_type=str(node["entity_type"]),
            observations=observations,
            relations_out=relations_out,
            relations_in=relations_in,
        )
