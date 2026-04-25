from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from lumen.db.database_manager import DatabaseManager


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


class GraphRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.db_path = database_manager.db_path

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(str(name or "").strip().lower().split())

    def create_entities(self, entities: list[dict[str, object]]) -> list[dict[str, object]]:
        created: list[dict[str, object]] = []
        with self.database_manager.transaction() as conn:
            for entity in entities:
                name = str(entity.get("name") or "").strip()
                entity_type = str(entity.get("entity_type") or "").strip() or "note"
                if not name:
                    continue
                node_id = self._upsert_node(conn, name=name, entity_type=entity_type)
                created.append(self._node_row(conn, node_id=node_id))
        return created

    def create_relations(self, relations: list[dict[str, object]]) -> list[dict[str, object]]:
        created: list[dict[str, object]] = []
        with self.database_manager.transaction() as conn:
            for relation in relations:
                source_name = str(relation.get("source_name") or relation.get("source") or "").strip()
                source_type = str(relation.get("source_type") or "").strip() or "note"
                target_name = str(relation.get("target_name") or relation.get("target") or "").strip()
                target_type = str(relation.get("target_type") or "").strip() or "note"
                relation_type = str(relation.get("relation_type") or "").strip()
                if not source_name or not target_name or not relation_type:
                    continue
                source_id = self._upsert_node(conn, name=source_name, entity_type=source_type)
                target_id = self._upsert_node(conn, name=target_name, entity_type=target_type)
                relation_id = self._upsert_relation(
                    conn,
                    source_id=source_id,
                    relation_type=relation_type,
                    target_id=target_id,
                )
                created.append(self._relation_row(conn, relation_id=relation_id))
        return created

    def add_observations(self, observations: list[dict[str, object]]) -> list[dict[str, object]]:
        created: list[dict[str, object]] = []
        with self.database_manager.transaction() as conn:
            for observation in observations:
                node_name = str(observation.get("entity_name") or observation.get("node_name") or "").strip()
                entity_type = str(observation.get("entity_type") or "").strip() or "note"
                content = str(observation.get("content") or "").strip()
                if not node_name or not content:
                    continue
                node_id = self._upsert_node(conn, name=node_name, entity_type=entity_type)
                observation_id = self._insert_observation(
                    conn,
                    node_id=node_id,
                    content=content,
                    session_id=str(observation.get("session_id") or "").strip() or None,
                    source_type=str(observation.get("source_type") or "").strip() or None,
                    source_path=str(observation.get("source_path") or "").strip() or None,
                )
                created.append(self._observation_row(conn, observation_id=observation_id))
        return created

    def delete_entities(self, *, names: list[str] | None = None, ids: list[int] | None = None) -> int:
        with self.database_manager.transaction() as conn:
            count = 0
            if ids:
                placeholders = ",".join("?" for _ in ids)
                count += conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", tuple(ids)).rowcount
            if names:
                normalized = tuple(self._normalize_name(name) for name in names if name.strip())
                if normalized:
                    placeholders = ",".join("?" for _ in normalized)
                    count += conn.execute(
                        f"DELETE FROM nodes WHERE normalized_name IN ({placeholders})",
                        normalized,
                    ).rowcount
        return count

    def delete_relations(self, *, ids: list[int]) -> int:
        if not ids:
            return 0
        with self.database_manager.transaction() as conn:
            placeholders = ",".join("?" for _ in ids)
            return conn.execute(f"DELETE FROM relations WHERE id IN ({placeholders})", tuple(ids)).rowcount

    def delete_observations(self, *, ids: list[int]) -> int:
        if not ids:
            return 0
        with self.database_manager.transaction() as conn:
            placeholders = ",".join("?" for _ in ids)
            return conn.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", tuple(ids)).rowcount

    def search_nodes(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        normalized_query = " ".join(str(query or "").strip().lower().split())
        if not normalized_query:
            return []
        tokens: list[str] = []
        for token in normalized_query.split():
            cleaned = token.strip()
            if not cleaned:
                continue
            tokens.append(cleaned)
            if len(cleaned) > 3 and cleaned.endswith("s"):
                tokens.append(cleaned[:-1])
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    n.id,
                    n.name,
                    n.entity_type,
                    o.id AS observation_id,
                    o.content AS observation_content
                FROM nodes n
                LEFT JOIN observations o ON o.node_id = n.id
                ORDER BY n.updated_at DESC, o.created_at DESC
                """
            ).fetchall()
        scored: dict[int, dict[str, object]] = {}
        for row in rows:
            node_id = int(row["id"])
            node_name = str(row["name"] or "")
            entity_type = str(row["entity_type"] or "")
            observation = str(row["observation_content"] or "")
            haystacks = [node_name.lower(), entity_type.lower(), observation.lower()]
            score = 0
            for token in tokens:
                if token in haystacks[0]:
                    score += 4
                if token in haystacks[1]:
                    score += 2
                if token in haystacks[2]:
                    score += 3
            if score <= 0:
                continue
            current = scored.get(node_id)
            preview = observation.strip()
            if current is None or score > int(current["score"]):
                scored[node_id] = {
                    "id": node_id,
                    "name": node_name,
                    "entity_type": entity_type,
                    "score": score,
                    "observation_preview": preview or None,
                }
        results = sorted(
            scored.values(),
            key=lambda item: (int(item["score"]), str(item["name"]).lower()),
            reverse=True,
        )
        return results[:limit]

    def open_nodes(self, *, names: list[str] | None = None, ids: list[int] | None = None) -> list[GraphNode]:
        rows: list[sqlite3.Row] = []
        with self.database_manager.connect() as conn:
            if ids:
                placeholders = ",".join("?" for _ in ids)
                rows.extend(conn.execute(f"SELECT id FROM nodes WHERE id IN ({placeholders})", tuple(ids)).fetchall())
            if names:
                normalized = tuple(self._normalize_name(name) for name in names if name.strip())
                if normalized:
                    placeholders = ",".join("?" for _ in normalized)
                    rows.extend(
                        conn.execute(
                            f"SELECT id FROM nodes WHERE normalized_name IN ({placeholders})",
                            normalized,
                        ).fetchall()
                    )
            unique_ids = sorted({int(row["id"]) for row in rows})
            return [self._open_node(conn, node_id=node_id) for node_id in unique_ids]

    def read_graph(self, *, limit: int = 50) -> dict[str, object]:
        with self.database_manager.connect() as conn:
            nodes = conn.execute(
                "SELECT id FROM nodes ORDER BY updated_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            payloads = [self._open_node(conn, node_id=int(row["id"])).to_dict() for row in nodes]
            relation_count = int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
            observation_count = int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
        return {
            "db_path": str(self.db_path),
            "node_count": len(payloads),
            "relation_count": relation_count,
            "observation_count": observation_count,
            "nodes": payloads,
        }

    def import_legacy_graph(self, legacy_db_path) -> None:
        if not legacy_db_path or not legacy_db_path.exists():
            return
        if legacy_db_path.resolve() == self.db_path.resolve():
            return
        legacy = sqlite3.connect(legacy_db_path)
        legacy.row_factory = sqlite3.Row
        try:
            nodes = legacy.execute("SELECT name, entity_type FROM nodes").fetchall()
            relations = legacy.execute(
                """
                SELECT s.name AS source_name, s.entity_type AS source_type, r.relation_type,
                       t.name AS target_name, t.entity_type AS target_type
                FROM relations r
                JOIN nodes s ON s.id = r.source_node_id
                JOIN nodes t ON t.id = r.target_node_id
                """
            ).fetchall()
            observations = legacy.execute(
                """
                SELECT n.name AS node_name, n.entity_type, o.content, o.session_id, o.source_type, o.source_path
                FROM observations o
                JOIN nodes n ON n.id = o.node_id
                """
            ).fetchall()
        finally:
            legacy.close()
        self.create_entities([dict(row) for row in nodes])
        self.create_relations([dict(row) for row in relations])
        self.add_observations([dict(row) for row in observations])

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

    def _upsert_relation(self, conn: sqlite3.Connection, *, source_id: int, relation_type: str, target_id: int) -> int:
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
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO observations (node_id, content, session_id, source_type, source_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (node_id, content, session_id, source_type, source_path),
        )
        return int(cursor.lastrowid)

    def _node_row(self, conn: sqlite3.Connection, *, node_id: int) -> dict[str, object]:
        row = conn.execute("SELECT id, name, entity_type FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return {"id": int(row["id"]), "name": str(row["name"]), "entity_type": str(row["entity_type"])}

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
            "SELECT id, node_id, content, session_id, source_type, source_path, created_at FROM observations WHERE id = ?",
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
        node = conn.execute("SELECT id, name, entity_type FROM nodes WHERE id = ?", (node_id,)).fetchone()
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
                SELECT r.id, r.relation_type, n.name AS target_name, n.entity_type AS target_type
                FROM relations r
                JOIN nodes n ON n.id = r.target_node_id
                WHERE r.source_node_id = ?
                ORDER BY r.id DESC
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
                SELECT r.id, r.relation_type, n.name AS source_name, n.entity_type AS source_type
                FROM relations r
                JOIN nodes n ON n.id = r.source_node_id
                WHERE r.target_node_id = ?
                ORDER BY r.id DESC
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
