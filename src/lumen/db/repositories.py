from __future__ import annotations

from datetime import UTC, datetime
import json
import sqlite3

from lumen.db.database_manager import DatabaseManager


def _json_dumps(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_loads(value: object) -> object:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _decode_row(
    row: sqlite3.Row | None,
    *,
    json_fields: tuple[str, ...] = (),
) -> dict[str, object] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in json_fields:
        if field in payload:
            payload[field] = _json_loads(payload.get(field))
    return payload


class ProjectRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    @staticmethod
    def normalize_name(name: str) -> str:
        return " ".join(str(name or "").strip().lower().split())

    def get(self, project_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _decode_row(row, json_fields=("tags_json",))

    def get_by_name(self, name: str) -> dict[str, object] | None:
        normalized_name = self.normalize_name(name)
        with self.database_manager.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE normalized_name = ?",
                (normalized_name,),
            ).fetchone()
        return _decode_row(row, json_fields=("tags_json",))

    def list_projects(self) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC, normalized_name ASC"
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json",)) or {} for row in rows]

    def list_active(self) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM projects
                WHERE status = 'active'
                ORDER BY updated_at DESC, normalized_name ASC
                """
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json",)) or {} for row in rows]

    def resolve_candidates(self, text: str | None, *, limit: int = 5) -> list[dict[str, object]]:
        normalized = self.normalize_name(text or "")
        if not normalized:
            return []
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            return []
        clauses = " OR ".join("normalized_name LIKE ?" for _ in tokens)
        params = tuple(f"%{token}%" for token in tokens)
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM projects
                WHERE {clauses}
                ORDER BY
                    CASE WHEN normalized_name = ? THEN 0 ELSE 1 END,
                    updated_at DESC,
                    normalized_name ASC
                LIMIT ?
                """,
                (*params, normalized, limit),
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json",)) or {} for row in rows]

    def upsert(
        self,
        *,
        project_id: str,
        name: str,
        description: str | None = None,
        status: str = "active",
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        normalized_name = self.normalize_name(name)
        timestamp = datetime.now(UTC).isoformat()
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, name, normalized_name, description, status, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    normalized_name = excluded.normalized_name,
                    description = excluded.description,
                    status = excluded.status,
                    tags_json = excluded.tags_json,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    name,
                    normalized_name,
                    description,
                    status,
                    _json_dumps(tags or []),
                    timestamp,
                    timestamp,
                ),
            )
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _decode_row(row, json_fields=("tags_json",)) or {}

    def ensure_general_project(self) -> dict[str, object]:
        existing = self.get("general")
        if existing is not None:
            return existing
        return self.upsert(
            project_id="general",
            name="general",
            description="Default local project for sessions without a more specific assignment.",
            status="active",
            tags=["default"],
        )


class SessionRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get(self, session_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",))

    def get_with_summary(self, session_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    s.*,
                    ss.summary_text,
                    ss.confidence_tier AS summary_confidence_tier,
                    ss.created_at AS summary_created_at
                FROM sessions s
                LEFT JOIN session_summaries ss ON ss.id = s.summary_id
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
        return _decode_row(row, json_fields=("metadata_json",))

    def list_recent(
        self,
        *,
        limit: int = 10,
        include_archived: bool = False,
        archived_only: bool = False,
        project_id: str | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        if archived_only:
            clauses.append("s.status = 'archived'")
        elif not include_archived:
            clauses.append("s.status != 'archived'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    s.*,
                    ss.summary_text,
                    m.content AS latest_message_content,
                    m.intent_domain AS latest_intent_domain,
                    m.confidence_tier AS latest_confidence_tier,
                    m.created_at AS latest_message_at,
                    um.content AS latest_user_message_content,
                    um.created_at AS latest_user_message_at
                FROM sessions s
                LEFT JOIN session_summaries ss ON ss.id = s.summary_id
                LEFT JOIN messages m ON m.id = (
                    SELECT id FROM messages
                    WHERE session_id = s.id AND role = 'assistant'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
                LEFT JOIN messages um ON um.id = (
                    SELECT id FROM messages
                    WHERE session_id = s.id AND role = 'user'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
                {where}
                ORDER BY s.updated_at DESC, s.id DESC
                LIMIT ?
                """,
                (*params, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def list_by_project(self, project_id: str, *, limit: int | None = None) -> list[dict[str, object]]:
        query = """
            SELECT * FROM sessions
            WHERE project_id = ?
            ORDER BY updated_at DESC, id DESC
        """
        params: tuple[object, ...] = (project_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (project_id, max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def upsert(
        self,
        *,
        session_id: str,
        project_id: str | None,
        title: str | None,
        mode: str | None,
        started_at: str,
        updated_at: str,
        status: str,
        summary_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, project_id, title, mode, started_at, updated_at, status, summary_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_id = COALESCE(excluded.project_id, sessions.project_id),
                    title = COALESCE(excluded.title, sessions.title),
                    mode = COALESCE(excluded.mode, sessions.mode),
                    started_at = COALESCE(sessions.started_at, excluded.started_at),
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    summary_id = COALESCE(excluded.summary_id, sessions.summary_id),
                    metadata_json = COALESCE(excluded.metadata_json, sessions.metadata_json)
                """,
                (
                    session_id,
                    project_id,
                    title,
                    mode,
                    started_at,
                    updated_at,
                    status,
                    summary_id,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",)) or {}

    def update_summary(self, *, session_id: str, summary_id: str | None, updated_at: str) -> None:
        with self.database_manager.transaction() as conn:
            conn.execute(
                "UPDATE sessions SET summary_id = ?, updated_at = ? WHERE id = ?",
                (summary_id, updated_at, session_id),
            )

    def delete(self, session_id: str) -> None:
        with self.database_manager.transaction() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


class MessageRepository:
    MESSAGE_JSON_FIELDS = ("route_decision_json", "message_metadata_json")

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        message_id: str,
        session_id: str,
        turn_key: str,
        role: str,
        content: str,
        created_at: str,
        intent_domain: str | None = None,
        confidence_tier: str | None = None,
        response_depth: str | None = None,
        conversation_phase: str | None = None,
        tool_usage_intent: str | None = None,
        route_decision: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, turn_key, role, content, created_at, intent_domain, confidence_tier,
                    response_depth, conversation_phase, tool_usage_intent, route_decision_json,
                    message_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    turn_key = excluded.turn_key,
                    role = excluded.role,
                    content = excluded.content,
                    created_at = excluded.created_at,
                    intent_domain = excluded.intent_domain,
                    confidence_tier = excluded.confidence_tier,
                    response_depth = excluded.response_depth,
                    conversation_phase = excluded.conversation_phase,
                    tool_usage_intent = excluded.tool_usage_intent,
                    route_decision_json = excluded.route_decision_json,
                    message_metadata_json = excluded.message_metadata_json
                """,
                (
                    message_id,
                    session_id,
                    turn_key,
                    role,
                    content,
                    created_at,
                    intent_domain,
                    confidence_tier,
                    response_depth,
                    conversation_phase,
                    tool_usage_intent,
                    _json_dumps(route_decision),
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {}

    def get(self, message_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS)

    def list_by_session(self, session_id: str) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY
                    created_at ASC,
                    CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 ELSE 2 END ASC,
                    id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]

    def list_recent_by_session(self, session_id: str, *, limit: int = 10) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND role = 'assistant'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]

    def list_message_window_by_session(
        self,
        session_id: str,
        *,
        limit: int = 6,
    ) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY
                    created_at DESC,
                    CASE role WHEN 'assistant' THEN 0 WHEN 'user' THEN 1 ELSE 2 END ASC,
                    id DESC
                LIMIT ?
                """,
                (session_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]

    def list_recent_by_project(self, project_id: str, *, limit: int = 10) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.project_id = ? AND m.role = 'assistant'
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT ?
                """,
                (project_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]

    def list_message_window_by_project(
        self,
        project_id: str,
        *,
        limit: int = 6,
    ) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.project_id = ?
                ORDER BY
                    m.created_at DESC,
                    CASE m.role WHEN 'assistant' THEN 0 WHEN 'user' THEN 1 ELSE 2 END ASC,
                    m.id DESC
                LIMIT ?
                """,
                (project_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]

    def list_interaction_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses = ["m.role = 'assistant'"]
        params: list[object] = []
        if session_id is not None:
            clauses.append("m.session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}"
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.*, s.project_id
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                {where}
                ORDER BY m.created_at DESC, m.id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        records: list[dict[str, object]] = []
        for row in rows:
            payload = _decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {}
            metadata = payload.get("message_metadata_json")
            if isinstance(metadata, dict) and isinstance(metadata.get("interaction_record"), dict):
                record = dict(metadata["interaction_record"])
                if not record.get("session_id"):
                    record["session_id"] = payload.get("session_id")
                records.append(record)
                continue
            record = {
                "session_id": payload.get("session_id"),
                "prompt": metadata.get("original_prompt") if isinstance(metadata, dict) else None,
                "summary": payload.get("content"),
                "created_at": payload.get("created_at"),
                "intent_domain": payload.get("intent_domain"),
                "confidence_posture": payload.get("confidence_tier"),
                "response_depth": payload.get("response_depth"),
                "conversation_phase": payload.get("conversation_phase"),
                "tool_route_origin": payload.get("tool_usage_intent"),
                "route": payload.get("route_decision_json"),
                "mode": metadata.get("mode") if isinstance(metadata, dict) else None,
                "kind": metadata.get("kind") if isinstance(metadata, dict) else None,
                "interaction_path": metadata.get("interaction_path") if isinstance(metadata, dict) else None,
                "project_id": payload.get("project_id"),
            }
            records.append(record)
        return records

    def search_interaction_records(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        needle = f"%{str(query or '').strip().lower()}%"
        clauses: list[str] = ["m.role = 'assistant'"]
        params: list[object] = []
        if session_id is not None:
            clauses.append("m.session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("s.project_id = ?")
            params.append(project_id)
        clauses.append(
            """
            (
                LOWER(m.content) LIKE ?
                OR LOWER(COALESCE(m.intent_domain, '')) LIKE ?
                OR LOWER(COALESCE(m.message_metadata_json, '')) LIKE ?
            )
            """
        )
        params.extend([needle, needle, needle, max(int(limit), 1)])
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.*, s.project_id
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE {' AND '.join(clauses)}
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=self.MESSAGE_JSON_FIELDS) or {} for row in rows]


class SessionSummaryRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        summary_id: str,
        session_id: str,
        summary_text: str,
        created_at: str,
        confidence_tier: str | None = None,
        tags: list[str] | None = None,
        summary_scope: str | None = None,
        source_message_start_id: str | None = None,
        source_message_end_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO session_summaries (
                    id, session_id, summary_text, created_at, confidence_tier, tags_json,
                    summary_scope, source_message_start_id, source_message_end_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    summary_text = excluded.summary_text,
                    created_at = excluded.created_at,
                    confidence_tier = excluded.confidence_tier,
                    tags_json = excluded.tags_json,
                    summary_scope = excluded.summary_scope,
                    source_message_start_id = excluded.source_message_start_id,
                    source_message_end_id = excluded.source_message_end_id,
                    metadata_json = excluded.metadata_json
                """,
                (
                    summary_id,
                    session_id,
                    summary_text,
                    created_at,
                    confidence_tier,
                    _json_dumps(tags or []),
                    summary_scope,
                    source_message_start_id,
                    source_message_end_id,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM session_summaries WHERE id = ?", (summary_id,)).fetchone()
        return _decode_row(row, json_fields=("tags_json", "metadata_json")) or {}

    def list_by_session(self, session_id: str) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM session_summaries
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (session_id,),
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json", "metadata_json")) or {} for row in rows]

    def latest_by_session(self, session_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM session_summaries
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return _decode_row(row, json_fields=("tags_json", "metadata_json"))

    def list_recent_by_project(self, project_id: str, *, limit: int = 5) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT ss.*
                FROM session_summaries ss
                JOIN sessions s ON s.id = ss.session_id
                WHERE s.project_id = ?
                ORDER BY ss.created_at DESC, ss.id DESC
                LIMIT ?
                """,
                (project_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json", "metadata_json")) or {} for row in rows]

    def list_recent_by_session(self, session_id: str, *, limit: int = 5) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM session_summaries
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=("tags_json", "metadata_json")) or {} for row in rows]


class MemoryRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get(self, memory_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",))

    def update_status(self, memory_id: str, *, status: str) -> dict[str, object] | None:
        with self.database_manager.transaction() as conn:
            conn.execute(
                "UPDATE memory_items SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, memory_id),
            )
            row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",))

    def update_status_by_identity(self, identity: str, *, status: str) -> dict[str, object] | None:
        with self.database_manager.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM memory_items WHERE id = ? OR source_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
                (identity, identity),
            ).fetchone()
            if row is None:
                return None
            memory_id = str(row["id"])
            conn.execute(
                "UPDATE memory_items SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, memory_id),
            )
            updated = conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        return _decode_row(updated, json_fields=("metadata_json",))

    def delete(self, memory_id: str) -> bool:
        with self.database_manager.transaction() as conn:
            cursor = conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
        return int(cursor.rowcount or 0) > 0

    def delete_by_identity(self, identity: str) -> bool:
        with self.database_manager.transaction() as conn:
            cursor = conn.execute("DELETE FROM memory_items WHERE id = ? OR source_id = ?", (identity, identity))
        return int(cursor.rowcount or 0) > 0

    def upsert(
        self,
        *,
        memory_id: str,
        source_type: str,
        source_id: str,
        project_id: str | None,
        session_id: str | None,
        category: str,
        domain: str | None,
        content: str,
        confidence_tier: str | None,
        created_at: str,
        updated_at: str,
        recency_weight: float | None = None,
        relevance_hint: str | None = None,
        status: str | None = None,
        source_summary: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, source_type, source_id, project_id, session_id, category, domain, content,
                    confidence_tier, created_at, updated_at, recency_weight, relevance_hint, status,
                    source_summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_id = excluded.source_id,
                    project_id = excluded.project_id,
                    session_id = excluded.session_id,
                    category = excluded.category,
                    domain = excluded.domain,
                    content = excluded.content,
                    confidence_tier = excluded.confidence_tier,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    recency_weight = excluded.recency_weight,
                    relevance_hint = excluded.relevance_hint,
                    status = excluded.status,
                    source_summary = excluded.source_summary,
                    metadata_json = excluded.metadata_json
                """,
                (
                    memory_id,
                    source_type,
                    source_id,
                    project_id,
                    session_id,
                    category,
                    domain,
                    content,
                    confidence_tier,
                    created_at,
                    updated_at,
                    recency_weight,
                    relevance_hint,
                    status,
                    source_summary,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",)) or {}

    def list_recent(
        self,
        *,
        limit: int = 20,
        source_type: str | None = None,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> list[dict[str, object]]:
        return self.list_by_filters(
            project_id=project_id,
            source_type=source_type,
            include_archived=include_archived,
            limit=limit,
        )

    def list_by_session(
        self,
        session_id: str,
        *,
        source_type: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, object]]:
        return self.list_by_filters(
            session_id=session_id,
            source_type=source_type,
            include_archived=include_archived,
        )

    def list_by_filters(
        self,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        domain: str | None = None,
        category: str | None = None,
        source_type: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        if archived_only:
            clauses.append("status = 'archived'")
        elif not include_archived:
            clauses.append("(status IS NULL OR status != 'archived')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_items
                {where}
                ORDER BY updated_at DESC, id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]


class MemoryItemEmbeddingRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def get(self, memory_item_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_item_embeddings WHERE memory_item_id = ?",
                (memory_item_id,),
            ).fetchone()
        return _decode_row(row)

    def get_by_source(self, *, source_type: str, source_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM memory_item_embeddings
                WHERE source_type = ? AND source_id = ?
                """,
                (source_type, source_id),
            ).fetchone()
        return _decode_row(row)

    def upsert(
        self,
        *,
        memory_item_id: str,
        source_id: str,
        source_type: str,
        model_name: str,
        embedding_dim: int | None,
        embedding_blob: bytes | None,
        content_hash: str,
        status: str,
        created_at: str,
        updated_at: str,
        error_message: str | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO memory_item_embeddings (
                    memory_item_id, source_id, source_type, model_name, embedding_dim, embedding_blob,
                    content_hash, status, created_at, updated_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_item_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    source_type = excluded.source_type,
                    model_name = excluded.model_name,
                    embedding_dim = excluded.embedding_dim,
                    embedding_blob = excluded.embedding_blob,
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    error_message = excluded.error_message
                """,
                (
                    memory_item_id,
                    source_id,
                    source_type,
                    model_name,
                    embedding_dim,
                    embedding_blob,
                    content_hash,
                    status,
                    created_at,
                    updated_at,
                    error_message,
                ),
            )
            row = conn.execute(
                "SELECT * FROM memory_item_embeddings WHERE memory_item_id = ?",
                (memory_item_id,),
            ).fetchone()
        return _decode_row(row) or {}

    def list_by_status(self, status: str) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_item_embeddings
                WHERE status = ?
                ORDER BY updated_at DESC, memory_item_id DESC
                """,
                (status,),
            ).fetchall()
        return [_decode_row(row) or {} for row in rows]

    def list_missing(
        self,
        *,
        limit: int | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if source_type is not None:
            clauses.append("m.source_type = ?")
            params.append(source_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.*
                FROM memory_items m
                LEFT JOIN memory_item_embeddings e ON e.memory_item_id = m.id
                {where}
                AND e.memory_item_id IS NULL
                ORDER BY m.updated_at DESC, m.id DESC
                {limit_clause}
                """
                if where
                else f"""
                SELECT m.*
                FROM memory_items m
                LEFT JOIN memory_item_embeddings e ON e.memory_item_id = m.id
                WHERE e.memory_item_id IS NULL
                ORDER BY m.updated_at DESC, m.id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def list_stale(
        self,
        *,
        content_hashes: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*, m.content
                FROM memory_item_embeddings e
                JOIN memory_items m ON m.id = e.memory_item_id
                ORDER BY e.updated_at DESC, e.memory_item_id DESC
                """
            ).fetchall()
        decoded = [_decode_row(row) or {} for row in rows]
        stale: list[dict[str, object]] = []
        content_hashes = dict(content_hashes or {})
        for row in decoded:
            memory_item_id = str(row.get("memory_item_id") or "")
            expected_hash = content_hashes.get(memory_item_id)
            if expected_hash and expected_hash != str(row.get("content_hash") or ""):
                stale.append(row)
            elif str(row.get("status") or "") in {"pending", "failed"}:
                stale.append(row)
        if limit is not None:
            return stale[: max(int(limit), 1)]
        return stale


class ToolRunRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        tool_run_id: str,
        session_id: str,
        message_id: str | None,
        project_id: str | None,
        tool_name: str,
        capability: str | None,
        input_summary: str | None,
        output_summary: str | None,
        success: bool,
        created_at: str,
        confidence_impact: str | None = None,
        latency_ms: int | None = None,
        tool_bundle: str | None = None,
        archive_path: str | None = None,
        run_dir: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO tool_runs (
                    id, session_id, message_id, project_id, tool_name, capability, input_summary,
                    output_summary, success, created_at, confidence_impact, latency_ms, tool_bundle,
                    archive_path, run_dir, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    project_id = excluded.project_id,
                    tool_name = excluded.tool_name,
                    capability = excluded.capability,
                    input_summary = excluded.input_summary,
                    output_summary = excluded.output_summary,
                    success = excluded.success,
                    created_at = excluded.created_at,
                    confidence_impact = excluded.confidence_impact,
                    latency_ms = excluded.latency_ms,
                    tool_bundle = excluded.tool_bundle,
                    archive_path = excluded.archive_path,
                    run_dir = excluded.run_dir,
                    metadata_json = excluded.metadata_json
                """,
                (
                    tool_run_id,
                    session_id,
                    message_id,
                    project_id,
                    tool_name,
                    capability,
                    input_summary,
                    output_summary,
                    1 if success else 0,
                    created_at,
                    confidence_impact,
                    latency_ms,
                    tool_bundle,
                    archive_path,
                    run_dir,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM tool_runs WHERE id = ?", (tool_run_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",)) or {}

    def get(self, tool_run_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM tool_runs WHERE id = ?", (tool_run_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",))

    def list_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_name: str | None = None,
        capability: str | None = None,
        success: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if capability is not None:
            clauses.append("capability = ?")
            params.append(capability)
        if success is not None:
            clauses.append("success = ?")
            params.append(1 if success else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tool_runs
                {where}
                ORDER BY created_at DESC, id DESC
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def search_records(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_name: str | None = None,
        capability: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        needle = f"%{str(query or '').strip().lower()}%"
        clauses: list[str] = [
            """
            (
                LOWER(COALESCE(tool_name, '')) LIKE ?
                OR LOWER(COALESCE(capability, '')) LIKE ?
                OR LOWER(COALESCE(output_summary, '')) LIKE ?
                OR LOWER(COALESCE(metadata_json, '')) LIKE ?
            )
            """
        ]
        params: list[object] = [needle, needle, needle, needle]
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if capability is not None:
            clauses.append("capability = ?")
            params.append(capability)
        if success is not None:
            clauses.append("success = ?")
            params.append(1 if success else 0)
        params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tool_runs
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def latest(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_name: str | None = None,
        capability: str | None = None,
        success: bool | None = None,
    ) -> dict[str, object] | None:
        records = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_name=tool_name,
            capability=capability,
            success=success,
            limit=1,
        )
        return records[0] if records else None

    def list_by_project(self, project_id: str, *, limit: int | None = None) -> list[dict[str, object]]:
        return self.list_records(project_id=project_id, limit=limit)

    def summary_by_filters(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_name: str | None = None,
        capability: str | None = None,
        success: bool | None = None,
    ) -> dict[str, object]:
        rows = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_name=tool_name,
            capability=capability,
            success=success,
        )
        status_counts: dict[str, int] = {}
        capability_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        for row in rows:
            metadata = row.get("metadata_json")
            status = str(metadata.get("status") if isinstance(metadata, dict) else row.get("success")).strip() or (
                "ok" if bool(row.get("success")) else "error"
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            tool_key = str(row.get("tool_name") or "unknown")
            cap_key = str(row.get("capability") or "unknown")
            tool_counts[tool_key] = tool_counts.get(tool_key, 0) + 1
            capability_counts[cap_key] = capability_counts.get(cap_key, 0) + 1
        return {
            "record_count": len(rows),
            "status_counts": status_counts,
            "tool_counts": tool_counts,
            "capability_counts": capability_counts,
            "records": rows,
        }


class TrainabilityTraceRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        trace_id: str,
        session_id: str,
        message_id: str | None,
        decision_type: str,
        input_context_summary: str | None,
        chosen_action: str | None,
        outcome: str | None,
        label: str | None,
        confidence_tier: str | None,
        created_at: str,
        model_assist_used: bool | None = None,
        evaluation_score: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO trainability_traces (
                    id, session_id, message_id, decision_type, input_context_summary, chosen_action,
                    outcome, label, confidence_tier, created_at, model_assist_used, evaluation_score,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    decision_type = excluded.decision_type,
                    input_context_summary = excluded.input_context_summary,
                    chosen_action = excluded.chosen_action,
                    outcome = excluded.outcome,
                    label = excluded.label,
                    confidence_tier = excluded.confidence_tier,
                    created_at = excluded.created_at,
                    model_assist_used = excluded.model_assist_used,
                    evaluation_score = excluded.evaluation_score,
                    metadata_json = excluded.metadata_json
                """,
                (
                    trace_id,
                    session_id,
                    message_id,
                    decision_type,
                    input_context_summary,
                    chosen_action,
                    outcome,
                    label,
                    confidence_tier,
                    created_at,
                    None if model_assist_used is None else (1 if model_assist_used else 0),
                    evaluation_score,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM trainability_traces WHERE id = ?", (trace_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",)) or {}

    def list_by_session(self, session_id: str) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trainability_traces
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (session_id,),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def list_recent_by_project(self, project_id: str, *, limit: int = 20) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*
                FROM trainability_traces t
                JOIN sessions s ON s.id = t.session_id
                WHERE s.project_id = ?
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT ?
                """,
                (project_id, max(int(limit), 1)),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]

    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        decision_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        needle = f"%{str(query or '').strip().lower()}%"
        clauses = [
            """
            (
                LOWER(COALESCE(decision_type, '')) LIKE ?
                OR LOWER(COALESCE(input_context_summary, '')) LIKE ?
                OR LOWER(COALESCE(chosen_action, '')) LIKE ?
                OR LOWER(COALESCE(outcome, '')) LIKE ?
                OR LOWER(COALESCE(metadata_json, '')) LIKE ?
            )
            """
        ]
        params: list[object] = [needle, needle, needle, needle, needle]
        join = ""
        if project_id is not None:
            join = "JOIN sessions s ON s.id = t.session_id"
            clauses.append("s.project_id = ?")
            params.append(project_id)
        if decision_type is not None:
            clauses.append("t.decision_type = ?")
            params.append(decision_type)
        params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*
                FROM trainability_traces t
                {join}
                WHERE {' AND '.join(clauses)}
                ORDER BY t.created_at DESC, t.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=("metadata_json",)) or {} for row in rows]


class DatasetImportRunRepository:
    JSON_FIELDS = ("notes_json",)

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        import_run_id: str,
        dataset_name: str,
        dataset_version: str | None,
        source_path: str | None,
        source_format: str,
        dataset_kind: str,
        import_strategy: str,
        ingestion_status: str,
        example_count: int,
        train_count: int = 0,
        validation_count: int = 0,
        test_count: int = 0,
        schema_version: str,
        created_at: str,
        completed_at: str | None = None,
        notes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dataset_import_runs (
                    id, dataset_name, dataset_version, source_path, source_format, dataset_kind,
                    import_strategy, ingestion_status, example_count, train_count, validation_count,
                    test_count, schema_version, notes_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    dataset_name = excluded.dataset_name,
                    dataset_version = excluded.dataset_version,
                    source_path = excluded.source_path,
                    source_format = excluded.source_format,
                    dataset_kind = excluded.dataset_kind,
                    import_strategy = excluded.import_strategy,
                    ingestion_status = excluded.ingestion_status,
                    example_count = excluded.example_count,
                    train_count = excluded.train_count,
                    validation_count = excluded.validation_count,
                    test_count = excluded.test_count,
                    schema_version = excluded.schema_version,
                    notes_json = excluded.notes_json,
                    created_at = excluded.created_at,
                    completed_at = excluded.completed_at
                """,
                (
                    import_run_id,
                    dataset_name,
                    dataset_version,
                    source_path,
                    source_format,
                    dataset_kind,
                    import_strategy,
                    ingestion_status,
                    int(example_count),
                    int(train_count),
                    int(validation_count),
                    int(test_count),
                    schema_version,
                    _json_dumps(notes or {}),
                    created_at,
                    completed_at,
                ),
            )
            row = conn.execute("SELECT * FROM dataset_import_runs WHERE id = ?", (import_run_id,)).fetchone()
        return _decode_row(row, json_fields=self.JSON_FIELDS) or {}

    def get(self, import_run_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM dataset_import_runs WHERE id = ?", (import_run_id,)).fetchone()
        return _decode_row(row, json_fields=self.JSON_FIELDS)

    def list_runs(self, *, dataset_name: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if dataset_name is not None:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM dataset_import_runs
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=self.JSON_FIELDS) or {} for row in rows]


class DatasetExampleRepository:
    JSON_FIELDS = ("provenance_json", "metadata_json")

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        example_id: str,
        import_run_id: str,
        example_type: str,
        source_format: str,
        split_assignment: str,
        ingestion_state: str,
        input_text: str,
        target_text: str | None,
        label_category: str | None,
        label_value: str | None,
        explanation_text: str | None,
        source_session_id: str | None,
        source_message_id: str | None,
        source_interaction_path: str | None,
        source_trace_id: str | None,
        source_tool_run_id: str | None,
        label_source: str,
        trainable: bool,
        provenance: dict[str, object],
        metadata: dict[str, object] | None,
        created_at: str,
        updated_at: str,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dataset_examples (
                    id, import_run_id, example_type, source_format, split_assignment, ingestion_state,
                    input_text, target_text, label_category, label_value, explanation_text,
                    source_session_id, source_message_id, source_interaction_path, source_trace_id,
                    source_tool_run_id, label_source, trainable, provenance_json, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    import_run_id = excluded.import_run_id,
                    example_type = excluded.example_type,
                    source_format = excluded.source_format,
                    split_assignment = excluded.split_assignment,
                    ingestion_state = excluded.ingestion_state,
                    input_text = excluded.input_text,
                    target_text = excluded.target_text,
                    label_category = excluded.label_category,
                    label_value = excluded.label_value,
                    explanation_text = excluded.explanation_text,
                    source_session_id = excluded.source_session_id,
                    source_message_id = excluded.source_message_id,
                    source_interaction_path = excluded.source_interaction_path,
                    source_trace_id = excluded.source_trace_id,
                    source_tool_run_id = excluded.source_tool_run_id,
                    label_source = excluded.label_source,
                    trainable = excluded.trainable,
                    provenance_json = excluded.provenance_json,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    example_id,
                    import_run_id,
                    example_type,
                    source_format,
                    split_assignment,
                    ingestion_state,
                    input_text,
                    target_text,
                    label_category,
                    label_value,
                    explanation_text,
                    source_session_id,
                    source_message_id,
                    source_interaction_path,
                    source_trace_id,
                    source_tool_run_id,
                    label_source,
                    1 if trainable else 0,
                    _json_dumps(provenance),
                    _json_dumps(metadata or {}),
                    created_at,
                    updated_at,
                ),
            )
            row = conn.execute("SELECT * FROM dataset_examples WHERE id = ?", (example_id,)).fetchone()
        return self._normalize_row(_decode_row(row, json_fields=self.JSON_FIELDS) or {})

    def get(self, example_id: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute("SELECT * FROM dataset_examples WHERE id = ?", (example_id,)).fetchone()
        return self._normalize_row(_decode_row(row, json_fields=self.JSON_FIELDS))

    def list_examples(
        self,
        *,
        dataset_name: str | None = None,
        import_run_id: str | None = None,
        example_type: str | None = None,
        split_assignment: str | None = None,
        ingestion_state: str | None = None,
        label_source: str | None = None,
        trainable: bool | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        join = "JOIN dataset_import_runs r ON r.id = e.import_run_id"
        if dataset_name is not None:
            clauses.append("r.dataset_name = ?")
            params.append(dataset_name)
        if import_run_id is not None:
            clauses.append("e.import_run_id = ?")
            params.append(import_run_id)
        if example_type is not None:
            clauses.append("e.example_type = ?")
            params.append(example_type)
        if split_assignment is not None:
            clauses.append("e.split_assignment = ?")
            params.append(split_assignment)
        if ingestion_state is not None:
            clauses.append("e.ingestion_state = ?")
            params.append(ingestion_state)
        if label_source is not None:
            clauses.append("e.label_source = ?")
            params.append(label_source)
        if trainable is not None:
            clauses.append("e.trainable = ?")
            params.append(1 if trainable else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    e.*,
                    r.dataset_name,
                    r.dataset_version,
                    r.dataset_kind,
                    r.import_strategy
                FROM dataset_examples e
                {join}
                {where}
                ORDER BY e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._normalize_row(_decode_row(row, json_fields=self.JSON_FIELDS) or {}) for row in rows]

    def update_example(
        self,
        *,
        example_id: str,
        trainable: bool | None = None,
        ingestion_state: str | None = None,
        split_assignment: str | None = None,
        label_source: str | None = None,
        metadata_patch: dict[str, object] | None = None,
        updated_at: str,
    ) -> dict[str, object] | None:
        current = self.get(example_id)
        if current is None:
            return None
        metadata = dict(current.get("metadata_json") or {})
        if metadata_patch:
            metadata.update(metadata_patch)
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE dataset_examples
                SET
                    trainable = ?,
                    ingestion_state = ?,
                    split_assignment = ?,
                    label_source = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if (trainable if trainable is not None else bool(current.get("trainable"))) else 0,
                    str(ingestion_state or current.get("ingestion_state") or "ready"),
                    str(split_assignment or current.get("split_assignment") or "train"),
                    str(label_source or current.get("label_source") or "imported_external"),
                    _json_dumps(metadata),
                    updated_at,
                    example_id,
                ),
            )
            row = conn.execute("SELECT * FROM dataset_examples WHERE id = ?", (example_id,)).fetchone()
        return self._normalize_row(_decode_row(row, json_fields=self.JSON_FIELDS))

    @staticmethod
    def _normalize_row(row: dict[str, object] | None) -> dict[str, object] | None:
        if row is None:
            return None
        normalized = dict(row)
        if "trainable" in normalized:
            normalized["trainable"] = bool(normalized.get("trainable"))
        return normalized


class DatasetExampleLabelRepository:
    JSON_FIELDS = ("metadata_json",)

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def clear_canonical_for_role(self, *, dataset_example_id: str, label_role: str) -> None:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE dataset_example_labels
                SET is_canonical = 0
                WHERE dataset_example_id = ? AND label_role = ?
                """,
                (dataset_example_id, label_role),
            )

    def upsert(
        self,
        *,
        label_id: str,
        dataset_example_id: str,
        label_role: str,
        label_value: str,
        label_category: str | None,
        is_canonical: bool,
        reviewer: str | None,
        reason: str | None,
        created_at: str,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dataset_example_labels (
                    id, dataset_example_id, label_role, label_value, label_category, is_canonical,
                    reviewer, reason, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    dataset_example_id = excluded.dataset_example_id,
                    label_role = excluded.label_role,
                    label_value = excluded.label_value,
                    label_category = excluded.label_category,
                    is_canonical = excluded.is_canonical,
                    reviewer = excluded.reviewer,
                    reason = excluded.reason,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    label_id,
                    dataset_example_id,
                    label_role,
                    label_value,
                    label_category,
                    1 if is_canonical else 0,
                    reviewer,
                    reason,
                    created_at,
                    _json_dumps(metadata or {}),
                ),
            )
            row = conn.execute("SELECT * FROM dataset_example_labels WHERE id = ?", (label_id,)).fetchone()
        return _decode_row(row, json_fields=self.JSON_FIELDS) or {}

    def list_labels(self, *, dataset_example_id: str | None = None, limit: int = 200) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if dataset_example_id is not None:
            clauses.append("dataset_example_id = ?")
            params.append(dataset_example_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(int(limit), 1))
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM dataset_example_labels
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_decode_row(row, json_fields=self.JSON_FIELDS) or {} for row in rows]

    def canonical_labels_for_examples(self, dataset_example_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        ids = [str(item) for item in dataset_example_ids if str(item).strip()]
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM dataset_example_labels
                WHERE dataset_example_id IN ({placeholders}) AND is_canonical = 1
                ORDER BY created_at DESC, id DESC
                """,
                tuple(ids),
            ).fetchall()
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            decoded = _decode_row(row, json_fields=self.JSON_FIELDS) or {}
            key = str(decoded.get("dataset_example_id") or "")
            grouped.setdefault(key, []).append(decoded)
        return grouped


class BugLogRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def upsert(
        self,
        *,
        bug_log_id: str,
        project_id: str | None,
        session_id: str | None,
        bug_type: str,
        severity: str,
        title: str,
        description: str,
        status: str,
        created_at: str,
        resolved_at: str | None = None,
        source_component: str | None = None,
        taxonomy_label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO bug_logs (
                    id, project_id, session_id, bug_type, severity, title, description, status,
                    created_at, resolved_at, source_component, taxonomy_label, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_id = excluded.project_id,
                    session_id = excluded.session_id,
                    bug_type = excluded.bug_type,
                    severity = excluded.severity,
                    title = excluded.title,
                    description = excluded.description,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    resolved_at = excluded.resolved_at,
                    source_component = excluded.source_component,
                    taxonomy_label = excluded.taxonomy_label,
                    metadata_json = excluded.metadata_json
                """,
                (
                    bug_log_id,
                    project_id,
                    session_id,
                    bug_type,
                    severity,
                    title,
                    description,
                    status,
                    created_at,
                    resolved_at,
                    source_component,
                    taxonomy_label,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM bug_logs WHERE id = ?", (bug_log_id,)).fetchone()
        return _decode_row(row, json_fields=("metadata_json",)) or {}


class PreferenceRepository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    def set(self, *, preference_id: str, key: str, value: str, scope: str) -> dict[str, object]:
        updated_at = datetime.now(UTC).isoformat()
        with self.database_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO preferences (id, key, value, scope, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    key = excluded.key,
                    value = excluded.value,
                    scope = excluded.scope,
                    updated_at = excluded.updated_at
                """,
                (preference_id, key, value, scope, updated_at),
            )
            row = conn.execute("SELECT * FROM preferences WHERE id = ?", (preference_id,)).fetchone()
        return _decode_row(row) or {}

    def get(self, *, key: str, scope: str) -> dict[str, object] | None:
        with self.database_manager.connect() as conn:
            row = conn.execute(
                "SELECT * FROM preferences WHERE key = ? AND scope = ?",
                (key, scope),
            ).fetchone()
        return _decode_row(row)

    def list_by_scope(self, scope: str) -> list[dict[str, object]]:
        with self.database_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM preferences
                WHERE scope = ?
                ORDER BY key ASC
                """,
                (scope,),
            ).fetchall()
        return [_decode_row(row) or {} for row in rows]
