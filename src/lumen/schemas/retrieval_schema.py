from __future__ import annotations

from typing import Any


RETRIEVAL_RESULT_SCHEMA_VERSION = "1"


class RetrievalResultSchema:
    """Validation helpers for versioned retrieval result payloads."""

    @staticmethod
    def build_search_payload(
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        query: str,
        matches: list[dict[str, Any]],
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_type": "archive_search_result",
            "schema_version": RETRIEVAL_RESULT_SCHEMA_VERSION,
            "repo_root": repo_root,
            "session_id": session_id,
            "tool_id": tool_id,
            "capability": capability,
            "query": query,
            "status_filter": status,
            "date_from": date_from,
            "date_to": date_to,
            "record_count": len(matches),
            "matches": matches,
        }

    @staticmethod
    def build_latest_payload(
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        status: str | None,
        record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_type": "archive_latest_result",
            "schema_version": RETRIEVAL_RESULT_SCHEMA_VERSION,
            "repo_root": repo_root,
            "session_id": session_id,
            "tool_id": tool_id,
            "capability": capability,
            "status_filter": status,
            "found": record is not None,
            "record": record,
        }

    @staticmethod
    def build_summary_payload(
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        date_from: str | None,
        date_to: str | None,
        record_count: int,
        status_counts: dict[str, int],
        tool_counts: dict[str, int],
        capability_counts: dict[str, int],
        target_label_counts: dict[str, int],
        result_quality_counts: dict[str, int],
        latest_by_capability: dict[str, dict[str, Any]],
        recent_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_type": "archive_summary_result",
            "schema_version": RETRIEVAL_RESULT_SCHEMA_VERSION,
            "repo_root": repo_root,
            "session_id": session_id,
            "tool_id": tool_id,
            "capability": capability,
            "date_from": date_from,
            "date_to": date_to,
            "record_count": record_count,
            "status_counts": status_counts,
            "tool_counts": tool_counts,
            "capability_counts": capability_counts,
            "target_label_counts": target_label_counts,
            "result_quality_counts": result_quality_counts,
            "latest_by_capability": latest_by_capability,
            "recent_records": recent_records,
        }

    @staticmethod
    def build_compare_payload(
        *,
        repo_root: str,
        session_id: str | None,
        tool_id: str | None,
        capability: str,
        date_from: str | None,
        date_to: str | None,
        record_count: int,
        target_count: int,
        target_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_type": "archive_compare_result",
            "schema_version": RETRIEVAL_RESULT_SCHEMA_VERSION,
            "repo_root": repo_root,
            "session_id": session_id,
            "tool_id": tool_id,
            "capability": capability,
            "date_from": date_from,
            "date_to": date_to,
            "record_count": record_count,
            "target_count": target_count,
            "target_groups": target_groups,
        }
