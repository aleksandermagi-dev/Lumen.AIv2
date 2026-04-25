from __future__ import annotations

from lumen.app.context_policy import ContextPolicy
from lumen.app.settings import AppSettings
from lumen.memory.archive_manager import ArchiveManager
from lumen.reporting.output_formatter import OutputFormatter
from lumen.retrieval.context_models import CompactContextMatch


class ArchiveService:
    """Handles archive inspection, retrieval, and summary/report composition."""

    def __init__(
        self,
        archive_manager: ArchiveManager,
        formatter: OutputFormatter,
        repo_root: str,
        settings: AppSettings,
    ):
        self.archive_manager = archive_manager
        self.formatter = formatter
        self.repo_root = repo_root
        self.settings = settings
        self.context_policy = ContextPolicy.from_settings(settings)

    def list_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        records = self.archive_manager.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )
        return self.formatter.archive_records_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            records=records,
            query=None,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

    def inspect_session(self, session_id: str) -> dict[str, object]:
        report = self.archive_manager.inspect_session(session_id)
        return self.formatter.archive_records_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=None,
            capability=None,
            records=report["records"],
            query=None,
            status=None,
            date_from=None,
            date_to=None,
        )

    def search_records(
        self,
        query: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        result = self.archive_manager.search_records(
            query,
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )
        return self.formatter.archive_search_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            query=result.query,
            matches=[
                {
                    "score": match.score,
                    "matched_fields": match.matched_fields,
                    "score_breakdown": match.score_breakdown,
                    "record": match.record,
                }
                for match in result.matches
            ],
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

    def retrieve_context(
        self,
        query: str,
        *,
        limit: int | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
    ) -> dict[str, object]:
        effective_limit = limit if limit is not None else self.settings.context_match_limit
        result = self.archive_manager.search_records(
            query,
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
        )
        top_matches = result.matches[:effective_limit]
        return {
            "query": query,
            "record_count": result.record_count,
            "top_matches": [
                CompactContextMatch(
                    score=match.score,
                    record=self.context_policy.compact_archive_context_record(match.record),
                    score_breakdown=match.score_breakdown,
                ).to_dict()
                | {"matched_fields": match.matched_fields}
                for match in top_matches
            ],
        }

    def latest_record(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = "ok",
    ) -> dict[str, object]:
        record = self.archive_manager.latest_record(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
        )
        return self.formatter.latest_record_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            record=record,
        )

    def summary(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        summary = self.archive_manager.summarize_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
        )
        return self.formatter.archive_summary_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
            record_count=summary["record_count"],
            status_counts=summary["status_counts"],
            tool_counts=summary["tool_counts"],
            capability_counts=summary["capability_counts"],
            target_label_counts=summary["target_label_counts"],
            result_quality_counts=summary["result_quality_counts"],
            latest_by_capability=summary["latest_by_capability"],
            recent_records=summary["recent_records"],
        )

    def compare_runs_by_target(
        self,
        *,
        capability: str,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, object]:
        comparison = self.archive_manager.compare_runs_by_target(
            capability=capability,
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            date_from=date_from,
            date_to=date_to,
        )
        return self.formatter.archive_compare_payload(
            repo_root=self.repo_root,
            session_id=session_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
            record_count=comparison["record_count"],
            target_count=comparison["target_count"],
            target_groups=comparison["target_groups"],
        )
