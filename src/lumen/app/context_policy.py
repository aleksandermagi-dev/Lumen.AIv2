from __future__ import annotations

from dataclasses import dataclass

from lumen.app.settings import AppSettings
from lumen.retrieval.context_models import (
    CompactArchiveContextRecord,
    CompactInteractionContextRecord,
    CompactPromptView,
)


@dataclass(slots=True)
class ContextPolicy:
    context_match_limit: int
    context_prompt_max_length: int
    context_summary_max_length: int
    session_objective_max_length: int
    session_thread_summary_max_length: int

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "ContextPolicy":
        return cls(
            context_match_limit=settings.context_match_limit,
            context_prompt_max_length=settings.context_prompt_max_length,
            context_summary_max_length=settings.context_summary_max_length,
            session_objective_max_length=settings.session_objective_max_length,
            session_thread_summary_max_length=settings.session_thread_summary_max_length,
        )

    @staticmethod
    def truncate_text(value: object, max_length: int) -> str | None:
        text = str(value).strip() if value is not None else ""
        if not text:
            return None
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def compact_archive_context_record(self, record: dict[str, object]) -> CompactArchiveContextRecord:
        return CompactArchiveContextRecord(
            session_id=str(record.get("session_id")) if record.get("session_id") is not None else None,
            tool_id=str(record.get("tool_id")) if record.get("tool_id") is not None else None,
            capability=str(record.get("capability")) if record.get("capability") is not None else None,
            status=str(record.get("status")) if record.get("status") is not None else None,
            run_id=str(record.get("run_id")) if record.get("run_id") is not None else None,
            target_label=str(record.get("target_label")) if record.get("target_label") is not None else None,
            result_quality=str(record.get("result_quality")) if record.get("result_quality") is not None else None,
            summary=self.truncate_text(
                record.get("summary"),
                self.context_summary_max_length,
            ),
            created_at=str(record.get("created_at")) if record.get("created_at") is not None else None,
            archive_path=str(record.get("archive_path")) if record.get("archive_path") is not None else None,
        )

    def compact_interaction_context_record(
        self,
        record: dict[str, object],
    ) -> CompactInteractionContextRecord:
        prompt_view = dict(record.get("prompt_view") or {})
        compact_prompt_view = CompactPromptView(
            canonical_prompt=self.truncate_text(
                prompt_view.get("canonical_prompt"),
                self.context_prompt_max_length,
            ),
            original_prompt=self.truncate_text(
                prompt_view.get("original_prompt"),
                self.context_prompt_max_length,
            ),
            resolved_prompt=self.truncate_text(
                prompt_view.get("resolved_prompt"),
                self.context_prompt_max_length,
            ),
            rewritten=bool(prompt_view.get("rewritten")),
        )
        return CompactInteractionContextRecord(
            session_id=str(record.get("session_id")) if record.get("session_id") is not None else None,
            prompt=self.truncate_text(record.get("prompt"), self.context_prompt_max_length),
            resolved_prompt=self.truncate_text(
                record.get("resolved_prompt"),
                self.context_prompt_max_length,
            ),
            mode=str(record.get("mode")) if record.get("mode") is not None else None,
            kind=str(record.get("kind")) if record.get("kind") is not None else None,
            summary=self.truncate_text(
                record.get("summary"),
                self.context_summary_max_length,
            ),
            created_at=str(record.get("created_at")) if record.get("created_at") is not None else None,
            interaction_path=str(record.get("interaction_path")) if record.get("interaction_path") is not None else None,
            resolution_strategy=str(record.get("resolution_strategy")) if record.get("resolution_strategy") is not None else None,
            resolution_reason=str(record.get("resolution_reason")) if record.get("resolution_reason") is not None else None,
            dominant_intent=str(record.get("dominant_intent")) if record.get("dominant_intent") is not None else None,
            extracted_entities=tuple(
                str(entity.get("value"))
                for entity in (record.get("extracted_entities") or [])
                if isinstance(entity, dict) and entity.get("value") is not None
            ),
            prompt_view=compact_prompt_view,
        )
