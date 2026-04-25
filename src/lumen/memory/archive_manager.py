from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any

from lumen.app.models import ArchivedRunRecord
from lumen.app.settings import AppSettings
from lumen.db.persistence_manager import PersistenceManager
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.research.research_run import ResearchRunSummary
from lumen.retrieval.archive_search import ArchiveSearchMatch, ArchiveSearchResult
from lumen.retrieval.semantic_matcher import SemanticCandidate, SemanticMatcher
from lumen.schemas.archive_schema import ArchiveRecordSchema
from lumen.tools.registry_types import ToolResult


class ArchiveManager:
    """Persists lightweight local records for completed tool runs."""

    INDEX_FILENAME = "_index.json"

    def __init__(self, settings: AppSettings, persistence_manager: PersistenceManager | None = None):
        self.settings = settings
        self.repo_root = settings.repo_root
        self.archive_root = settings.archive_root
        self.prompt_nlu = PromptNLU()
        self.semantic_matcher = SemanticMatcher()
        self.persistence_manager = persistence_manager or PersistenceManager(settings)

    def record_tool_run(self, session_id: str, result: ToolResult) -> ArchivedRunRecord:
        timestamp = datetime.now(UTC)
        run_stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        target_dir = self.archive_root / session_id / result.tool_id / result.capability
        target_dir.mkdir(parents=True, exist_ok=True)

        archive_path = target_dir / f"{run_stamp}.json"
        payload = ArchiveRecordSchema.normalize(
            self._build_payload(session_id=session_id, result=result, timestamp=timestamp)
        )
        ArchiveRecordSchema.validate(payload)
        archive_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_index_entry(payload, archive_path)
        payload["archive_path"] = str(archive_path)
        self.persistence_manager.record_tool_run(session_id=session_id, archive_record=payload)

        return ArchivedRunRecord(
            session_id=session_id,
            tool_id=result.tool_id,
            capability=result.capability,
            status=result.status,
            summary=result.summary,
            run_dir=result.run_dir,
            archive_path=archive_path,
            created_at=timestamp,
        )

    def list_sessions(self) -> list[str]:
        if not self.archive_root.exists():
            return []
        return sorted(item.name for item in self.archive_root.iterdir() if item.is_dir())

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
    ) -> list[dict[str, Any]]:
        root = self.archive_root
        if session_id:
            root = root / session_id
        if tool_id:
            root = root / tool_id
        if capability and tool_id:
            root = root / capability

        try:
            self.persistence_manager.bootstrap()
            db_records = self.persistence_manager.tool_runs.list_records(
                session_id=session_id,
                project_id=project_id,
                tool_name=tool_id,
                capability=capability,
                success=(
                    True if status and status.strip().lower() == "ok"
                    else False if status
                    else None
                ),
            )
            if db_records:
                records = [
                    self._hydrate_db_record(item)
                    for item in db_records
                ]
                return [
                    record
                    for record in records
                    if record
                    and (not status or str(record.get("status", "")).lower() == status.strip().lower())
                    and self._record_in_date_range(record, date_from=date_from, date_to=date_to)
                ]
        except Exception:
            pass

        if not root.exists():
            return []

        self.persistence_manager.record_fallback_read("archive structured metadata fallback")
        records: list[dict[str, Any]] = []
        for archive_file in self._iter_record_paths(root):
            record = self._load_structured_record(archive_file)
            if session_id and record.get("session_id") != session_id:
                continue
            if project_id and str(record.get("project_id") or "").strip() != project_id:
                continue
            if tool_id and record.get("tool_id") != tool_id:
                continue
            if capability and record.get("capability") != capability:
                continue
            if status and str(record.get("status", "")).lower() != status.strip().lower():
                continue
            if not self._record_in_date_range(record, date_from=date_from, date_to=date_to):
                continue
            records.append(record)
        return records

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        records = self.list_records(session_id=session_id)
        return {
            "session_id": session_id,
            "record_count": len(records),
            "records": records,
        }

    def index_status(self, *, session_id: str | None = None) -> dict[str, Any]:
        root = self.archive_root / session_id if session_id else self.archive_root
        if not root.exists():
            return {
                "scope": str(root),
                "record_file_count": 0,
                "indexed_record_count": 0,
                "index_file_count": 0,
                "legacy_record_count": 0,
                "coverage_ratio": 0.0,
            }

        record_paths = {
            path
            for path in root.rglob("*.json")
            if path.name != self.INDEX_FILENAME
        }
        indexed_paths = set(self._indexed_paths(root))
        legacy_record_count = len(record_paths - indexed_paths)
        indexed_record_count = len(record_paths & indexed_paths)
        record_file_count = len(record_paths)
        coverage_ratio = (indexed_record_count / record_file_count) if record_file_count else 0.0
        return {
            "scope": str(root),
            "record_file_count": record_file_count,
            "indexed_record_count": indexed_record_count,
            "index_file_count": len(list(root.rglob(self.INDEX_FILENAME))),
            "legacy_record_count": legacy_record_count,
            "coverage_ratio": round(coverage_ratio, 4),
        }

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
    ) -> ArchiveSearchResult:
        needle = query.strip().lower()
        records = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )
        if not needle:
            return ArchiveSearchResult(
                query=query,
                record_count=len(records),
                matches=[
                    ArchiveSearchMatch(
                        score=0,
                        matched_fields=[],
                        score_breakdown={"keyword_score": 0, "semantic_score": 0},
                        record=record,
                    )
                    for record in records
                ],
            )
        query_understanding = self.prompt_nlu.analyze(query)

        matches: list[ArchiveSearchMatch] = []
        candidate_records = records

        for record in candidate_records:
            match = self._score_record(record, needle, query_understanding=query_understanding)
            if match is not None:
                matches.append(match)

        matches.sort(
            key=lambda item: (
                item.score,
                self._semantic_tiebreak(item.matched_fields),
                item.record.get("created_at", ""),
            ),
            reverse=True,
        )
        if any(int(match.score_breakdown.get("keyword_score", 0)) > 0 for match in matches):
            matches = [
                match
                for match in matches
                if int(match.score_breakdown.get("keyword_score", 0)) > 0
            ]
        return ArchiveSearchResult(
            query=query,
            record_count=len(matches),
            matches=matches,
        )

    def latest_record(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        status: str | None = "ok",
    ) -> dict[str, Any] | None:
        records = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            status=status,
        )
        if not records:
            return None
        return max(records, key=lambda record: record.get("created_at", ""))

    def summarize_records(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        capability: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        records = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
        )

        status_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        capability_counts: dict[str, int] = {}
        target_label_counts: dict[str, int] = {}
        result_quality_counts: dict[str, int] = {}
        latest_by_capability: dict[str, dict[str, Any]] = {}

        for record in records:
            status = str(record.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

            tool = str(record.get("tool_id", "unknown"))
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

            capability_name = str(record.get("capability", "unknown"))
            capability_counts[capability_name] = capability_counts.get(capability_name, 0) + 1

            target_label = str(record.get("target_label") or "").strip()
            if target_label:
                target_label_counts[target_label] = target_label_counts.get(target_label, 0) + 1

            result_quality = str(record.get("result_quality") or "").strip()
            if result_quality:
                result_quality_counts[result_quality] = result_quality_counts.get(result_quality, 0) + 1

            existing = latest_by_capability.get(capability_name)
            if existing is None or str(record.get("created_at", "")) > str(existing.get("created_at", "")):
                latest_by_capability[capability_name] = record

        recent_records = sorted(
            records,
            key=lambda record: record.get("created_at", ""),
            reverse=True,
        )[:5]

        return {
            "session_id": session_id,
            "tool_id": tool_id,
            "capability": capability,
            "date_from": date_from,
            "date_to": date_to,
            "record_count": len(records),
            "status_counts": status_counts,
            "tool_counts": tool_counts,
            "capability_counts": capability_counts,
            "target_label_counts": target_label_counts,
            "result_quality_counts": result_quality_counts,
            "latest_by_capability": latest_by_capability,
            "recent_records": recent_records,
        }

    def compare_runs_by_target(
        self,
        *,
        capability: str,
        session_id: str | None = None,
        project_id: str | None = None,
        tool_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        if not capability.strip():
            raise ValueError("archive comparison requires a capability")

        records = self.list_records(
            session_id=session_id,
            project_id=project_id,
            tool_id=tool_id,
            capability=capability,
            date_from=date_from,
            date_to=date_to,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            target_label = str(record.get("target_label") or "").strip() or "<unlabeled>"
            grouped.setdefault(target_label, []).append(record)

        target_groups: list[dict[str, Any]] = []
        for target_label, target_records in grouped.items():
            ordered_records = sorted(
                target_records,
                key=lambda record: record.get("created_at", ""),
                reverse=True,
            )
            result_quality_counts: dict[str, int] = {}
            for record in ordered_records:
                result_quality = str(record.get("result_quality") or "unknown")
                result_quality_counts[result_quality] = result_quality_counts.get(result_quality, 0) + 1

            recent_runs = [
                {
                    "created_at": record.get("created_at"),
                    "run_id": record.get("run_id"),
                    "result_quality": record.get("result_quality"),
                    "status": record.get("status"),
                    "summary": record.get("summary"),
                }
                for record in ordered_records[:3]
            ]
            latest_quality = str(ordered_records[0].get("result_quality") or "unknown")
            prior_qualities = [str(record.get("result_quality") or "unknown") for record in ordered_records[1:]]
            target_groups.append(
                {
                    "target_label": target_label,
                    "run_count": len(ordered_records),
                    "result_quality_counts": result_quality_counts,
                    "recent_runs": recent_runs,
                    "latest_run_id": ordered_records[0].get("run_id"),
                    "trend_summary": self._trend_summary(
                        latest_quality=latest_quality,
                        prior_qualities=prior_qualities,
                    ),
                }
            )

        target_groups.sort(
            key=lambda item: (
                item["target_label"] == "<unlabeled>",
                -int(item["run_count"]),
                str(item["target_label"]).lower(),
            )
        )
        return {
            "session_id": session_id,
            "project_id": project_id,
            "tool_id": tool_id,
            "capability": capability,
            "date_from": date_from,
            "date_to": date_to,
            "record_count": len(records),
            "target_count": len(target_groups),
            "target_groups": target_groups,
        }

    def _score_record(
        self,
        record: dict[str, Any],
        needle: str,
        *,
        query_understanding,
    ) -> ArchiveSearchMatch | None:
        keyword_score = 0
        matched_fields: list[str] = []
        field_weights = {
            "summary": 5,
            "tool_id": 3,
            "capability": 3,
            "status": 2,
            "structured_data": 4,
            "logs": 1,
            "provenance": 1,
        }

        for field, weight in field_weights.items():
            value = record.get(field)
            haystack = value.lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()
            if needle in haystack:
                keyword_score += weight
                matched_fields.append(field)

        if keyword_score == 0:
            haystack = json.dumps(record, sort_keys=True).lower()
            if needle not in haystack:
                semantic_score = self._score_semantic_candidate(
                    query_understanding=query_understanding,
                    prompt=self._semantic_text(
                        summary=str(record.get("summary") or ""),
                        capability=str(record.get("capability") or ""),
                        tool_id=str(record.get("tool_id") or ""),
                        target_label=self._research_target_label(record),
                        result_quality=self._research_result_quality(record),
                    ),
                    semantic_signature=None,
                )
                if semantic_score <= 0:
                    return None
                score = semantic_score
                matched_fields.append("semantic")
                score_breakdown = {
                    "keyword_score": 0,
                    "semantic_score": semantic_score,
                }
            else:
                score = 1
                matched_fields.append("record")
                score_breakdown = {
                    "keyword_score": 1,
                    "semantic_score": 0,
                }
        else:
            semantic_score = self._score_semantic_candidate(
                query_understanding=query_understanding,
                prompt=self._semantic_text(
                    summary=str(record.get("summary") or ""),
                    capability=str(record.get("capability") or ""),
                    tool_id=str(record.get("tool_id") or ""),
                    target_label=self._research_target_label(record),
                    result_quality=self._research_result_quality(record),
                ),
                semantic_signature=None,
            )
            score = self._blend_search_score(keyword_score, semantic_score)
            if semantic_score > 0:
                matched_fields.append("semantic")
            score_breakdown = {
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
            }

        return ArchiveSearchMatch(
            score=score,
            matched_fields=matched_fields,
            score_breakdown=score_breakdown,
            record=record,
        )

    @staticmethod
    def _trend_summary(*, latest_quality: str, prior_qualities: list[str]) -> str:
        latest_label = latest_quality.replace("_", " ")
        normalized_prior = [quality for quality in prior_qualities if quality]
        if not normalized_prior:
            return f"latest run is {latest_label}; no earlier runs to compare"
        if all(quality == latest_quality for quality in normalized_prior):
            return f"latest run remains {latest_label}; trend is steady"
        unique_prior = sorted({quality.replace("_", " ") for quality in normalized_prior})
        return (
            f"latest run is {latest_label}; earlier runs were mixed "
            f"({', '.join(unique_prior)})"
        )

    @staticmethod
    def _record_in_date_range(
        record: dict[str, Any],
        *,
        date_from: str | None,
        date_to: str | None,
    ) -> bool:
        if not date_from and not date_to:
            return True

        created_at = record.get("created_at")
        if not created_at:
            return False

        try:
            record_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return False

        if date_from:
            start_dt = ArchiveManager._parse_filter_datetime(date_from, end_of_day=False)
            if record_dt < start_dt:
                return False

        if date_to:
            end_dt = ArchiveManager._parse_filter_datetime(date_to, end_of_day=True)
            if record_dt > end_dt:
                return False

        return True

    @staticmethod
    def _parse_filter_datetime(value: str, *, end_of_day: bool) -> datetime:
        raw = value.strip()
        try:
            if "T" in raw:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            else:
                suffix = "T23:59:59+00:00" if end_of_day else "T00:00:00+00:00"
                dt = datetime.fromisoformat(f"{raw}{suffix}")
        except ValueError as exc:
            raise ValueError(
                f"Invalid date filter '{value}'. Use YYYY-MM-DD or ISO datetime."
            ) from exc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    def load_record(self, archive_path: Path) -> dict[str, Any]:
        self.persistence_manager.record_fallback_read("archive raw artifact hydration")
        return self._load_structured_record(archive_path)

    def _load_structured_record(self, archive_path: Path) -> dict[str, Any]:
        payload = ArchiveRecordSchema.normalize(
            json.loads(archive_path.read_text(encoding="utf-8"))
        )
        payload["structured_data"] = self._sanitize_nested_payload(payload.get("structured_data"))
        payload["provenance"] = self._sanitize_nested_payload(payload.get("provenance"))
        research_run = ResearchRunSummary.from_structured_data(payload.get("structured_data"))
        if research_run is not None:
            payload["research_run"] = research_run.to_dict()
            payload["run_id"] = research_run.run_id
            payload["target_label"] = research_run.target_label
            payload["result_quality"] = research_run.analysis_status.result_quality
        ArchiveRecordSchema.validate(payload)
        payload["archive_path"] = str(archive_path)
        return payload

    def _iter_record_paths(self, root: Path) -> list[Path]:
        indexed_paths = self._indexed_paths(root)
        extras = sorted(
            (
                path
                for path in root.rglob("*.json")
                if path.name != self.INDEX_FILENAME and path not in indexed_paths
            ),
            reverse=True,
        )
        return indexed_paths + extras

    def _indexed_paths(self, root: Path) -> list[Path]:
        entries = self._load_recursive_index_entries(root)
        paths: list[Path] = []
        for entry_root, entry in entries:
            path = entry_root / str(entry.get("path", ""))
            if path.exists():
                paths.append(path)
        paths.sort(reverse=True)
        return paths

    def _append_index_entry(self, payload: dict[str, Any], archive_path: Path) -> None:
        index_root = archive_path.parent
        index_path = self._index_path(index_root)
        entries = self._load_index(index_root)
        relative_path = archive_path.relative_to(index_root).as_posix()
        entries = [entry for entry in entries if entry.get("path") != relative_path]
        semantic_signature = self.semantic_matcher.signature_from_candidate(
            SemanticCandidate(
                prompt=self._semantic_text(
                    summary=str(payload.get("summary") or ""),
                    capability=str(payload.get("capability") or ""),
                    tool_id=str(payload.get("tool_id") or ""),
                    target_label=self._research_target_label(payload),
                    result_quality=self._research_result_quality(payload),
                ),
                normalized_topic=None,
                dominant_intent=None,
                extracted_entities=(),
            )
        )
        entries.append(
            {
                "path": relative_path,
                "session_id": payload.get("session_id"),
                "tool_id": payload.get("tool_id"),
                "capability": payload.get("capability"),
                "status": payload.get("status"),
                "summary": payload.get("summary"),
                "target_label": self._research_target_label(payload),
                "result_quality": self._research_result_quality(payload),
                "semantic_signature": semantic_signature.to_dict(),
                "created_at": payload.get("created_at"),
            }
        )
        entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def _load_index(self, root: Path) -> list[dict[str, Any]]:
        index_path = self._index_path(root)
        if not index_path.exists():
            return []
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _search_index(
        self,
        needle: str,
        *,
        query_understanding,
        session_id: str | None,
        tool_id: str | None,
        capability: str | None,
        status: str | None,
    ) -> list[tuple[int, int, dict[str, int], Path]]:
        root = self.archive_root
        if session_id:
            root = root / session_id
        if tool_id:
            root = root / tool_id
        if capability:
            root = root / capability
        entries = self._load_recursive_index_entries(root)
        matches: list[tuple[int, int, dict[str, int], Path]] = []
        for entry_root, entry in entries:
            if session_id and entry.get("session_id") != session_id:
                continue
            if tool_id and entry.get("tool_id") != tool_id:
                continue
            if capability and entry.get("capability") != capability:
                continue
            if status and str(entry.get("status", "")).lower() != status.strip().lower():
                continue
            score, semantic_score, score_breakdown = self._score_index_entry(
                entry,
                needle,
                query_understanding=query_understanding,
            )
            if score <= 0:
                continue
            path = entry_root / str(entry.get("path", ""))
            if path.exists():
                matches.append((score, semantic_score, score_breakdown, path))
        matches.sort(
            key=lambda item: (
                item[0],
                item[1],
                self._indexed_created_at(item[3]),
                str(item[3]),
            ),
            reverse=True,
        )
        return matches[: self.settings.search_candidate_limit]

    def _score_index_entry(
        self,
        entry: dict[str, Any],
        needle: str,
        *,
        query_understanding,
    ) -> tuple[int, int, dict[str, int]]:
        keyword_score = 0
        field_weights = {
            "summary": 5,
            "tool_id": 3,
            "capability": 3,
            "status": 2,
        }
        for field, weight in field_weights.items():
            value = entry.get(field)
            haystack = str(value or "").lower()
            if needle in haystack:
                keyword_score += weight
        semantic_score = self._score_semantic_candidate(
            query_understanding=query_understanding,
            prompt=self._semantic_text(
                summary=str(entry.get("summary") or ""),
                capability=str(entry.get("capability") or ""),
                tool_id=str(entry.get("tool_id") or ""),
                target_label=str(entry.get("target_label") or ""),
                result_quality=str(entry.get("result_quality") or ""),
            ),
            semantic_signature=entry.get("semantic_signature"),
        )
        score_breakdown = {
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
        }
        return self._blend_search_score(keyword_score, semantic_score), semantic_score, score_breakdown

    def _score_semantic_candidate(
        self,
        *,
        query_understanding,
        prompt: str,
        semantic_signature: object | None,
    ) -> int:
        if isinstance(semantic_signature, dict):
            match = self.semantic_matcher.score_signature(
                query_understanding,
                self._signature_from_index_entry(semantic_signature),
            )
        else:
            understanding = self.prompt_nlu.analyze(prompt)
            match = self.semantic_matcher.score(
                query_understanding,
                SemanticCandidate(
                    prompt=understanding.normalized_text,
                    normalized_topic=understanding.topic.value,
                    dominant_intent=understanding.intent.label,
                    extracted_entities=tuple(
                        entity.value.strip().lower()
                        for entity in understanding.entities
                        if entity.value
                    ),
                ),
            )
        return match.score

    @staticmethod
    def _signature_from_index_entry(signature: dict[str, object]):
        from lumen.retrieval.semantic_matcher import SemanticSignature

        return SemanticSignature(
            prompt_tokens=tuple(
                str(item).strip().lower()
                for item in (signature.get("prompt_tokens") or [])
                if str(item).strip()
            ),
            topic_tokens=tuple(
                str(item).strip().lower()
                for item in (signature.get("topic_tokens") or [])
                if str(item).strip()
            ),
            dominant_intent=str(signature.get("dominant_intent") or "").strip().lower() or None,
            entities=tuple(
                str(item).strip().lower()
                for item in (signature.get("entities") or [])
                if str(item).strip()
            ),
        )

    @staticmethod
    def _semantic_text(
        *,
        summary: str,
        capability: str,
        tool_id: str,
        target_label: str = "",
        result_quality: str = "",
    ) -> str:
        capability_text = capability.replace(".", " ").replace("_", " ")
        semantic_aliases = ArchiveManager._semantic_alias_text(
            tool_id=tool_id,
            capability=capability,
            target_label=target_label,
        )
        return " ".join(
            part
            for part in [
                summary.strip(),
                capability_text.strip(),
                tool_id.strip(),
                target_label.strip(),
                result_quality.strip().replace("_", " "),
                semantic_aliases,
            ]
            if part
        )

    @staticmethod
    def _semantic_alias_text(*, tool_id: str, capability: str, target_label: str) -> str:
        normalized_tool = str(tool_id or "").strip().lower()
        normalized_capability = str(capability or "").strip().lower()
        normalized_target = str(target_label or "").strip().lower()

        aliases: list[str] = []
        if normalized_tool == "anh" or normalized_capability == "spectral_dip_scan":
            aliases.extend(
                [
                    "anh",
                    "great attractor",
                    "ga",
                    "local bulk flow analysis",
                    "spectral dip scan",
                    "si iv spectral scan",
                ]
            )
        if "great attractor" in normalized_target or normalized_target.startswith("ga "):
            aliases.extend(["great attractor", "ga", "local bulk flow analysis"])
        return " ".join(dict.fromkeys(alias for alias in aliases if alias))

    @staticmethod
    def _research_target_label(record: dict[str, Any]) -> str:
        research_run = record.get("research_run")
        if isinstance(research_run, dict):
            return str(research_run.get("target_label") or "")
        structured_data = record.get("structured_data")
        if isinstance(structured_data, dict):
            return str(structured_data.get("target_label") or "")
        return ""

    @staticmethod
    def _research_result_quality(record: dict[str, Any]) -> str:
        research_run = record.get("research_run")
        if isinstance(research_run, dict):
            analysis_status = research_run.get("analysis_status")
            if isinstance(analysis_status, dict):
                return str(analysis_status.get("result_quality") or "")
        structured_data = record.get("structured_data")
        if isinstance(structured_data, dict):
            analysis_status = structured_data.get("analysis_status")
            if isinstance(analysis_status, dict):
                return str(analysis_status.get("result_quality") or "")
        return ""

    def _index_path(self, root: Path) -> Path:
        return root / self.INDEX_FILENAME

    def _load_recursive_index_entries(self, root: Path) -> list[tuple[Path, dict[str, Any]]]:
        entries: list[tuple[Path, dict[str, Any]]] = []
        if not root.exists():
            return entries
        for index_path in root.rglob(self.INDEX_FILENAME):
            index_root = index_path.parent
            for entry in self._load_index(index_root):
                entries.append((index_root, entry))
        return entries

    @staticmethod
    def _indexed_created_at(path: Path) -> str:
        return path.stem

    @staticmethod
    def _blend_search_score(keyword_score: int, semantic_score: int) -> int:
        if keyword_score <= 0:
            return semantic_score
        semantic_bonus = min(semantic_score, max(2, min(6, keyword_score // 2 + 1)))
        return keyword_score + semantic_bonus

    @staticmethod
    def _semantic_tiebreak(matched_fields: list[str]) -> int:
        return 1 if "semantic" in matched_fields else 0

    def _build_payload(
        self,
        *,
        session_id: str,
        result: ToolResult,
        timestamp: datetime,
    ) -> dict[str, Any]:
        structured_data = self._sanitize_nested_payload(result.structured_data)
        provenance = self._sanitize_nested_payload(result.provenance)
        return {
            "session_id": session_id,
            "tool_id": result.tool_id,
            "capability": result.capability,
            "status": result.status,
            "summary": result.summary,
            "created_at": timestamp.isoformat(),
            "run_dir": str(result.run_dir) if result.run_dir else None,
            "error": result.error,
            "structured_data": structured_data,
            "artifacts": [
                {
                    "name": artifact.name,
                    "path": str(artifact.path),
                    "media_type": artifact.media_type,
                    "description": artifact.description,
                }
                for artifact in result.artifacts
            ],
            "logs": result.logs,
            "provenance": provenance,
        }

    @staticmethod
    def _sanitize_nested_payload(value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, nested_value in value.items():
                if key in {"response", "top_interaction_matches", "top_matches"}:
                    continue
                sanitized[key] = ArchiveManager._sanitize_nested_payload(nested_value)
            return sanitized
        if isinstance(value, list):
            return [ArchiveManager._sanitize_nested_payload(item) for item in value]
        return value

    @staticmethod
    def serialize_record(record: ArchivedRunRecord) -> dict[str, Any]:
        payload = asdict(record)
        payload["archive_path"] = str(record.archive_path)
        payload["run_dir"] = str(record.run_dir) if record.run_dir else None
        payload["created_at"] = record.created_at.isoformat()
        return payload

    def _hydrate_db_record(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        archive_record = metadata.get("archive_record") if isinstance(metadata, dict) else None
        if isinstance(archive_record, dict):
            hydrated = dict(archive_record)
            archive_path = str(hydrated.get("archive_path") or row.get("archive_path") or "").strip()
            if archive_path:
                hydrated["archive_path"] = archive_path
            return self._derive_research_run_fields(hydrated)
        return self._derive_research_run_fields({
            "session_id": row.get("session_id"),
            "project_id": row.get("project_id"),
            "tool_id": row.get("tool_name"),
            "capability": row.get("capability"),
            "status": "ok" if bool(row.get("success")) else "error",
            "summary": row.get("output_summary"),
            "created_at": row.get("created_at"),
            "run_dir": row.get("run_dir"),
            "archive_path": row.get("archive_path"),
            "target_label": metadata.get("target_label") if isinstance(metadata, dict) else None,
            "result_quality": metadata.get("result_quality") if isinstance(metadata, dict) else None,
            "structured_data": metadata.get("archive_record", {}).get("structured_data")
            if isinstance(metadata.get("archive_record"), dict)
            else None,
        })

    @staticmethod
    def _derive_research_run_fields(payload: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(payload)
        structured_data = hydrated.get("structured_data")
        if isinstance(structured_data, dict):
            hydrated["structured_data"] = ArchiveManager._sanitize_nested_payload(structured_data)
        research_run = ResearchRunSummary.from_structured_data(hydrated.get("structured_data"))
        if research_run is not None:
            hydrated["research_run"] = research_run.to_dict()
            hydrated["run_id"] = research_run.run_id
            hydrated["target_label"] = research_run.target_label
            hydrated["result_quality"] = research_run.analysis_status.result_quality
        elif isinstance(hydrated.get("structured_data"), dict):
            structured_data = hydrated["structured_data"]
            analysis_status = structured_data.get("analysis_status")
            if not hydrated.get("run_id") and structured_data.get("run_id") is not None:
                hydrated["run_id"] = structured_data.get("run_id")
            if not hydrated.get("target_label") and structured_data.get("target_label") is not None:
                hydrated["target_label"] = structured_data.get("target_label")
            if (
                not hydrated.get("result_quality")
                and isinstance(analysis_status, dict)
                and analysis_status.get("result_quality") is not None
            ):
                hydrated["result_quality"] = analysis_status.get("result_quality")
        return hydrated
