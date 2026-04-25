from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from lumen.app.settings import AppSettings


@dataclass(slots=True)
class _CleanupCandidate:
    path: Path
    category: str
    size_bytes: int | None = None
    reason: str = ""


class StorageHygieneService:
    """Reports and prunes oversized or stale runtime artifacts."""

    def __init__(self, settings: AppSettings):
        self.settings = settings

    def report(self) -> dict[str, object]:
        oversized_interactions = self._oversized_json_files(
            root=self.settings.interactions_root,
            max_bytes=self.settings.max_interaction_record_bytes,
        )
        oversized_sessions = self._oversized_json_files(
            root=self.settings.sessions_root,
            max_bytes=self.settings.max_session_state_bytes,
        )
        prunable_tool_runs = self._prunable_tool_run_dirs(
            retain_per_capability=self.settings.tool_run_retention_per_capability,
        )
        return {
            "repo_root": str(self.settings.repo_root),
            "limits": {
                "max_interaction_record_bytes": self.settings.max_interaction_record_bytes,
                "max_session_state_bytes": self.settings.max_session_state_bytes,
                "tool_run_retention_per_capability": self.settings.tool_run_retention_per_capability,
            },
            "oversized_interaction_files": [self._candidate_payload(item) for item in oversized_interactions],
            "oversized_session_files": [self._candidate_payload(item) for item in oversized_sessions],
            "prunable_tool_run_dirs": [self._candidate_payload(item) for item in prunable_tool_runs],
            "counts": {
                "oversized_interaction_files": len(oversized_interactions),
                "oversized_session_files": len(oversized_sessions),
                "prunable_tool_run_dirs": len(prunable_tool_runs),
            },
        }

    def cleanup(
        self,
        *,
        prune_oversized: bool = True,
        prune_tool_runs: bool = True,
        retain_per_capability: int | None = None,
    ) -> dict[str, object]:
        retain = max(1, int(retain_per_capability or self.settings.tool_run_retention_per_capability))
        removed: list[_CleanupCandidate] = []

        if prune_oversized:
            removed.extend(
                self._remove_candidates(
                    self._oversized_json_files(
                        root=self.settings.interactions_root,
                        max_bytes=self.settings.max_interaction_record_bytes,
                    )
                )
            )
            removed.extend(
                self._remove_candidates(
                    self._oversized_json_files(
                        root=self.settings.sessions_root,
                        max_bytes=self.settings.max_session_state_bytes,
                    )
                )
            )

        if prune_tool_runs:
            removed.extend(
                self._remove_candidates(
                    self._prunable_tool_run_dirs(retain_per_capability=retain)
                )
            )

        self._remove_empty_dirs(self.settings.tool_runs_root)
        self._remove_empty_dirs(self.settings.interactions_root)
        self._remove_empty_dirs(self.settings.sessions_root)

        return {
            "repo_root": str(self.settings.repo_root),
            "removed_count": len(removed),
            "removed": [self._candidate_payload(item) for item in removed],
            "retain_per_capability": retain,
        }

    def _oversized_json_files(self, *, root: Path, max_bytes: int) -> list[_CleanupCandidate]:
        if not root.exists():
            return []
        candidates: list[_CleanupCandidate] = []
        for path in root.rglob("*.json"):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                candidates.append(
                    _CleanupCandidate(
                        path=path,
                        category="oversized_json",
                        size_bytes=size,
                        reason=f"File exceeds {max_bytes} bytes",
                    )
                )
        candidates.sort(key=lambda item: (item.size_bytes or 0, str(item.path)), reverse=True)
        return candidates

    def _prunable_tool_run_dirs(self, *, retain_per_capability: int) -> list[_CleanupCandidate]:
        root = self.settings.tool_runs_root
        if not root.exists():
            return []
        candidates: list[_CleanupCandidate] = []
        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            for tool_dir in session_dir.iterdir():
                if not tool_dir.is_dir():
                    continue
                for capability_dir in tool_dir.iterdir():
                    if not capability_dir.is_dir():
                        continue
                    run_dirs = sorted((path for path in capability_dir.iterdir() if path.is_dir()), reverse=True)
                    for stale in run_dirs[retain_per_capability:]:
                        candidates.append(
                            _CleanupCandidate(
                                path=stale,
                                category="stale_tool_run",
                                size_bytes=self._dir_size(stale),
                                reason=(
                                    f"Older than the newest {retain_per_capability} runs for "
                                    f"{session_dir.name}/{tool_dir.name}/{capability_dir.name}"
                                ),
                            )
                        )
        candidates.sort(key=lambda item: (str(item.path.parent), str(item.path)), reverse=True)
        return candidates

    def _remove_candidates(self, candidates: list[_CleanupCandidate]) -> list[_CleanupCandidate]:
        removed: list[_CleanupCandidate] = []
        for candidate in candidates:
            try:
                if candidate.path.is_dir():
                    shutil.rmtree(candidate.path)
                elif candidate.path.exists():
                    candidate.path.unlink()
                else:
                    continue
            except OSError:
                continue
            removed.append(candidate)
        return removed

    @staticmethod
    def _remove_empty_dirs(root: Path) -> None:
        if not root.exists():
            return
        for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                next(path.iterdir())
            except StopIteration:
                try:
                    path.rmdir()
                except OSError:
                    continue
            except OSError:
                continue

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except OSError:
            return total
        return total

    @staticmethod
    def _candidate_payload(candidate: _CleanupCandidate) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": str(candidate.path),
            "category": candidate.category,
            "reason": candidate.reason,
        }
        if candidate.size_bytes is not None:
            payload["size_bytes"] = candidate.size_bytes
        return payload
