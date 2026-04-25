from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import zipfile

from tool_bundles.anh.adapters.anh_input_intake import ANHInputIntake, ANHRecognizedMember


@dataclass(frozen=True, slots=True)
class ANHRawStagingResult:
    discovered_files: tuple[str, ...]
    extracted_files: tuple[str, ...]
    staged_good_raw: tuple[Path, ...]
    staged_processed: tuple[Path, ...]
    staged_failed: tuple[dict[str, object], ...]
    skipped_non_spectral: tuple[dict[str, object], ...]
    manifest_paths: dict[str, str]
    member_classifications: tuple[ANHRecognizedMember, ...]
    logs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "discovered_files": list(self.discovered_files),
            "extracted_files": list(self.extracted_files),
            "staged_good_raw": [str(path) for path in self.staged_good_raw],
            "staged_processed": [str(path) for path in self.staged_processed],
            "staged_failed": [dict(item) for item in self.staged_failed],
            "skipped_non_spectral": [dict(item) for item in self.skipped_non_spectral],
            "manifest_paths": dict(self.manifest_paths),
            "member_classifications": [item.to_dict() for item in self.member_classifications],
            "logs": list(self.logs),
        }


class ANHRawIntakeStaging:
    """Rebuild the old ANH pre-sort flow as a Lumen-native staging step."""

    def __init__(self, *, intake: ANHInputIntake | None = None):
        self.intake = intake or ANHInputIntake()

    def stage(self, *, source: Path, run_dir: Path, max_files: int = 100) -> ANHRawStagingResult:
        inputs_dir = run_dir / "inputs"
        manifests_dir = inputs_dir / "manifests"
        staged_good_dir = inputs_dir / "staged" / "good"
        staged_failed_dir = inputs_dir / "staged" / "failed"
        staged_processed_dir = inputs_dir / "staged" / "processed"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        staged_good_dir.mkdir(parents=True, exist_ok=True)
        staged_failed_dir.mkdir(parents=True, exist_ok=True)
        staged_processed_dir.mkdir(parents=True, exist_ok=True)

        logs: list[str] = []
        extracted_files: list[str] = []
        if source.is_file() and source.suffix.lower() == ".zip":
            extracted_root = inputs_dir / "raw_archive"
            extracted_root.mkdir(parents=True, exist_ok=True)
            extracted_files = self._extract_archive(source=source, destination=extracted_root, logs=logs)
            candidate_root = extracted_root
        else:
            candidate_root = source
            if source.is_dir():
                logs.append(f"Scanning raw batch directory {source}.")
            else:
                logs.append(f"Scanning staged source {source}.")

        discovered_members = [item for item in sorted(candidate_root.rglob("*")) if item.is_file()]
        discovered_paths = [str(item) for item in discovered_members]
        staged_good_raw: list[Path] = []
        staged_processed: list[Path] = []
        staged_failed: list[dict[str, object]] = []
        skipped_non_spectral: list[dict[str, object]] = []
        classifications: list[ANHRecognizedMember] = []

        classified_members: list[tuple[int, Path, ANHRecognizedMember]] = []
        for index, member_path in enumerate(discovered_members, start=1):
            source_kind = "archive_member" if source.suffix.lower() == ".zip" else "batch_member"
            classification = self.intake.classify_path(member_path, source_kind=source_kind)
            classifications.append(classification)
            classified_members.append((index, member_path, classification))

        valid_members = [
            item
            for item in classified_members
            if item[2].category in {"raw_spectral_input", "processed_results_input"}
        ]
        limited_valid_members = valid_members[: max(1, max_files)]
        limited_member_paths = {item[1] for item in limited_valid_members}

        for index, member_path, classification in classified_members:
            if member_path not in limited_member_paths:
                if classification.category in {"raw_spectral_input", "processed_results_input"}:
                    staged_failed.append(
                        {
                            "path": str(member_path),
                            "reason": f"Valid ANH-compatible file limit exceeded; only {max(1, max_files)} valid file(s) were staged.",
                        }
                    )
                    self._copy_if_possible(member_path, staged_failed_dir, index=index)
                else:
                    skipped_non_spectral.append({"path": str(member_path), "reason": "; ".join(classification.reasons)})
                continue
            if classification.category == "raw_spectral_input":
                staged_good_raw.append(self._stage_file(member_path, staged_good_dir, index=index))
            elif classification.category == "processed_results_input":
                staged_processed.append(self._stage_file(member_path, staged_processed_dir, index=index))

        manifest_paths = self._write_manifests(
            manifests_dir=manifests_dir,
            discovered_paths=discovered_paths,
            staged_good=staged_good_raw,
            staged_failed=staged_failed,
            staged_processed=staged_processed,
            skipped_non_spectral=skipped_non_spectral,
        )
        logs.append(f"Staged {len(staged_good_raw)} raw spectral file(s) and {len(staged_processed)} processed summary file(s).")
        if skipped_non_spectral:
            logs.append(f"Skipped {len(skipped_non_spectral)} non-spectral support file(s).")
        if staged_failed:
            logs.append(f"Marked {len(staged_failed)} valid ANH-compatible file(s) as failed or not staged.")

        return ANHRawStagingResult(
            discovered_files=tuple(discovered_paths),
            extracted_files=tuple(extracted_files),
            staged_good_raw=tuple(staged_good_raw),
            staged_processed=tuple(staged_processed),
            staged_failed=tuple(staged_failed),
            skipped_non_spectral=tuple(skipped_non_spectral),
            manifest_paths=manifest_paths,
            member_classifications=tuple(classifications),
            logs=tuple(logs),
        )

    @staticmethod
    def _extract_archive(*, source: Path, destination: Path, logs: list[str]) -> list[str]:
        extracted: list[str] = []
        with zipfile.ZipFile(source) as archive:
            archive.extractall(destination)
            for member in archive.infolist():
                if member.is_dir():
                    continue
                extracted.append(str(destination / member.filename))
        logs.append(f"Extracted {len(extracted)} archive member(s) from {source.name}.")
        return extracted

    @staticmethod
    def _stage_file(source: Path, destination_dir: Path, *, index: int) -> Path:
        staged_name = source.name if index == 1 else f"{index:03d}_{source.name}"
        destination = destination_dir / staged_name
        shutil.copy2(source, destination)
        return destination

    @staticmethod
    def _copy_if_possible(source: Path, destination_dir: Path, *, index: int) -> None:
        try:
            staged_name = source.name if index == 1 else f"{index:03d}_{source.name}"
            shutil.copy2(source, destination_dir / staged_name)
        except Exception:
            return

    @staticmethod
    def _write_manifests(
        *,
        manifests_dir: Path,
        discovered_paths: list[str],
        staged_good: list[Path],
        staged_failed: list[dict[str, object]],
        staged_processed: list[Path],
        skipped_non_spectral: list[dict[str, object]],
    ) -> dict[str, str]:
        inputs_manifest = manifests_dir / "inputs_manifest.txt"
        good_manifest = manifests_dir / "good_manifest.txt"
        failed_manifest = manifests_dir / "failed_manifest.txt"
        staging_summary = manifests_dir / "staging_summary.json"
        inputs_manifest.write_text("\n".join(discovered_paths), encoding="utf-8")
        good_manifest.write_text("\n".join(path.name for path in staged_good), encoding="utf-8")
        with failed_manifest.open("w", encoding="utf-8") as handle:
            for item in staged_failed:
                handle.write(f"{item['path']}\t{item['reason']}\n")
        staging_summary.write_text(
            json.dumps(
                {
                    "discovered_count": len(discovered_paths),
                    "staged_good_count": len(staged_good),
                    "staged_failed_count": len(staged_failed),
                    "staged_processed_count": len(staged_processed),
                    "skipped_non_spectral_count": len(skipped_non_spectral),
                    "good_manifest": [path.name for path in staged_good],
                    "processed_manifest": [path.name for path in staged_processed],
                    "failed_manifest": list(staged_failed),
                    "skipped_non_spectral": list(skipped_non_spectral),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "inputs_manifest": str(inputs_manifest),
            "good_manifest": str(good_manifest),
            "failed_manifest": str(failed_manifest),
            "staging_summary": str(staging_summary),
        }
