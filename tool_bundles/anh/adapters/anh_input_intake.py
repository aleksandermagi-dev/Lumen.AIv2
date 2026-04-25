from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import zipfile


RAW_SPECTRAL_SUFFIXES = {".fits", ".fit", ".fz"}
STRUCTURED_TEXT_SUFFIXES = {".csv", ".json", ".txt"}
ARCHIVE_SUFFIXES = {".zip"}
RESULT_FILENAME_HINTS = (
    "si_iv",
    "absorption",
    "velocity",
    "candidate",
    "summary",
    "results",
)
MANIFEST_FILENAME_HINTS = ("manifest", "batch", "filelist", "files", "inputs")
PATH_COLUMN_HINTS = ("path", "file", "filepath", "filename", "input_path")
RESULT_COLUMN_HINTS = (
    "si iv 1393",
    "si iv 1402",
    "velocity",
    "v (km/s)",
    "lambda",
    "wavelength",
    "candidate",
)
SPECTRAL_REQUIRED_COLUMNS = {"WAVELENGTH", "FLUX"}


@dataclass(frozen=True, slots=True)
class ANHRecognizedMember:
    path: str
    category: str
    recognized_format: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    source_kind: str = "input"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "category": self.category,
            "recognized_format": self.recognized_format,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True, slots=True)
class ANHInputRecognitionResult:
    category: str
    execution_mode: str
    recognized_format: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    archive_inputs: tuple[Path, ...] = ()
    raw_inputs: tuple[Path, ...] = ()
    processed_inputs: tuple[Path, ...] = ()
    skipped_inputs: tuple[dict[str, object], ...] = ()
    member_classifications: tuple[ANHRecognizedMember, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "execution_mode": self.execution_mode,
            "recognized_format": self.recognized_format,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "archive_inputs": [str(path) for path in self.archive_inputs],
            "raw_inputs": [str(path) for path in self.raw_inputs],
            "processed_inputs": [str(path) for path in self.processed_inputs],
            "skipped_inputs": [dict(item) for item in self.skipped_inputs],
            "member_classifications": [item.to_dict() for item in self.member_classifications],
        }


class ANHInputIntake:
    """Recognize ANH-compatible inputs before scientific execution."""

    def recognize(self, input_path: Path, *, max_files: int = 25) -> ANHInputRecognitionResult:
        source = input_path.resolve()
        if not source.exists():
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format="missing_path",
                confidence="high",
                reasons=[f"Input does not exist: {source}"],
            )

        if source.is_dir():
            return self._recognize_directory(source, max_files=max_files)

        return self._recognize_file(source, max_files=max_files)

    def _recognize_directory(self, source: Path, *, max_files: int) -> ANHInputRecognitionResult:
        files = [item for item in sorted(source.rglob("*")) if item.is_file()]
        if not files:
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format="directory",
                confidence="high",
                reasons=[f"Input directory contains no files: {source}"],
            )

        considered = files[: max(1, max_files)]
        raw_inputs: list[Path] = []
        processed_inputs: list[Path] = []
        skipped: list[dict[str, object]] = []
        members: list[ANHRecognizedMember] = []

        if len(files) > len(considered):
            skipped.append(
                {
                    "path": str(source),
                    "reason": f"Directory contained {len(files)} files; only the first {len(considered)} were considered.",
                }
            )

        for item in considered:
            member = self._classify_member(item)
            members.append(member)
            if member.category == "raw_spectral_input":
                raw_inputs.append(item)
            elif member.category == "processed_results_input":
                processed_inputs.append(item)
            else:
                skipped.append({"path": str(item), "reason": "; ".join(member.reasons)})

        return self._batch_result(
            recognized_format="directory_batch",
            reasons=[f"Scanned directory batch at {source}."],
            raw_inputs=raw_inputs,
            processed_inputs=processed_inputs,
            skipped=skipped,
            members=members,
        )

    def _recognize_file(self, source: Path, *, max_files: int) -> ANHInputRecognitionResult:
        suffix = source.suffix.lower()
        if suffix in ARCHIVE_SUFFIXES:
            archive_member = self._classify_archive_file(source)
            if archive_member.category == "archive_input":
                return ANHInputRecognitionResult(
                    category="archive_input",
                    execution_mode="archive_scan",
                    recognized_format=archive_member.recognized_format,
                    confidence=archive_member.confidence,
                    reasons=list(archive_member.reasons),
                    archive_inputs=(source,),
                    member_classifications=(archive_member,),
                )
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format=archive_member.recognized_format,
                confidence=archive_member.confidence,
                reasons=list(archive_member.reasons),
                skipped_inputs=({"path": str(source), "reason": "; ".join(archive_member.reasons)},),
                member_classifications=(archive_member,),
            )
        if suffix in RAW_SPECTRAL_SUFFIXES:
            member = self._classify_raw_spectral_file(source)
            if member.category == "raw_spectral_input":
                return ANHInputRecognitionResult(
                    category=member.category,
                    execution_mode="scan",
                    recognized_format=member.recognized_format,
                    confidence=member.confidence,
                    reasons=list(member.reasons),
                    raw_inputs=(source,),
                    member_classifications=(member,),
                )
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format=member.recognized_format,
                confidence=member.confidence,
                reasons=list(member.reasons),
                skipped_inputs=({"path": str(source), "reason": "; ".join(member.reasons)},),
                member_classifications=(member,),
            )

        if suffix in STRUCTURED_TEXT_SUFFIXES:
            manifest_paths = self._parse_manifest_members(source)
            if manifest_paths is not None:
                raw_inputs: list[Path] = []
                processed_inputs: list[Path] = []
                skipped: list[dict[str, object]] = []
                members: list[ANHRecognizedMember] = [
                    ANHRecognizedMember(
                        path=str(source),
                        category="batch_input",
                        recognized_format=f"{suffix.lstrip('.')} manifest",
                        confidence="high",
                        reasons=[f"Manifest lists {len(manifest_paths)} file reference(s)."],
                        source_kind="manifest",
                    )
                ]
                limited_paths = manifest_paths[: max(1, max_files)]
                if len(manifest_paths) > len(limited_paths):
                    skipped.append(
                        {
                            "path": str(source),
                            "reason": f"Manifest listed {len(manifest_paths)} file(s); only the first {len(limited_paths)} were considered.",
                        }
                    )
                for item in limited_paths:
                    if not item.exists():
                        skipped.append({"path": str(item), "reason": "Manifest entry does not exist."})
                        members.append(
                            ANHRecognizedMember(
                                path=str(item),
                                category="invalid_input",
                                recognized_format="missing_manifest_entry",
                                confidence="high",
                                reasons=["Manifest entry does not exist."],
                                source_kind="manifest_member",
                            )
                        )
                        continue
                    member = self._classify_member(item, source_kind="manifest_member")
                    members.append(member)
                    if member.category == "raw_spectral_input":
                        raw_inputs.append(item)
                    elif member.category == "processed_results_input":
                        processed_inputs.append(item)
                    else:
                        skipped.append({"path": str(item), "reason": "; ".join(member.reasons)})
                return self._batch_result(
                    recognized_format=f"{suffix.lstrip('.')} manifest",
                    reasons=[f"Expanded batch manifest at {source}."],
                    raw_inputs=raw_inputs,
                    processed_inputs=processed_inputs,
                    skipped=skipped,
                    members=members,
                )

            member = self._classify_processed_results_file(source)
            if member.category == "processed_results_input":
                return ANHInputRecognitionResult(
                    category=member.category,
                    execution_mode="summarize_results",
                    recognized_format=member.recognized_format,
                    confidence=member.confidence,
                    reasons=list(member.reasons),
                    processed_inputs=(source,),
                    member_classifications=(member,),
                )
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format=member.recognized_format,
                confidence=member.confidence,
                reasons=list(member.reasons),
                skipped_inputs=({"path": str(source), "reason": "; ".join(member.reasons)},),
                member_classifications=(member,),
            )

        return ANHInputRecognitionResult(
            category="invalid_input",
            execution_mode="reject",
            recognized_format=suffix.lstrip(".") or "unknown",
            confidence="high",
            reasons=[f"Unsupported input type: {source.suffix or '<no suffix>'}"],
            skipped_inputs=({"path": str(source), "reason": "Unsupported input type."},),
            member_classifications=(
                ANHRecognizedMember(
                    path=str(source),
                    category="invalid_input",
                    recognized_format=suffix.lstrip(".") or "unknown",
                    confidence="high",
                    reasons=[f"Unsupported input type: {source.suffix or '<no suffix>'}"],
                ),
            ),
        )

    def _batch_result(
        self,
        *,
        recognized_format: str,
        reasons: list[str],
        raw_inputs: list[Path],
        processed_inputs: list[Path],
        skipped: list[dict[str, object]],
        members: list[ANHRecognizedMember],
    ) -> ANHInputRecognitionResult:
        if not raw_inputs and not processed_inputs:
            return ANHInputRecognitionResult(
                category="invalid_input",
                execution_mode="reject",
                recognized_format=recognized_format,
                confidence="high",
                reasons=list(reasons) + ["No usable ANH-compatible files were found."],
                skipped_inputs=tuple(skipped),
                member_classifications=tuple(members),
            )
        if raw_inputs and processed_inputs:
            execution_mode = "batch_mixed"
        else:
            execution_mode = "batch_scan" if raw_inputs else "summarize_results"
        return ANHInputRecognitionResult(
            category="batch_input",
            execution_mode=execution_mode,
            recognized_format=recognized_format,
            confidence="high" if raw_inputs or processed_inputs else "medium",
            reasons=list(reasons),
            archive_inputs=(),
            raw_inputs=tuple(raw_inputs),
            processed_inputs=tuple(processed_inputs),
            skipped_inputs=tuple(skipped),
            member_classifications=tuple(members),
        )

    def classify_path(self, path: Path, *, source_kind: str = "batch_member") -> ANHRecognizedMember:
        return self._classify_member(path, source_kind=source_kind)

    def _classify_member(self, path: Path, *, source_kind: str = "batch_member") -> ANHRecognizedMember:
        suffix = path.suffix.lower()
        if suffix in RAW_SPECTRAL_SUFFIXES:
            member = self._classify_raw_spectral_file(path)
            return ANHRecognizedMember(
                path=member.path,
                category=member.category,
                recognized_format=member.recognized_format,
                confidence=member.confidence,
                reasons=member.reasons,
                source_kind=source_kind,
            )
        if suffix in STRUCTURED_TEXT_SUFFIXES:
            member = self._classify_processed_results_file(path)
            return ANHRecognizedMember(
                path=member.path,
                category=member.category,
                recognized_format=member.recognized_format,
                confidence=member.confidence,
                reasons=member.reasons,
                source_kind=source_kind,
            )
        return ANHRecognizedMember(
            path=str(path),
            category="invalid_input",
            recognized_format=suffix.lstrip(".") or "unknown",
            confidence="high",
            reasons=[f"Unsupported input type: {path.suffix or '<no suffix>'}"],
            source_kind=source_kind,
        )

    def _classify_archive_file(self, path: Path) -> ANHRecognizedMember:
        reasons: list[str] = []
        try:
            with zipfile.ZipFile(path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if not members:
                    return ANHRecognizedMember(
                        path=str(path),
                        category="invalid_input",
                        recognized_format="zip_empty",
                        confidence="high",
                        reasons=["Archive contains no files."],
                    )
                member_names = [item.filename.lower() for item in members]
        except Exception as exc:
            return ANHRecognizedMember(
                path=str(path),
                category="invalid_input",
                recognized_format="zip_unreadable",
                confidence="high",
                reasons=[f"Archive inspection failed: {exc.__class__.__name__}: {exc}"],
            )

        has_raw = any(
            name.endswith(tuple(RAW_SPECTRAL_SUFFIXES))
            or "_x1d" in Path(name).stem.lower()
            or "x1dsum" in Path(name).stem.lower()
            for name in member_names
        )
        has_processed = any(name.endswith(tuple(STRUCTURED_TEXT_SUFFIXES)) for name in member_names)
        if has_raw or has_processed:
            reasons.append(f"Archive lists {len(member_names)} file member(s).")
            if has_raw:
                reasons.append("Archive contains likely FITS/x1d spectral products.")
            if has_processed:
                reasons.append("Archive also contains structured text members.")
            return ANHRecognizedMember(
                path=str(path),
                category="archive_input",
                recognized_format="mast_archive_zip",
                confidence="high" if has_raw else "medium",
                reasons=reasons,
            )
        return ANHRecognizedMember(
            path=str(path),
            category="invalid_input",
            recognized_format="zip_unknown",
            confidence="medium",
            reasons=["Archive does not contain recognizable ANH-compatible spectral or summary members."],
        )

    def _classify_raw_spectral_file(self, path: Path) -> ANHRecognizedMember:
        reasons: list[str] = []
        lowered = path.name.lower()
        if "x1d" in lowered:
            reasons.append("Filename suggests an x1d-style spectral product.")
        inspection = self._inspect_fits_structure(path)
        if inspection["is_spectral"]:
            reasons.extend(inspection["reasons"])
            confidence = "high"
            recognized_format = str(inspection.get("recognized_format") or "fits_spectrum")
            return ANHRecognizedMember(
                path=str(path),
                category="raw_spectral_input",
                recognized_format=recognized_format,
                confidence=confidence,
                reasons=reasons,
            )
        if "x1d" in lowered:
            reasons.extend(inspection["reasons"] or ["Spectral-table inspection was unavailable, but filename strongly suggests x1d input."])
            return ANHRecognizedMember(
                path=str(path),
                category="raw_spectral_input",
                recognized_format="x1d_candidate",
                confidence="medium",
                reasons=reasons,
            )
        reasons.extend(inspection["reasons"] or ["FITS file did not expose expected spectral columns."])
        return ANHRecognizedMember(
            path=str(path),
            category="invalid_input",
            recognized_format="fits_unknown",
            confidence="medium",
            reasons=reasons,
        )

    def _classify_processed_results_file(self, path: Path) -> ANHRecognizedMember:
        suffix = path.suffix.lower()
        lowered_name = path.name.lower()
        reasons: list[str] = []
        if any(hint in lowered_name for hint in RESULT_FILENAME_HINTS):
            reasons.append("Filename suggests processed ANH result content.")

        try:
            if suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = [str(name or "") for name in (reader.fieldnames or [])]
                normalized_fields = [name.strip().lower() for name in fieldnames if name]
                if self._looks_like_processed_result_fields(normalized_fields):
                    reasons.append("CSV columns match ANH-style summary/result fields.")
                    return ANHRecognizedMember(
                        path=str(path),
                        category="processed_results_input",
                        recognized_format="csv_summary",
                        confidence="high",
                        reasons=reasons,
                    )
                return ANHRecognizedMember(
                    path=str(path),
                    category="invalid_input",
                    recognized_format="csv_unknown",
                    confidence="medium",
                    reasons=reasons or ["CSV file does not contain ANH-style summary fields or manifest path columns."],
                )

            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if self._looks_like_processed_json(payload):
                    reasons.append("JSON keys match ANH-style processed result content.")
                    return ANHRecognizedMember(
                        path=str(path),
                        category="processed_results_input",
                        recognized_format="json_summary",
                        confidence="high",
                        reasons=reasons,
                    )
                return ANHRecognizedMember(
                    path=str(path),
                    category="invalid_input",
                    recognized_format="json_unknown",
                    confidence="medium",
                    reasons=reasons or ["JSON file is not an ANH result summary or manifest."],
                )

            text = path.read_text(encoding="utf-8")
            normalized_text = text.lower()
            if "si iv" in normalized_text and ("km/s" in normalized_text or "velocity" in normalized_text):
                reasons.append("Text file contains Si IV velocity summary cues.")
                return ANHRecognizedMember(
                    path=str(path),
                    category="processed_results_input",
                    recognized_format="text_summary",
                    confidence="medium",
                    reasons=reasons,
                )
            return ANHRecognizedMember(
                path=str(path),
                category="invalid_input",
                recognized_format="text_unknown",
                confidence="medium",
                reasons=reasons or ["Text file is not an ANH-style result summary or supported manifest."],
            )
        except Exception as exc:
            return ANHRecognizedMember(
                path=str(path),
                category="invalid_input",
                recognized_format=suffix.lstrip(".") or "unknown",
                confidence="high",
                reasons=[f"Failed to inspect structured input: {exc.__class__.__name__}: {exc}"],
            )

    def _parse_manifest_members(self, path: Path) -> list[Path] | None:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                members = self._json_manifest_paths(payload, path.parent)
                return members if members else None
            if suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    fieldnames = [str(name or "") for name in (reader.fieldnames or [])]
                path_columns = [name for name in fieldnames if name and name.strip().lower() in PATH_COLUMN_HINTS]
                if not path_columns:
                    return None
                resolved: list[Path] = []
                path_like_entries = 0
                for row in rows:
                    for column in path_columns:
                        candidate = str(row.get(column) or "").strip()
                        if not candidate:
                            continue
                        if not self._looks_like_path_reference(candidate):
                            continue
                        path_like_entries += 1
                        resolved.append(self._resolve_manifest_path(candidate, path.parent))
                        break
                return resolved if path_like_entries > 0 and resolved else None
            if suffix == ".txt":
                resolved = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    candidate = line.strip()
                    if not candidate or candidate.startswith("#"):
                        continue
                    resolved.append(self._resolve_manifest_path(candidate, path.parent))
                return resolved or None
        except Exception:
            return None
        return None

    @staticmethod
    def _json_manifest_paths(payload: object, base_dir: Path) -> list[Path]:
        if isinstance(payload, list) and payload and all(isinstance(item, str) for item in payload):
            return [ANHInputIntake._resolve_manifest_path(str(item), base_dir) for item in payload]
        if isinstance(payload, dict):
            for key in ("files", "paths", "input_files", "members"):
                value = payload.get(key)
                if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                    return [ANHInputIntake._resolve_manifest_path(str(item), base_dir) for item in value]
        return []

    @staticmethod
    def _resolve_manifest_path(candidate: str, base_dir: Path) -> Path:
        path = Path(candidate)
        return path if path.is_absolute() else (base_dir / path).resolve()

    @staticmethod
    def _looks_like_path_reference(candidate: str) -> bool:
        normalized = str(candidate or "").strip()
        if not normalized:
            return False
        if any(token in normalized for token in ("/", "\\")):
            return True
        suffix = Path(normalized).suffix.lower()
        if suffix in RAW_SPECTRAL_SUFFIXES | STRUCTURED_TEXT_SUFFIXES:
            return True
        return normalized.lower().endswith("_x1d")

    @staticmethod
    def _looks_like_processed_result_fields(fieldnames: list[str]) -> bool:
        if not fieldnames:
            return False
        joined = " ".join(fieldnames)
        return any(hint in joined for hint in RESULT_COLUMN_HINTS)

    @staticmethod
    def _looks_like_processed_json(payload: object) -> bool:
        if isinstance(payload, dict):
            keys = {str(key).lower() for key in payload}
            return bool(
                {"candidate_files", "strongest_candidate", "line_results", "parsed_results", "analysis_status"} & keys
            )
        if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
            keys = {str(key).lower() for item in payload[:5] for key in item}
            return any(hint in " ".join(keys) for hint in RESULT_COLUMN_HINTS)
        return False

    @staticmethod
    def _inspect_fits_structure(path: Path) -> dict[str, object]:
        try:
            from astropy.io import fits
        except Exception as exc:
            return {
                "is_spectral": False,
                "recognized_format": "fits_unchecked",
                "reasons": [f"Astropy was unavailable for spectral inspection: {exc.__class__.__name__}."],
            }

        try:
            with fits.open(path) as hdul:
                for hdu in hdul:
                    data = getattr(hdu, "data", None)
                    names = getattr(data, "names", None)
                    if not names:
                        continue
                    normalized = {str(name).upper() for name in names}
                    if SPECTRAL_REQUIRED_COLUMNS.issubset(normalized):
                        return {
                            "is_spectral": True,
                            "recognized_format": "mast_x1d_fits",
                            "reasons": ["FITS table contains WAVELENGTH and FLUX columns."],
                        }
            return {
                "is_spectral": False,
                "recognized_format": "fits_unknown",
                "reasons": ["FITS file does not expose WAVELENGTH and FLUX columns."],
            }
        except Exception as exc:
            return {
                "is_spectral": False,
                "recognized_format": "fits_unreadable",
                "reasons": [f"FITS inspection failed: {exc.__class__.__name__}: {exc}"],
            }
