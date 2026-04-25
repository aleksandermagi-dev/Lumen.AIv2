from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

from tool_bundles.anh.adapters.research_bundle_standard import ResearchArtifactRecord


@dataclass(frozen=True, slots=True)
class ProcessedSummaryOutput:
    status: str
    logs: list[str]
    summary: dict[str, object]
    artifact_records: list[ResearchArtifactRecord]


class ANHResultSummary:
    """Normalize already-derived ANH outputs into a stable structured summary."""

    def summarize(self, *, staged_input: Path, output_dir: Path) -> ProcessedSummaryOutput:
        suffix = staged_input.suffix.lower()
        if suffix == ".csv":
            return self._summarize_csv(staged_input=staged_input, output_dir=output_dir)
        if suffix == ".json":
            return self._summarize_json(staged_input=staged_input, output_dir=output_dir)
        return self._summarize_text(staged_input=staged_input, output_dir=output_dir)

    def _summarize_csv(self, *, staged_input: Path, output_dir: Path) -> ProcessedSummaryOutput:
        with staged_input.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = [str(name or "") for name in (reader.fieldnames or [])]

        normalized_rows = [self._normalize_csv_row(row) for row in rows]
        strongest = self._strongest_candidate(normalized_rows)
        summary_statistics = {
            "record_count": len(normalized_rows),
            "fieldnames": fieldnames,
            "candidate_count": len([row for row in normalized_rows if row.get("best_velocity_kms") is not None]),
        }
        normalized_path = output_dir / f"{staged_input.stem}_normalized_results.json"
        normalized_path.write_text(
            json.dumps(
                {
                    "source_file": str(staged_input),
                    "source_format": "csv_summary",
                    "records": normalized_rows,
                    "summary_statistics": summary_statistics,
                    "strongest_candidate": strongest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_records = [
            ResearchArtifactRecord(
                role="processed_results_normalized",
                path=normalized_path,
                media_type="application/json",
                description="Normalized ANH processed-result summary.",
            )
        ]
        return ProcessedSummaryOutput(
            status="ok",
            logs=[f"Summarized processed CSV results from {staged_input.name}."],
            summary={
                "filename": staged_input.name,
                "staged_path": str(staged_input),
                "source_kind": "processed_results",
                "recognized_format": "csv_summary",
                "line_detected": strongest is not None,
                "strongest_candidate": strongest,
                "line_results": [],
                "parsed_rows": normalized_rows,
                "summary_statistics": summary_statistics,
                "artifacts": [str(normalized_path)],
                "failure_reason": None,
            },
            artifact_records=artifact_records,
        )

    def _summarize_json(self, *, staged_input: Path, output_dir: Path) -> ProcessedSummaryOutput:
        payload = json.loads(staged_input.read_text(encoding="utf-8"))
        normalized_rows = self._normalize_json_payload(payload)
        strongest = self._strongest_candidate(normalized_rows)
        summary_statistics = {
            "record_count": len(normalized_rows),
            "source_shape": type(payload).__name__,
            "candidate_count": len([row for row in normalized_rows if row.get("best_velocity_kms") is not None]),
        }
        normalized_path = output_dir / f"{staged_input.stem}_normalized_results.json"
        normalized_path.write_text(
            json.dumps(
                {
                    "source_file": str(staged_input),
                    "source_format": "json_summary",
                    "records": normalized_rows,
                    "summary_statistics": summary_statistics,
                    "strongest_candidate": strongest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_records = [
            ResearchArtifactRecord(
                role="processed_results_normalized",
                path=normalized_path,
                media_type="application/json",
                description="Normalized ANH processed-result summary.",
            )
        ]
        return ProcessedSummaryOutput(
            status="ok",
            logs=[f"Summarized processed JSON results from {staged_input.name}."],
            summary={
                "filename": staged_input.name,
                "staged_path": str(staged_input),
                "source_kind": "processed_results",
                "recognized_format": "json_summary",
                "line_detected": strongest is not None,
                "strongest_candidate": strongest,
                "line_results": [],
                "parsed_rows": normalized_rows,
                "summary_statistics": summary_statistics,
                "artifacts": [str(normalized_path)],
                "failure_reason": None,
            },
            artifact_records=artifact_records,
        )

    def _summarize_text(self, *, staged_input: Path, output_dir: Path) -> ProcessedSummaryOutput:
        text = staged_input.read_text(encoding="utf-8")
        summary_statistics = {
            "line_count": len(text.splitlines()),
            "contains_si_iv": "si iv" in text.lower(),
            "contains_velocity": "km/s" in text.lower() or "velocity" in text.lower(),
        }
        normalized_path = output_dir / f"{staged_input.stem}_normalized_results.json"
        strongest = None
        normalized_path.write_text(
            json.dumps(
                {
                    "source_file": str(staged_input),
                    "source_format": "text_summary",
                    "summary_statistics": summary_statistics,
                    "preview": text.splitlines()[:20],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_records = [
            ResearchArtifactRecord(
                role="processed_results_normalized",
                path=normalized_path,
                media_type="application/json",
                description="Normalized ANH processed-result summary.",
            )
        ]
        return ProcessedSummaryOutput(
            status="ok",
            logs=[f"Summarized processed text results from {staged_input.name}."],
            summary={
                "filename": staged_input.name,
                "staged_path": str(staged_input),
                "source_kind": "processed_results",
                "recognized_format": "text_summary",
                "line_detected": False,
                "strongest_candidate": strongest,
                "line_results": [],
                "parsed_rows": [],
                "summary_statistics": summary_statistics,
                "artifacts": [str(normalized_path)],
                "failure_reason": None,
            },
            artifact_records=artifact_records,
        )

    @staticmethod
    def _normalize_csv_row(row: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in row.items():
            label = str(key or "").strip()
            cell = str(value or "").strip()
            normalized[label] = cell
        best_velocity = None
        best_wavelength = None
        for key, value in normalized.items():
            lower = key.lower()
            if best_velocity is None and "v (km/s)" in lower:
                best_velocity = ANHResultSummary._to_float(value)
            if best_wavelength is None and ("λ" in lower or "lambda" in lower):
                best_wavelength = ANHResultSummary._to_float(value)
        return {
            "file_label": normalized.get("File") or normalized.get("file"),
            "best_velocity_kms": best_velocity,
            "best_wavelength": best_wavelength,
            "raw": normalized,
        }

    @staticmethod
    def _normalize_json_payload(payload: object) -> list[dict[str, object]]:
        if isinstance(payload, dict):
            candidate_files = payload.get("candidate_files")
            if isinstance(candidate_files, list):
                return [
                    {
                        "file_label": item.get("filename") if isinstance(item, dict) else None,
                        "best_velocity_kms": ANHResultSummary._to_float(item.get("velocity_kms")) if isinstance(item, dict) else None,
                        "best_wavelength": ANHResultSummary._to_float(item.get("lambda_min")) if isinstance(item, dict) else None,
                        "raw": dict(item) if isinstance(item, dict) else {"value": item},
                    }
                    for item in candidate_files
                ]
            return [
                {
                    "file_label": None,
                    "best_velocity_kms": ANHResultSummary._to_float(payload.get("velocity_kms")) if isinstance(payload, dict) else None,
                    "best_wavelength": ANHResultSummary._to_float(payload.get("lambda_min")) if isinstance(payload, dict) else None,
                    "raw": dict(payload),
                }
            ]
        if isinstance(payload, list):
            rows = []
            for item in payload:
                if isinstance(item, dict):
                    rows.append(
                        {
                            "file_label": item.get("filename") or item.get("file"),
                            "best_velocity_kms": ANHResultSummary._to_float(item.get("velocity_kms") or item.get("best_velocity_kms")),
                            "best_wavelength": ANHResultSummary._to_float(item.get("lambda_min") or item.get("best_wavelength")),
                            "raw": dict(item),
                        }
                    )
            return rows
        return []

    @staticmethod
    def _strongest_candidate(rows: list[dict[str, object]]) -> dict[str, object] | None:
        ranked = [row for row in rows if row.get("best_velocity_kms") is not None]
        if not ranked:
            return None
        ranked.sort(key=lambda item: abs(float(item.get("best_velocity_kms") or 0.0)), reverse=True)
        top = dict(ranked[0])
        return {
            "filename": top.get("file_label"),
            "velocity_kms": top.get("best_velocity_kms"),
            "lambda_min": top.get("best_wavelength"),
        }

    @staticmethod
    def _to_float(value: object) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None
