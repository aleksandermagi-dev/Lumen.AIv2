from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ResearchRunAnalysisStatus:
    validated: bool
    analysis_ran: bool
    plot_generated: bool
    line_detected: bool | None
    result_quality: str | None
    failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ResearchRunSummary:
    bundle_standard: str
    bundle_id: str
    capability: str
    run_id: str | None
    target_label: str | None
    input_files: list[str]
    analysis_status: ResearchRunAnalysisStatus
    batch_record: dict[str, Any]
    domain_payload: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_standard": self.bundle_standard,
            "bundle_id": self.bundle_id,
            "capability": self.capability,
            "run_id": self.run_id,
            "target_label": self.target_label,
            "input_files": list(self.input_files),
            "analysis_status": self.analysis_status.to_dict(),
            "batch_record": dict(self.batch_record),
            "domain_payload": dict(self.domain_payload),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_structured_data(cls, payload: dict[str, Any] | None) -> "ResearchRunSummary" | None:
        if not isinstance(payload, dict):
            return None
        if str(payload.get("bundle_standard") or "").strip() != "lumen_research_bundle_v1":
            return None
        analysis_status_payload = payload.get("analysis_status")
        if not isinstance(analysis_status_payload, dict):
            return None
        return cls(
            bundle_standard="lumen_research_bundle_v1",
            bundle_id=str(payload.get("bundle_id") or ""),
            capability=str(payload.get("capability") or ""),
            run_id=str(payload.get("run_id") or "").strip() or None,
            target_label=str(payload.get("target_label") or "").strip() or None,
            input_files=[
                str(item)
                for item in (payload.get("input_files") or [])
                if str(item).strip()
            ],
            analysis_status=ResearchRunAnalysisStatus(
                validated=bool(analysis_status_payload.get("validated")),
                analysis_ran=bool(analysis_status_payload.get("analysis_ran")),
                plot_generated=bool(analysis_status_payload.get("plot_generated")),
                line_detected=(
                    None
                    if analysis_status_payload.get("line_detected") is None
                    else bool(analysis_status_payload.get("line_detected"))
                ),
                result_quality=str(analysis_status_payload.get("result_quality") or "").strip() or None,
                failure_reason=str(analysis_status_payload.get("failure_reason") or "").strip() or None,
            ),
            batch_record=dict(payload.get("batch_record") or {}),
            domain_payload=dict(payload.get("domain_payload") or {}),
            provenance=dict(payload.get("provenance") or {}),
        )
