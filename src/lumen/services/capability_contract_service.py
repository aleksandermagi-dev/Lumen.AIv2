from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CapabilityContract:
    domain_id: str
    label: str
    status: str
    scope_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "status": self.status,
            "scope_note": self.scope_note,
        }


class CapabilityContractService:
    """Defines which adjacent capabilities are supported, bounded, gated, or not promised."""

    CONTRACTS: tuple[CapabilityContract, ...] = (
        CapabilityContract(
            domain_id="writing_editing",
            label="Writing and editing",
            status="provider_gated",
            scope_note="Summaries are available locally, but rewrite/translation/drafting workflows depend on hosted inference when requested explicitly.",
        ),
        CapabilityContract(
            domain_id="explainability_transparency",
            label="Explainability and transparency",
            status="supported",
            scope_note="Route, tool, retrieval, and diagnostic transparency are part of the current runtime.",
        ),
        CapabilityContract(
            domain_id="dataset_analysis",
            label="Dataset and analysis workflows",
            status="bounded",
            scope_note="Analysis and trend work are bounded to supplied or local data and advisory guidance.",
        ),
        CapabilityContract(
            domain_id="academic_writing",
            label="Academic writing support",
            status="bounded",
            scope_note="Supports brainstorming, outlining, revision, rhetorical analysis, and integrity-safe academic help.",
        ),
        CapabilityContract(
            domain_id="citation_support",
            label="Citation support",
            status="bounded",
            scope_note="Formats supplied citation metadata, flags uncertainty, and avoids fabricating missing source details.",
        ),
        CapabilityContract(
            domain_id="literature_synthesis",
            label="Literature synthesis",
            status="bounded",
            scope_note="Synthesizes supplied papers or local source text without claiming external scholarly verification.",
        ),
        CapabilityContract(
            domain_id="college_math_science_support",
            label="College math and science support",
            status="bounded",
            scope_note="Supports bounded conceptual reasoning across core college math/science topics with prerequisite bridge help.",
        ),
        CapabilityContract(
            domain_id="supervised_ml_data_support",
            label="Supervised ML data support",
            status="bounded",
            scope_note="Supports local or user-supplied dataset readiness review, split guidance, leakage checks, and evaluation planning.",
        ),
        CapabilityContract(
            domain_id="invention_design",
            label="Invention and design support",
            status="bounded",
            scope_note="Concept, constraint, material, and failure-mode support are available, but outputs stay high-level and non-signoff.",
        ),
        CapabilityContract(
            domain_id="self_modification",
            label="Self-editing / self-modification",
            status="not_promised",
            scope_note="Lumen does not edit itself from inside runtime conversations.",
        ),
        CapabilityContract(
            domain_id="autonomous_automation",
            label="Autonomous automation / RPA",
            status="not_promised",
            scope_note="Broad background automation is not part of the current runtime contract.",
        ),
        CapabilityContract(
            domain_id="speech_audio",
            label="Speech / audio understanding",
            status="not_promised",
            scope_note="No first-class speech or audio reasoning surface is currently available.",
        ),
        CapabilityContract(
            domain_id="vision_imaging",
            label="Vision / imaging understanding",
            status="not_promised",
            scope_note="The desktop shell can display images, but the runtime does not promise production image understanding.",
        ),
        CapabilityContract(
            domain_id="live_news_politics",
            label="Live world news and politics",
            status="not_promised",
            scope_note="Lumen does not claim real-time news or politics authority.",
        ),
        CapabilityContract(
            domain_id="investing_advice",
            label="Investing and market advice",
            status="not_promised",
            scope_note="Lumen does not provide smart-market or investing-advice authority.",
        ),
        CapabilityContract(
            domain_id="health_prediction",
            label="Health-risk prediction and diagnosis",
            status="not_promised",
            scope_note="Lumen does not provide diagnosis, treatment, or health-risk prediction authority.",
        ),
        CapabilityContract(
            domain_id="fabrication_grade_schematics",
            label="Fabrication-grade schematics",
            status="not_promised",
            scope_note="High-level design support exists, but fabrication-ready engineering signoff is not promised.",
        ),
    )

    _NOT_PROMISED_PROMPT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "self_modification",
            (
                "edit yourself",
                "modify yourself",
                "rewrite your own code",
                "patch yourself",
                "change your own code",
                "upgrade yourself",
                "self edit",
                "self-edit",
            ),
        ),
        (
            "speech_audio",
            (
                "transcribe this audio",
                "analyze this audio",
                "analyze this song",
                "tempo",
                "rhythm",
                "lyrics from this audio",
            ),
        ),
        (
            "vision_imaging",
            (
                "what is in this image",
                "analyze this image",
                "identify this object",
                "identify this person",
                "analyze this photo",
                "what is in this picture",
            ),
        ),
        (
            "live_news_politics",
            (
                "latest news",
                "news today",
                "what happened today",
                "latest politics",
                "current politics",
            ),
        ),
        (
            "investing_advice",
            (
                "what stock should i buy",
                "which stock should i buy",
                "predict the market",
                "investment advice",
                "make investments",
            ),
        ),
        (
            "health_prediction",
            (
                "diagnose me",
                "diagnose this condition",
                "treatment plan",
                "predict my health risk",
                "health risk prediction",
            ),
        ),
        (
            "fabrication_grade_schematics",
            (
                "fabrication-ready schematic",
                "engineering signoff",
                "manufacturing blueprint",
                "detailed production schematic",
            ),
        ),
        (
            "autonomous_automation",
            (
                "run this automatically forever",
                "autonomously manage",
                "automate my computer",
                "robotic process automation",
            ),
        ),
    )

    _DATASET_GUIDANCE_PATTERNS: tuple[str, ...] = (
        "dataset license",
        "dataset licensing",
        "data license",
        "check licensing",
        "training dataset",
        "ml dataset",
        "machine learning dataset",
        "openml",
        "hugging face dataset",
        "kaggle",
        "google dataset search",
    )

    @classmethod
    def build_report(cls) -> dict[str, object]:
        payload = [item.to_dict() for item in cls.CONTRACTS]
        counts: dict[str, int] = {}
        for item in payload:
            status = str(item["status"])
            counts[status] = counts.get(status, 0) + 1
        return {
            "contracts": payload,
            "status_counts": counts,
        }

    @classmethod
    def match_not_promised_surface(cls, *, prompt: str, input_path: Path | None = None) -> dict[str, object] | None:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if not normalized:
            return None
        if input_path is not None:
            suffix = input_path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"} and any(
                token in normalized for token in ("image", "photo", "picture", "identify", "what is in")
            ):
                return cls.contract_for("vision_imaging")
            if suffix in {".mp3", ".wav", ".m4a", ".flac", ".ogg"} and any(
                token in normalized for token in ("audio", "song", "tempo", "rhythm", "transcribe", "lyrics")
            ):
                return cls.contract_for("speech_audio")
        for domain_id, patterns in cls._NOT_PROMISED_PROMPT_PATTERNS:
            if any(pattern in normalized for pattern in patterns):
                return cls.contract_for(domain_id)
        return None

    @classmethod
    def is_dataset_guidance_request(cls, *, prompt: str) -> bool:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        return bool(normalized) and any(pattern in normalized for pattern in cls._DATASET_GUIDANCE_PATTERNS)

    @classmethod
    def contract_for(cls, domain_id: str) -> dict[str, object] | None:
        for item in cls.CONTRACTS:
            if item.domain_id == domain_id:
                return item.to_dict()
        return None

    @classmethod
    def response_status_for_payload(cls, response: dict[str, object]) -> dict[str, object] | None:
        existing = response.get("capability_status")
        if isinstance(existing, dict) and existing.get("status"):
            return dict(existing)
        mode = str(response.get("mode") or "").strip().lower()
        tool_execution = response.get("tool_execution")
        tool_id = ""
        if isinstance(tool_execution, dict):
            tool_id = str(tool_execution.get("tool_id") or "").strip().lower()
        if mode == "safety":
            safety_decision = response.get("safety_decision")
            if isinstance(safety_decision, dict) and str(safety_decision.get("category") or "").strip() == "self_modification":
                return cls._status_payload(
                    domain_id="self_modification",
                    status="not_promised",
                    details="Lumen does not edit itself from inside the runtime conversation.",
                )
        if tool_id == "content":
            tool_result = response.get("tool_result")
            structured = getattr(tool_result, "structured_data", {}) if tool_result is not None else {}
            if not isinstance(structured, dict):
                structured = {}
            failure_category = str(structured.get("failure_category") or "").strip()
            if failure_category == "missing_provider_config":
                return cls._status_payload(
                    domain_id="writing_editing",
                    status="provider_gated",
                    details="Hosted writing and content generation need a configured hosted provider.",
                )
            return cls._status_payload(
                domain_id="writing_editing",
                status="supported",
                details="Hosted writing/content generation completed successfully.",
            )
        if tool_id == "paper":
            return cls._status_payload(
                domain_id="literature_synthesis",
                status="bounded",
                details="Paper support is bounded to supplied or configured source text and should be verified before formal use.",
            )
        if tool_id == "math":
            return cls._status_payload(
                domain_id="college_math_science_support",
                status="bounded",
                details="Math support is bounded to explanation and existing tool-backed problem-solving surfaces.",
            )
        if tool_id in {"data", "viz", "report"}:
            return cls._status_payload(
                domain_id="supervised_ml_data_support",
                status="bounded",
                details="Data-analysis support is bounded to local or user-supplied datasets and should be treated as advisory.",
            )
        if tool_id in {"invent", "design", "experiment"}:
            return cls._status_payload(
                domain_id="invention_design",
                status="bounded",
                details="Design and invention support stay conceptual, constraint-aware, and non-signoff.",
            )
        return None

    @staticmethod
    def _status_payload(*, domain_id: str, status: str, details: str) -> dict[str, object]:
        return {
            "domain_id": domain_id,
            "status": status,
            "details": details,
        }
