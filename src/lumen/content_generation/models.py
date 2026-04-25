from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


VALID_STYLE_PROFILES = {"observational", "analytical", "educational", "story"}
VALID_PLATFORMS = {"all", "master", "tiktok", "youtube_shorts"}
VALID_POST_TIME_SLOTS = {"morning", "afternoon", "evening"}
VALID_SAFETY_DECISIONS = {"PASS", "REWRITE", "DISCARD"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned


def _normalize_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, list):
        lines = [_clean_string(str(item), "script line") for item in value if str(item).strip()]
    else:
        raise ValueError("script_lines must be a list of strings or a multiline string.")
    if not lines:
        raise ValueError("script_lines cannot be empty.")
    return lines


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_tags = [part.strip() for part in value.replace(",", " ").split()]
    elif isinstance(value, list):
        raw_tags = [str(part).strip() for part in value]
    else:
        raise ValueError("hashtags must be a string, list, or null.")
    return [tag for tag in raw_tags if tag]


def _normalize_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_string(value, field_name)


@dataclass(slots=True)
class GeneratedIdea:
    rank: int
    title: str
    rationale: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1.")
        self.title = _clean_string(self.title, "title")
        self.rationale = _clean_string(self.rationale, "rationale")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, rank: int | None = None) -> "GeneratedIdea":
        item_rank = rank if rank is not None else int(payload.get("rank", 0))
        return cls(
            rank=item_rank,
            title=payload.get("title", payload.get("topic", "")),
            rationale=payload.get("rationale", payload.get("reason", "")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentSafetyAssessment:
    decision: str = "PASS"
    dimension_results: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, int | None] = field(default_factory=dict)
    rewrite_attempts: int = 0
    last_evaluated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        self.decision = _clean_string(self.decision, "decision").upper()
        if self.decision not in VALID_SAFETY_DECISIONS:
            raise ValueError("decision must be PASS, REWRITE, or DISCARD.")
        self.dimension_results = {
            _clean_string(str(key), "dimension"): _clean_string(str(value), "dimension result").upper()
            for key, value in dict(self.dimension_results or {}).items()
        }
        self.reasons = [_clean_string(str(reason), "reason") for reason in self.reasons]
        self.last_evaluated_at = _clean_string(self.last_evaluated_at, "last_evaluated_at")

    @classmethod
    def from_mapping(cls, payload: Any) -> "ContentSafetyAssessment":
        if isinstance(payload, cls):
            return payload
        data = payload if isinstance(payload, dict) else {}
        return cls(
            decision=str(data.get("decision", "PASS")),
            dimension_results=dict(data.get("dimension_results", {})),
            reasons=list(data.get("reasons", [])),
            scores=dict(data.get("scores", {})),
            rewrite_attempts=int(data.get("rewrite_attempts", 0)),
            last_evaluated_at=str(data.get("last_evaluated_at", _utc_now_iso())),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneratedContentDraft:
    id: str
    topic: str
    hook: str
    script_lines: list[str]
    caption: str
    hashtags: list[str]
    platform_notes: str | None
    retention_check: str
    suggested_post_time: str
    style_profile: str = "observational"
    platform: str = "master"
    status: str = "draft"
    created_at: str = field(default_factory=_utc_now_iso)
    safety: ContentSafetyAssessment = field(default_factory=ContentSafetyAssessment)

    def __post_init__(self) -> None:
        self.id = _clean_string(self.id, "id")
        self.topic = _clean_string(self.topic, "topic")
        self.hook = _clean_string(self.hook, "hook")
        self.script_lines = _normalize_lines(self.script_lines)
        self.caption = _clean_string(self.caption, "caption")
        self.hashtags = _normalize_tags(self.hashtags)
        self.platform_notes = _normalize_optional_string(self.platform_notes, "platform_notes")
        self.retention_check = _clean_string(self.retention_check, "retention_check")
        self.suggested_post_time = _clean_string(self.suggested_post_time, "suggested_post_time").lower()
        if self.suggested_post_time not in VALID_POST_TIME_SLOTS:
            raise ValueError("suggested_post_time must be morning, afternoon, or evening.")
        self.style_profile = _clean_string(self.style_profile, "style_profile").lower()
        if self.style_profile not in VALID_STYLE_PROFILES:
            raise ValueError("Unknown style_profile.")
        self.platform = _clean_string(self.platform, "platform").lower()
        if self.platform not in VALID_PLATFORMS:
            raise ValueError("Unknown platform.")
        self.status = _clean_string(self.status, "status").lower()
        self.created_at = _clean_string(self.created_at, "created_at")
        if not isinstance(self.safety, ContentSafetyAssessment):
            self.safety = ContentSafetyAssessment.from_mapping(self.safety)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "GeneratedContentDraft":
        return cls(
            id=str(payload.get("id") or uuid4().hex[:8]),
            topic=payload.get("topic", ""),
            hook=payload.get("hook", ""),
            script_lines=payload.get("script_lines", payload.get("script", "")),
            caption=payload.get("caption", ""),
            hashtags=payload.get("hashtags", payload.get("tags", [])),
            platform_notes=payload.get("platform_notes"),
            retention_check=payload.get("retention_check", ""),
            suggested_post_time=payload.get("suggested_post_time", "afternoon"),
            style_profile=payload.get("style_profile", "observational"),
            platform=payload.get("platform", "master"),
            status=payload.get("status", "draft"),
            created_at=payload.get("created_at", _utc_now_iso()),
            safety=ContentSafetyAssessment.from_mapping(payload.get("safety")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **changes: Any) -> "GeneratedContentDraft":
        payload = self.to_mapping()
        payload.update(changes)
        return GeneratedContentDraft.from_mapping(payload)


@dataclass(slots=True)
class GeneratedContentVariant:
    id: str
    source_draft_id: str
    platform: str
    topic: str
    hook: str
    script_lines: list[str]
    caption: str
    hashtags: list[str]
    platform_notes: str | None
    retention_check: str
    suggested_post_time: str
    style_profile: str = "observational"
    created_at: str = field(default_factory=_utc_now_iso)
    safety: ContentSafetyAssessment = field(default_factory=ContentSafetyAssessment)

    def __post_init__(self) -> None:
        self.id = _clean_string(self.id, "id")
        self.source_draft_id = _clean_string(self.source_draft_id, "source_draft_id")
        self.platform = _clean_string(self.platform, "platform").lower()
        if self.platform not in {"tiktok", "youtube_shorts"}:
            raise ValueError("platform must be tiktok or youtube_shorts.")
        self.topic = _clean_string(self.topic, "topic")
        self.hook = _clean_string(self.hook, "hook")
        self.script_lines = _normalize_lines(self.script_lines)
        self.caption = _clean_string(self.caption, "caption")
        self.hashtags = _normalize_tags(self.hashtags)
        self.platform_notes = _normalize_optional_string(self.platform_notes, "platform_notes")
        self.retention_check = _clean_string(self.retention_check, "retention_check")
        self.suggested_post_time = _clean_string(self.suggested_post_time, "suggested_post_time").lower()
        if self.suggested_post_time not in VALID_POST_TIME_SLOTS:
            raise ValueError("suggested_post_time must be morning, afternoon, or evening.")
        self.style_profile = _clean_string(self.style_profile, "style_profile").lower()
        if self.style_profile not in VALID_STYLE_PROFILES:
            raise ValueError("Unknown style_profile.")
        self.created_at = _clean_string(self.created_at, "created_at")
        if not isinstance(self.safety, ContentSafetyAssessment):
            self.safety = ContentSafetyAssessment.from_mapping(self.safety)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "GeneratedContentVariant":
        return cls(
            id=str(payload.get("id") or uuid4().hex[:8]),
            source_draft_id=payload.get("source_draft_id", ""),
            platform=payload.get("platform", ""),
            topic=payload.get("topic", ""),
            hook=payload.get("hook", ""),
            script_lines=payload.get("script_lines", payload.get("script", "")),
            caption=payload.get("caption", ""),
            hashtags=payload.get("hashtags", payload.get("tags", [])),
            platform_notes=payload.get("platform_notes"),
            retention_check=payload.get("retention_check", ""),
            suggested_post_time=payload.get("suggested_post_time", "afternoon"),
            style_profile=payload.get("style_profile", "observational"),
            created_at=payload.get("created_at", _utc_now_iso()),
            safety=ContentSafetyAssessment.from_mapping(payload.get("safety")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentArtifactPackage:
    root_dir: str
    artifact_paths: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    summary_path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentBatchResult:
    items: list[GeneratedContentDraft] = field(default_factory=list)
    variants: list[GeneratedContentVariant] = field(default_factory=list)
    discarded_items: list[dict[str, Any]] = field(default_factory=list)
    package: ContentArtifactPackage | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "items": [item.to_mapping() for item in self.items],
            "variants": [item.to_mapping() for item in self.variants],
            "discarded_items": list(self.discarded_items),
            "package": self.package.to_mapping() if self.package is not None else None,
        }
