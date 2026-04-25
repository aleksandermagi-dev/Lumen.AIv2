from __future__ import annotations

from datetime import UTC, datetime

from lumen.reasoning.assistant_context import AssistantContext
from lumen.reasoning.evidence_builder import EvidenceBuilder
from lumen.reasoning.pipeline_models import EvidenceUnit, ValidationTargetView


class EvidenceLedgerBuilder:
    """Owns evidence-ledger construction, aging, reaffirmation, and reference selection."""

    def __init__(self, *, evidence_builder: EvidenceBuilder) -> None:
        self.evidence_builder = evidence_builder

    def build(
        self,
        *,
        assistant_context: AssistantContext,
        targets: list[ValidationTargetView],
        contradiction_flags: list[str],
    ) -> list[EvidenceUnit]:
        ledger: list[EvidenceUnit] = []
        active_thread = assistant_context.active_thread or {}
        if active_thread.get("prompt"):
            active_topic = str(active_thread.get("normalized_topic") or "").strip() or None
            age_bucket, decay_factor, reaffirmed = self._age_profile(
                created_at=None,
                source="active_thread",
                topic=active_topic,
                assistant_context=assistant_context,
            )
            ledger.append(
                EvidenceUnit(
                    evidence_id="active_thread:0",
                    source="active_thread",
                    summary=str(active_thread.get("prompt")),
                    strength="supported",
                    authority_score=1.0,
                    topic=active_topic,
                    created_at=None,
                    age_bucket=age_bucket,
                    decay_factor=decay_factor,
                    reaffirmed=reaffirmed,
                    supports=["continuity"],
                    contradicts=list(contradiction_flags),
                )
            )
        if assistant_context.top_matches:
            ledger.append(
                self._archive_unit(
                    top_archive=assistant_context.top_matches[0],
                    assistant_context=assistant_context,
                    contradiction_flags=contradiction_flags,
                )
            )
        if assistant_context.top_interaction_matches:
            ledger.append(
                self._interaction_unit(
                    top_interaction=assistant_context.top_interaction_matches[0],
                    assistant_context=assistant_context,
                    contradiction_flags=contradiction_flags,
                )
            )
        for target in targets:
            if target.quality != "missing":
                continue
            ledger.append(
                EvidenceUnit(
                    evidence_id=f"{target.source}:missing",
                    source=target.source,
                    summary=target.summary or f"{target.source} context is missing",
                    strength="missing",
                )
            )
        return ledger

    def select_references(
        self,
        *,
        reasoning_frame: dict[str, str],
        evidence_ledger: list[EvidenceUnit],
    ) -> dict[str, object]:
        source_to_id = {unit.source: unit.evidence_id for unit in evidence_ledger}
        anchor_source = str(reasoning_frame.get("primary_anchor_source") or "").strip()
        anchor_evidence_id = source_to_id.get(anchor_source)
        for unit in evidence_ledger:
            unit.selected_as_anchor = unit.evidence_id == anchor_evidence_id
        supporting_evidence_id = None
        if reasoning_frame.get("supporting_signal"):
            if anchor_source == "archive":
                supporting_evidence_id = source_to_id.get("interaction")
            elif anchor_source == "interaction":
                supporting_evidence_id = source_to_id.get("archive")
        tension_evidence_ids: list[str] = []
        if reasoning_frame.get("tension"):
            for source in ("archive", "interaction", "active_thread"):
                evidence_id = source_to_id.get(source)
                if evidence_id and evidence_id != anchor_evidence_id:
                    tension_evidence_ids.append(evidence_id)
        return {
            "anchor_evidence_id": anchor_evidence_id,
            "supporting_evidence_id": supporting_evidence_id,
            "tension_evidence_ids": tension_evidence_ids[:2],
        }

    def _archive_unit(
        self,
        *,
        top_archive: dict[str, object],
        assistant_context: AssistantContext,
        contradiction_flags: list[str],
    ) -> EvidenceUnit:
        archive_record = top_archive.get("record") or {}
        archive_summary = str(archive_record.get("summary") or "").strip()
        archive_topic = " ".join(self.evidence_builder._meaningful_tokens(archive_summary)) or None
        age_bucket, decay_factor, reaffirmed = self._age_profile(
            created_at=archive_record.get("created_at"),
            source="archive",
            topic=archive_topic,
            assistant_context=assistant_context,
        )
        raw_score = float(top_archive.get("score") or 0.0)
        authority_score = round(raw_score * decay_factor, 4)
        return EvidenceUnit(
            evidence_id="archive:0",
            source="archive",
            summary=archive_summary or str(archive_record.get("capability") or "archive context"),
            strength=self._strength_from_score(authority_score),
            authority_score=authority_score,
            topic=archive_topic,
            created_at=str(archive_record.get("created_at") or "").strip() or None,
            age_bucket=age_bucket,
            decay_factor=decay_factor,
            reaffirmed=reaffirmed,
            supports=["archive_validation"],
            contradicts=list(contradiction_flags),
            matched_fields=[
                str(item) for item in (top_archive.get("matched_fields") or []) if str(item).strip()
            ],
            score_breakdown=(
                {
                    str(key): int(value)
                    for key, value in (top_archive.get("score_breakdown") or {}).items()
                }
                if isinstance(top_archive.get("score_breakdown"), dict)
                else None
            ),
        )

    def _interaction_unit(
        self,
        *,
        top_interaction: dict[str, object],
        assistant_context: AssistantContext,
        contradiction_flags: list[str],
    ) -> EvidenceUnit:
        interaction_record = top_interaction.get("record") or {}
        interaction_summary = self.evidence_builder._interaction_prompt_label(interaction_record)
        interaction_topic = " ".join(self.evidence_builder._meaningful_tokens(interaction_summary)) or None
        age_bucket, decay_factor, reaffirmed = self._age_profile(
            created_at=interaction_record.get("created_at"),
            source="interaction",
            topic=interaction_topic,
            assistant_context=assistant_context,
        )
        raw_score = float(top_interaction.get("score") or 0.0)
        authority_score = round(raw_score * decay_factor, 4)
        return EvidenceUnit(
            evidence_id="interaction:0",
            source="interaction",
            summary=interaction_summary,
            strength=self._strength_from_score(authority_score),
            authority_score=authority_score,
            topic=interaction_topic,
            created_at=str(interaction_record.get("created_at") or "").strip() or None,
            age_bucket=age_bucket,
            decay_factor=decay_factor,
            reaffirmed=reaffirmed,
            supports=["interaction_validation"],
            contradicts=list(contradiction_flags),
            matched_fields=[
                str(item) for item in (top_interaction.get("matched_fields") or []) if str(item).strip()
            ],
            score_breakdown=(
                {
                    str(key): int(value)
                    for key, value in (top_interaction.get("score_breakdown") or {}).items()
                }
                if isinstance(top_interaction.get("score_breakdown"), dict)
                else None
            ),
        )

    @staticmethod
    def _strength_from_score(score: float) -> str:
        if score >= 7:
            return "strong"
        if score >= 4:
            return "supported"
        if score > 0:
            return "light"
        return "missing"

    def _age_profile(
        self,
        *,
        created_at: object,
        source: str,
        topic: str | None,
        assistant_context: AssistantContext,
    ) -> tuple[str | None, float, bool]:
        if not created_at:
            return (None, 1.0, False)
        try:
            created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return (None, 1.0, False)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=UTC)
        age_days = max((datetime.now(UTC) - created_dt).days, 0)
        age_bucket, base_decay = self._source_age_thresholds(source=source, age_days=age_days)
        reaffirmed = self._is_reaffirmed(topic=topic, assistant_context=assistant_context)
        if reaffirmed and source == "interaction":
            return ("recent", 1.0, True)
        if reaffirmed and base_decay < 1.0:
            return (age_bucket, min(base_decay + 0.2, 1.0), True)
        return (age_bucket, base_decay, reaffirmed)

    @staticmethod
    def _source_age_thresholds(*, source: str, age_days: int) -> tuple[str, float]:
        if source == "interaction":
            if age_days >= 365:
                return ("old", 0.55)
            if age_days >= 120:
                return ("stale", 0.75)
            if age_days >= 30:
                return ("aging", 0.9)
            return ("recent", 1.0)
        if age_days >= 180:
            return ("old", 0.45)
        if age_days >= 60:
            return ("stale", 0.65)
        if age_days >= 14:
            return ("aging", 0.85)
        return ("recent", 1.0)

    def _is_reaffirmed(
        self,
        *,
        topic: str | None,
        assistant_context: AssistantContext,
    ) -> bool:
        if not topic:
            return False
        topic_tokens = self.evidence_builder._meaningful_tokens(topic)
        if not topic_tokens:
            return False
        active_thread = assistant_context.active_thread or {}
        active_topic = str(active_thread.get("normalized_topic") or active_thread.get("prompt") or "").strip()
        if active_topic and (topic_tokens & self.evidence_builder._meaningful_tokens(active_topic)):
            return True
        coherence_topic = self.evidence_builder._cross_source_coherence_topic(
            top_matches=assistant_context.top_matches or [],
            top_interaction_matches=assistant_context.top_interaction_matches or [],
        )
        if coherence_topic and (topic_tokens & self.evidence_builder._meaningful_tokens(coherence_topic)):
            return True
        return False
