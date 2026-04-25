from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult, RetrievedMemory
from lumen.reasoning.memory_ranking_signals import MemoryRankingSignals


@dataclass(slots=True, frozen=True)
class MemoryContextCandidateDecision:
    source: str
    memory_kind: str
    label: str
    summary: str
    relevance_score: float
    domain_match: float
    recency_weight: float
    confidence: float
    intent_alignment: float
    accepted: bool
    rationale: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "memory_kind": self.memory_kind,
            "label": self.label,
            "summary": self.summary,
            "relevance_score": round(float(self.relevance_score), 4),
            "domain_match": round(float(self.domain_match), 4),
            "recency_weight": round(float(self.recency_weight), 4),
            "confidence": round(float(self.confidence), 4),
            "intent_alignment": round(float(self.intent_alignment), 4),
            "accepted": self.accepted,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class MemoryContextDecision:
    selected: tuple[MemoryContextCandidateDecision, ...] = ()
    rejected: tuple[MemoryContextCandidateDecision, ...] = ()
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "rejected": [item.to_dict() for item in self.rejected],
            "rationale": self.rationale,
        }

    def filtered_result(self, retrieval: MemoryRetrievalResult) -> MemoryRetrievalResult:
        selected_lookup = {
            (item.source, item.memory_kind, item.label, item.summary)
            for item in self.selected
        }
        filtered_selected = [
            item for item in retrieval.selected
            if (item.source, item.memory_kind, item.label, item.summary) in selected_lookup
        ]
        diagnostics = dict(retrieval.diagnostics or {})
        diagnostics["memory_context_classifier"] = self.to_dict()
        return MemoryRetrievalResult(
            query=retrieval.query,
            selected=filtered_selected,
            memory_reply_hint=(
                retrieval.memory_reply_hint
                if filtered_selected or retrieval.recall_prompt
                else None
            ),
            recall_prompt=retrieval.recall_prompt,
            project_return_prompt=retrieval.project_return_prompt,
            project_reply_hint=retrieval.project_reply_hint if filtered_selected else None,
            diagnostics=diagnostics,
        )


class MemoryContextClassifier:
    """Filter retrieved memory so only aligned, timely context reaches synthesis."""

    _TECHNICAL_DOMAINS = {"technical_engineering", "research_investigation", "planning_strategy", "problem_solving"}
    _PERSONAL_MEMORY_KINDS = {"durable_user_memory", "profile", "preference"}
    _EXPLICIT_MEMORY_CUES = (
        "what do you remember",
        "what do we remember",
        "remember about",
        "what do you know about my",
        "what do you have on me",
        "what have i told you",
        "what have you saved",
    )
    _PERSONAL_RELEVANCE_CUES = (
        "about me",
        "for me",
        "with me",
        "my ",
        "i like",
        "i prefer",
        "i usually",
        "i tend",
        "you know me",
        "as you know",
        "my preference",
        "my style",
    )

    def classify(
        self,
        *,
        retrieval: MemoryRetrievalResult | None,
        route_mode: str,
        intent_domain: str | None,
        prompt: str,
    ) -> MemoryContextDecision:
        if retrieval is None or not retrieval.selected:
            return MemoryContextDecision(rationale="No retrieved memory needed classification.")

        normalized_prompt = " ".join(str(prompt or "").lower().split())
        explicit_memory_prompt = self._is_explicit_memory_prompt(normalized_prompt)
        selected: list[MemoryContextCandidateDecision] = []
        rejected: list[MemoryContextCandidateDecision] = []
        for item in retrieval.selected:
            decision = self._score_candidate(
                item=item,
                route_mode=route_mode,
                intent_domain=intent_domain,
                normalized_prompt=normalized_prompt,
                explicit_memory_prompt=explicit_memory_prompt,
            )
            if decision.accepted:
                selected.append(decision)
            else:
                rejected.append(decision)
        selected.sort(key=lambda item: (item.relevance_score, item.confidence, item.intent_alignment), reverse=True)
        selected = self._cap_personal_memory(
            selected=selected,
            rejected=rejected,
            explicit_memory_prompt=explicit_memory_prompt,
        )
        selected = selected[:2]
        rationale = (
            "Filtered retrieved memory to keep only high-signal context aligned with the current route and intent."
            if selected
            else "Rejected retrieved memory because it was too weak, stale, or misaligned for the current turn."
        )
        return MemoryContextDecision(
            selected=tuple(selected),
            rejected=tuple(rejected),
            rationale=rationale,
        )

    def _score_candidate(
        self,
        *,
        item: RetrievedMemory,
        route_mode: str,
        intent_domain: str | None,
        normalized_prompt: str,
        explicit_memory_prompt: bool,
    ) -> MemoryContextCandidateDecision:
        domain_match = self._domain_match(item=item, route_mode=route_mode, intent_domain=intent_domain)
        recency_weight = self._recency_weight(item.metadata)
        confidence = self._memory_confidence(item=item)
        intent_alignment = self._intent_alignment(item=item, normalized_prompt=normalized_prompt)
        composite = (
            (item.relevance * 0.40)
            + (domain_match * 0.22)
            + (recency_weight * 0.12)
            + (confidence * 0.14)
            + (intent_alignment * 0.12)
        )
        strict_mode = str(route_mode or "").strip() in {"planning", "research", "tool"} or str(intent_domain or "").strip() in self._TECHNICAL_DOMAINS
        accepted = composite >= (0.64 if strict_mode else 0.54)
        if strict_mode and item.memory_kind == "ephemeral_context" and item.relevance < 0.85:
            accepted = False
        age_bucket = str(item.metadata.get("age_bucket") or "").strip()
        reaffirmed = bool(item.metadata.get("reaffirmed"))
        contradiction_status = str(item.metadata.get("contradiction_status") or "").strip().lower()
        generic_penalty = float(item.metadata.get("generic_label_penalty") or 0.0)
        mentions_contradiction = any(
            cue in normalized_prompt for cue in ("contradiction", "conflict", "mismatch", "disagree", "inconsistency")
        )
        if strict_mode and age_bucket in {"stale", "old"} and not reaffirmed and item.relevance < 0.88:
            accepted = False
        if contradiction_status == "potential_conflict" and not mentions_contradiction and item.relevance < 0.82:
            accepted = False
        if strict_mode and generic_penalty >= 0.08 and item.relevance < 0.86:
            accepted = False
        exposure_policy = "standard_context"
        if self._is_personal_memory(item):
            exposure_policy = self._personal_memory_exposure_policy(
                item=item,
                route_mode=route_mode,
                intent_domain=intent_domain,
                normalized_prompt=normalized_prompt,
                explicit_memory_prompt=explicit_memory_prompt,
                intent_alignment=intent_alignment,
            )
            if exposure_policy == "blocked_noise":
                accepted = False
            elif exposure_policy == "explicit_recall":
                accepted = accepted or item.relevance >= 0.5
        rationale = (
            "Admitted because the memory is relevant, aligned, and recent enough for this turn."
            if accepted
            else "Rejected because the memory is too weak, too generic, or mismatched for this turn."
        )
        if self._is_personal_memory(item) and exposure_policy == "blocked_noise":
            rationale = "Rejected because personal memory was not explicitly asked for or clearly useful to this turn."
        return MemoryContextCandidateDecision(
            source=item.source,
            memory_kind=item.memory_kind,
            label=item.label,
            summary=item.summary,
            relevance_score=item.relevance,
            domain_match=domain_match,
            recency_weight=recency_weight,
            confidence=confidence,
            intent_alignment=intent_alignment,
            accepted=accepted,
            rationale=rationale,
            metadata={**dict(item.metadata), "memory_exposure_policy": exposure_policy},
        )

    def _domain_match(self, *, item: RetrievedMemory, route_mode: str, intent_domain: str | None) -> float:
        route_mode = str(route_mode or "").strip()
        intent_domain = str(intent_domain or "").strip()
        if item.memory_kind == "active_project_memory":
            if route_mode in {"planning", "research", "tool"} or intent_domain in self._TECHNICAL_DOMAINS:
                return 0.95
            return 0.65
        if item.memory_kind == "ephemeral_context":
            if route_mode == "conversation" and intent_domain not in self._TECHNICAL_DOMAINS:
                return 0.72
            return 0.35
        if item.source == "graph_memory":
            return 0.74 if intent_domain in self._TECHNICAL_DOMAINS else 0.58
        return 0.56

    @staticmethod
    def _memory_confidence(*, item: RetrievedMemory) -> float:
        source_reliability = float(item.metadata.get("source_reliability") or 0.0)
        confidence_hint = float(item.metadata.get("confidence_hint") or 0.0)
        if source_reliability or confidence_hint:
            return min(0.96, max(0.35, (source_reliability * 0.55) + (confidence_hint * 0.45)))
        if item.memory_kind == "active_project_memory":
            return 0.82
        if item.source == "graph_memory":
            return 0.7
        if item.memory_kind == "ephemeral_context":
            return 0.52
        return 0.6

    @staticmethod
    def _intent_alignment(*, item: RetrievedMemory, normalized_prompt: str) -> float:
        if not normalized_prompt:
            return 0.5
        label_text = " ".join((item.label or "", item.summary or "")).lower()
        prompt_tokens = {token for token in normalized_prompt.split() if token}
        label_tokens = {token for token in label_text.split() if token}
        overlap = prompt_tokens & label_tokens
        if overlap:
            return min(0.95, 0.55 + (len(overlap) * 0.1))
        return 0.35

    @staticmethod
    def _recency_weight(metadata: dict[str, object]) -> float:
        age_bucket = str(metadata.get("age_bucket") or "").strip()
        if age_bucket:
            if age_bucket in {"active", "recent"}:
                return 0.95
            if age_bucket == "aging":
                return 0.7 if metadata.get("reaffirmed") else 0.62
            if age_bucket == "stale":
                return 0.62 if metadata.get("reaffirmed") else 0.38
            if age_bucket == "old":
                return 0.52 if metadata.get("reaffirmed") else 0.26
            if age_bucket == "unknown":
                return 0.55
        created_at = str(metadata.get("created_at") or "").strip()
        if not created_at:
            return 0.55
        try:
            moment = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.55
        age_days = max(0.0, (datetime.now(UTC) - moment.astimezone(UTC)).total_seconds() / 86400.0)
        if age_days <= 3:
            return 0.95
        if age_days <= 14:
            return 0.75
        if age_days <= 45:
            return 0.58
        return 0.4

    @classmethod
    def _cap_personal_memory(
        cls,
        *,
        selected: list[MemoryContextCandidateDecision],
        rejected: list[MemoryContextCandidateDecision],
        explicit_memory_prompt: bool,
    ) -> list[MemoryContextCandidateDecision]:
        if explicit_memory_prompt:
            return selected
        capped: list[MemoryContextCandidateDecision] = []
        personal_seen = False
        for item in selected:
            if item.memory_kind in cls._PERSONAL_MEMORY_KINDS or item.source == "personal_memory":
                if personal_seen:
                    rejected.append(
                        MemoryContextCandidateDecision(
                            source=item.source,
                            memory_kind=item.memory_kind,
                            label=item.label,
                            summary=item.summary,
                            relevance_score=item.relevance_score,
                            domain_match=item.domain_match,
                            recency_weight=item.recency_weight,
                            confidence=item.confidence,
                            intent_alignment=item.intent_alignment,
                            accepted=False,
                            rationale="Rejected because ordinary chat should use at most one clearly useful personal memory.",
                            metadata={**dict(item.metadata), "memory_exposure_policy": "capped_familiarity"},
                        )
                    )
                    continue
                personal_seen = True
            capped.append(item)
        return capped

    @classmethod
    def _is_personal_memory(cls, item: RetrievedMemory) -> bool:
        return item.source == "personal_memory" or item.memory_kind in cls._PERSONAL_MEMORY_KINDS

    @classmethod
    def _is_explicit_memory_prompt(cls, normalized_prompt: str) -> bool:
        return any(cue in normalized_prompt for cue in cls._EXPLICIT_MEMORY_CUES)

    @classmethod
    def _has_personal_relevance_cue(cls, normalized_prompt: str) -> bool:
        return any(cue in f" {normalized_prompt} " for cue in cls._PERSONAL_RELEVANCE_CUES)

    @classmethod
    def _personal_memory_exposure_policy(
        cls,
        *,
        item: RetrievedMemory,
        route_mode: str,
        intent_domain: str | None,
        normalized_prompt: str,
        explicit_memory_prompt: bool,
        intent_alignment: float,
    ) -> str:
        if explicit_memory_prompt:
            return "explicit_recall"
        focus_overlap = int(item.metadata.get("focus_overlap") or 0)
        personal_cue = cls._has_personal_relevance_cue(normalized_prompt)
        route_mode = str(route_mode or "").strip()
        intent_domain = str(intent_domain or "").strip()
        if route_mode == "conversation" and intent_domain not in cls._TECHNICAL_DOMAINS:
            if personal_cue and (intent_alignment >= 0.6 or focus_overlap >= 1 or item.relevance >= 0.76):
                return "bounded_familiarity"
            if intent_alignment >= 0.78 or focus_overlap >= 2:
                return "bounded_familiarity"
            return "blocked_noise"
        if item.relevance >= 0.88 and intent_alignment >= 0.75:
            return "task_relevant"
        return "blocked_noise"
