from __future__ import annotations

from typing import Any

from lumen.reasoning.assistant_context import AssistantContext


class EvidenceBuilder:
    """Builds compact, deduplicated evidence summaries from local context."""

    def __init__(self, *, limit: int = 6):
        self.limit = limit

    def build(self, *, mode: str, context: AssistantContext | None) -> list[str]:
        if context is None:
            return []

        evidence: list[str] = []
        ranked_local_evidence: list[tuple[float, str]] = []
        route = context.route or {}
        route_reason = route.get("reason")
        route_source = route.get("source")
        if route_reason:
            detail = f"Routing selected {mode} because {route_reason}."
            if route_source:
                detail += f" Source: {route_source}."
            evidence.append(detail)
        evidence.append(f"Route status: {self.route_status_label(context=context)}.")

        route_evidence = route.get("evidence") or []
        if route_evidence:
            top_evidence = route_evidence[0]
            evidence.append(
                "Routing evidence: "
                f"{top_evidence.get('label')} -> {top_evidence.get('detail')}."
            )

        active_thread = context.active_thread or {}
        top_matches = context.top_matches or []
        if top_matches:
            archive_evidence = self._archive_match_evidence(top_matches[0])
            if archive_evidence is not None:
                ranked_local_evidence.append(archive_evidence)
        archive_target_comparison = context.archive_target_comparison or {}
        if archive_target_comparison:
            comparison_evidence = self._archive_target_comparison_evidence(archive_target_comparison)
            if comparison_evidence is not None:
                ranked_local_evidence.append(comparison_evidence)

        top_interaction_matches = context.top_interaction_matches or []
        if top_interaction_matches:
            interaction_evidence = self._interaction_match_evidence(top_interaction_matches[0])
            if interaction_evidence is not None:
                ranked_local_evidence.append(interaction_evidence)
        coherence_evidence = self._cross_source_coherence_evidence(
            top_matches=top_matches,
            top_interaction_matches=top_interaction_matches,
        )
        if coherence_evidence is not None:
            ranked_local_evidence.append(coherence_evidence)

        active_prompt = active_thread.get("prompt")
        original_prompt = active_thread.get("original_prompt")
        if active_prompt:
            if original_prompt:
                ranked_local_evidence.append(
                    (
                        6.0,
                        f"Current active prompt resolves '{original_prompt}' to '{active_prompt}'.",
                    )
                )
            else:
                ranked_local_evidence.append((6.0, f"Current active prompt: {active_prompt}."))

        objective = active_thread.get("objective")
        if objective:
            prefix = (
                "Keep the active session objective in view"
                if mode == "planning"
                else "Current session objective"
            )
            ranked_local_evidence.append((5.0, f"{prefix}: {objective}."))

        for _, item in sorted(ranked_local_evidence, key=lambda pair: pair[0], reverse=True):
            evidence.append(item)

        thread_summary = active_thread.get("thread_summary")
        if thread_summary:
            evidence.append(f"Current thread summary: {thread_summary}.")

        matched_record_count = context.matched_record_count
        if matched_record_count:
            evidence.append(
                f"Review {matched_record_count} relevant archived runs before continuing."
            )
        elif context.record_count:
            evidence.append(
                f"Review {context.record_count} relevant archived runs before continuing."
            )

        if mode == "research" and context.status_counts:
            evidence.append(f"Local archive status mix: {context.status_counts}.")

        return self._dedupe_and_limit(evidence)

    def summarize_best_evidence(self, evidence: list[str]) -> str | None:
        if not evidence:
            return None
        return evidence[0]

    def summarize_local_context(self, *, context: AssistantContext | None) -> str | None:
        if context is None:
            return None

        parts: list[str] = []
        top_matches = context.top_matches or []
        if top_matches:
            best_record = (top_matches[0].get("record") or {})
            tool_id = best_record.get("tool_id")
            capability = best_record.get("capability")
            summary = str(best_record.get("summary") or "").strip()
            target_label = str(best_record.get("target_label") or "").strip()
            run_id = str(best_record.get("run_id") or "").strip()
            result_quality = str(best_record.get("result_quality") or "").strip()
            if tool_id and capability:
                prefix = "Closest semantically aligned archive run" if self._is_semantic_match(top_matches[0]) else "Closest archive run"
                detail = f"{prefix}: {tool_id}/{capability}"
                if target_label:
                    detail += f" [{target_label}]"
                if run_id:
                    detail += f" ({run_id})"
                if result_quality:
                    detail += f" [{result_quality.replace('_', ' ')}]"
                if summary:
                    detail += f" - {summary}"
                parts.append(detail)
        archive_target_comparison = context.archive_target_comparison or {}
        if archive_target_comparison:
            comparison_summary = self._archive_target_comparison_summary(archive_target_comparison)
            if comparison_summary:
                parts.append(comparison_summary)

        top_interaction_matches = context.top_interaction_matches or []
        if top_interaction_matches:
            best_interaction = top_interaction_matches[0].get("record") or {}
            prefix = (
                "Closest semantically aligned prior session prompt"
                if self._is_semantic_match(top_interaction_matches[0])
                else "Closest prior session prompt"
            )
            parts.append(
                f"{prefix}: {self._interaction_prompt_label(best_interaction)}"
            )

        active_thread = context.active_thread or {}
        active_prompt = str(active_thread.get("prompt") or "").strip()
        if active_prompt:
            parts.append(f"Active thread: {active_prompt}")

        if not parts:
            return None
        return " | ".join(parts[:3])

    def build_reasoning_frame(self, *, context: AssistantContext | None) -> dict[str, str]:
        if context is None:
            return {}

        frame: dict[str, str] = {}
        top_matches = context.top_matches or []
        top_interaction_matches = context.top_interaction_matches or []
        active_thread = context.active_thread or {}

        archive_anchor = self._archive_anchor_label(top_matches)
        interaction_anchor = self._interaction_anchor_label(top_interaction_matches)
        active_anchor = self._active_anchor_label(active_thread)
        assessment = self.assess_local_context(context=context)

        if archive_anchor:
            frame["primary_anchor"] = archive_anchor
            frame["primary_anchor_source"] = "archive"
        elif interaction_anchor:
            frame["primary_anchor"] = interaction_anchor
            frame["primary_anchor_source"] = "interaction"
        elif active_anchor:
            frame["primary_anchor"] = active_anchor
            frame["primary_anchor_source"] = "active_thread"

        if assessment == "aligned":
            if archive_anchor and interaction_anchor:
                frame["supporting_signal"] = interaction_anchor
                coherence_topic = self._cross_source_coherence_topic(
                    top_matches=top_matches,
                    top_interaction_matches=top_interaction_matches,
                )
                if coherence_topic:
                    frame["coherence_topic"] = coherence_topic
        elif assessment == "mixed":
            if archive_anchor and interaction_anchor:
                frame["tension"] = (
                    f"Archive evidence emphasizes {archive_anchor}, while prior session context emphasizes {interaction_anchor}"
                )
            elif archive_anchor and active_anchor:
                frame["tension"] = (
                    f"Archive evidence emphasizes {archive_anchor}, while the active thread emphasizes {active_anchor}"
                )

        return frame

    def build_working_hypothesis(
        self,
        *,
        mode: str,
        context: AssistantContext | None,
    ) -> str | None:
        if context is None:
            return None

        frame = self.build_reasoning_frame(context=context)
        assessment = self.assess_local_context(context=context)
        anchor = frame.get("primary_anchor")
        supporting_signal = frame.get("supporting_signal")
        tension = frame.get("tension")

        if assessment == "aligned" and anchor:
            if supporting_signal:
                return (
                    f"The best working hypothesis is to treat {anchor} as the main local anchor, "
                    f"supported by {supporting_signal}."
                )
            return f"The best working hypothesis is to treat {anchor} as the main local anchor."
        if assessment == "mixed" and tension:
            return (
                f"The best working hypothesis is that the next {mode} step should reconcile this tension: "
                f"{tension}."
            )
        if anchor:
            return f"The best working hypothesis is to use {anchor} as the provisional anchor for the next {mode} step."
        return None

    def synthesize_interpretation(
        self,
        *,
        mode: str,
        context: AssistantContext | None,
    ) -> str | None:
        if context is None:
            return None

        local_context_summary = self.summarize_local_context(context=context)
        local_context_assessment = self.assess_local_context(context=context)
        grounding_strength = self.grounding_strength(context=context)
        reasoning_frame = self.build_reasoning_frame(context=context)
        route = context.route or {}
        route_strength = str(route.get("strength") or "").strip()
        route_quality = self.route_quality_label(context=context)

        if not local_context_summary:
            if grounding_strength == "low":
                return f"Local evidence for this {mode} response is thin, so treat the result as exploratory."
            return f"Only limited local context is available for this {mode} response."

        if local_context_assessment == "aligned":
            anchor = reasoning_frame.get("primary_anchor")
            return (
                f"Local evidence is aligned and points to a consistent {mode} direction"
                + (f" anchored by {anchor}: " if anchor else ": ")
                + f"{local_context_summary}."
            )
        if local_context_assessment == "mixed":
            tension = reasoning_frame.get("tension")
            return (
                f"Local evidence is mixed, so this {mode} response should first reconcile competing signals: "
                + (f"{tension}. " if tension else "")
                + f"{local_context_summary}."
            )
        if route_quality == "weak" and grounding_strength in {"high", "medium"}:
            anchor = reasoning_frame.get("primary_anchor")
            return (
                f"Local evidence is fairly strong for this {mode} response, but the winning route remains comparatively weak"
                + (f" around {anchor}: " if anchor else ": ")
                + f"{local_context_summary}."
            )
        if grounding_strength == "high" or route_strength == "high":
            anchor = reasoning_frame.get("primary_anchor")
            return (
                f"The strongest local signal for this {mode} response is"
                + (f" {anchor}: " if anchor else ": ")
                + f"{local_context_summary}."
            )
        if route_quality == "weak":
            return (
                f"Local evidence offers a partial direction for this {mode} response, but the winning route remains comparatively weak: "
                f"{local_context_summary}."
            )
        return (
            f"Use this partial local context as a provisional anchor for the {mode} response: "
            f"{local_context_summary}."
        )

    def assess_local_context(self, *, context: AssistantContext | None) -> str | None:
        if context is None:
            return None

        archive_summary = self._archive_summary_text(context.top_matches or [])
        interaction_summary = self._interaction_summary_text(context.top_interaction_matches or [])
        if archive_summary and interaction_summary:
            archive_tokens = self._meaningful_tokens(archive_summary)
            interaction_tokens = self._meaningful_tokens(interaction_summary)
            overlap = archive_tokens & interaction_tokens
            if overlap:
                return "aligned"
            return "mixed"
        if archive_summary or interaction_summary or (context.active_thread or {}).get("prompt"):
            return "partial"
        return None

    def grounding_strength(self, *, context: AssistantContext | None) -> str:
        if context is None:
            return "low"

        route = context.route or {}
        route_confidence = float(route.get("confidence") or 0.0)
        route_normalized_score = self.route_normalized_score(context=context)
        archive_score = self._top_match_score(context.top_matches or [])
        interaction_score = self._top_match_score(context.top_interaction_matches or [])

        strong_route = route_normalized_score is None or route_normalized_score >= 1.8
        supported_route = route_normalized_score is None or route_normalized_score >= 1.7
        medium_route = route_normalized_score is None or route_normalized_score >= 1.65
        baseline_route = route_normalized_score is not None and route_normalized_score >= 1.45

        if route_confidence >= 0.8 and strong_route and (archive_score >= 5 or interaction_score >= 4):
            return "high"
        if self._cross_source_semantic_coherence(context=context):
            return "high" if route_confidence >= 0.7 and supported_route else "medium"
        if self._semantic_support_strength(context=context) >= 2 and route_confidence >= 0.7 and medium_route:
            return "high"
        if route_confidence >= 0.6 or baseline_route or archive_score >= 4 or interaction_score >= 3:
            return "medium"
        if self._semantic_support_strength(context=context) >= 1:
            return "medium"
        return "low"

    def route_normalized_score(self, *, context: AssistantContext | None) -> float | None:
        if context is None:
            return None
        route = context.route or {}
        decision_summary = route.get("decision_summary") or {}
        selected = decision_summary.get("selected") or {}
        raw_score = selected.get("normalized_score")
        if raw_score is None:
            return None
        return float(raw_score)

    def route_quality_label(self, *, context: AssistantContext | None) -> str:
        score = self.route_normalized_score(context=context)
        if score is None:
            return "unknown"
        if score >= 1.8:
            return "strong"
        if score >= 1.45:
            return "supported"
        return "weak"

    def route_status_label(self, *, context: AssistantContext | None) -> str:
        if context is None:
            return "stable"
        route = context.route or {}
        ambiguity = route.get("ambiguity") or {}
        if ambiguity.get("ambiguous"):
            return "under_tension"
        if self.route_quality_label(context=context) == "weak":
            return "weakened"
        return "stable"

    def _archive_match_evidence(self, match: dict[str, Any]) -> tuple[float, str] | None:
        best_record = match.get("record") or {}
        tool_id = best_record.get("tool_id")
        capability = best_record.get("capability")
        if not tool_id or not capability:
            return None
        summary = str(best_record.get("summary") or "").strip()
        target_label = str(best_record.get("target_label") or "").strip()
        run_id = str(best_record.get("run_id") or "").strip()
        result_quality = str(best_record.get("result_quality") or "").strip()
        score = float(match.get("score") or 0)
        detail = "Closest archive match"
        if self._is_semantic_match(match):
            detail += " (semantic)"
        detail += f": {tool_id}/{capability}"
        if target_label:
            detail += f" [{target_label}]"
        if run_id:
            detail += f" ({run_id})"
        if result_quality:
            detail += f" [{result_quality.replace('_', ' ')}]"
        if summary:
            detail += f" - {summary}"
        if score:
            detail += f" (score={int(score) if score.is_integer() else score})"
        return (20.0 + score, detail + ".")

    @staticmethod
    def _archive_target_comparison_evidence(comparison: dict[str, Any]) -> tuple[float, str] | None:
        target_label = str(comparison.get("target_label") or "").strip()
        trend_summary = str(comparison.get("trend_summary") or "").strip()
        if not target_label or not trend_summary:
            return None
        run_count = int(comparison.get("run_count") or 0)
        return (
            17.5 + min(run_count, 3),
            f"Same-capability archive trend for {target_label}: {trend_summary}.",
        )

    @staticmethod
    def _archive_target_comparison_summary(comparison: dict[str, Any]) -> str | None:
        target_label = str(comparison.get("target_label") or "").strip()
        trend_summary = str(comparison.get("trend_summary") or "").strip()
        if not target_label or not trend_summary:
            return None
        return f"Same-capability trend for {target_label}: {trend_summary}"

    def _archive_anchor_label(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return ""
        best_record = matches[0].get("record") or {}
        tool_id = best_record.get("tool_id")
        capability = best_record.get("capability")
        summary = str(best_record.get("summary") or "").strip()
        if tool_id and capability and summary:
            return f"{tool_id}/{capability} ({summary})"
        if tool_id and capability:
            return f"{tool_id}/{capability}"
        return summary

    def _archive_summary_text(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return ""
        best_record = matches[0].get("record") or {}
        return " ".join(
            str(part).strip()
            for part in (
                best_record.get("tool_id"),
                best_record.get("capability"),
                best_record.get("summary"),
            )
            if str(part or "").strip()
        )

    def _interaction_match_evidence(self, match: dict[str, Any]) -> tuple[float, str] | None:
        best_interaction = match.get("record") or {}
        label = self._interaction_prompt_label(best_interaction)
        if not label:
            return None
        score = float(match.get("score") or 0)
        prefix = "Closest prior session prompt"
        if self._is_semantic_match(match):
            prefix += " (semantic)"
        return (
            18.0 + score,
            f"{prefix}: {label}.",
        )

    def _interaction_summary_text(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return ""
        best_interaction = matches[0].get("record") or {}
        return self._interaction_prompt_label(best_interaction)

    def _interaction_anchor_label(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return ""
        best_interaction = matches[0].get("record") or {}
        return self._interaction_prompt_label(best_interaction)

    @staticmethod
    def _active_anchor_label(active_thread: dict[str, Any]) -> str:
        return str(active_thread.get("prompt") or "").strip()

    @staticmethod
    def _top_match_score(matches: list[dict[str, Any]]) -> float:
        if not matches:
            return 0.0
        return float(matches[0].get("score") or 0.0)

    def _semantic_support_strength(self, *, context: AssistantContext | None) -> int:
        if context is None:
            return 0
        strength = 0
        top_matches = context.top_matches or []
        if top_matches and self._is_semantic_match(top_matches[0]):
            strength += 1
        top_interaction_matches = context.top_interaction_matches or []
        if top_interaction_matches and self._is_semantic_match(top_interaction_matches[0]):
            strength += 1
        return strength

    def _cross_source_coherence_evidence(
        self,
        *,
        top_matches: list[dict[str, Any]],
        top_interaction_matches: list[dict[str, Any]],
    ) -> tuple[float, str] | None:
        coherence_topic = self._cross_source_coherence_topic(
            top_matches=top_matches,
            top_interaction_matches=top_interaction_matches,
        )
        if not coherence_topic:
            return None
        return (
            19.5,
            f"Archive and prior session context are semantically coherent around {coherence_topic}.",
        )

    def _cross_source_semantic_coherence(self, *, context: AssistantContext | None) -> bool:
        if context is None:
            return False
        return bool(
            self._cross_source_coherence_topic(
                top_matches=context.top_matches or [],
                top_interaction_matches=context.top_interaction_matches or [],
            )
        )

    def _cross_source_coherence_topic(
        self,
        *,
        top_matches: list[dict[str, Any]],
        top_interaction_matches: list[dict[str, Any]],
    ) -> str | None:
        if not top_matches or not top_interaction_matches:
            return None
        if not self._is_semantic_match(top_matches[0]) or not self._is_semantic_match(top_interaction_matches[0]):
            return None
        archive_tokens = self._meaningful_tokens(self._archive_summary_text(top_matches))
        interaction_tokens = self._meaningful_tokens(self._interaction_summary_text(top_interaction_matches))
        overlap = sorted(archive_tokens & interaction_tokens)
        if not overlap:
            return None
        return " ".join(overlap[:3])

    @staticmethod
    def _is_semantic_match(match: dict[str, Any]) -> bool:
        matched_fields = match.get("matched_fields")
        if isinstance(matched_fields, list):
            return "semantic" in matched_fields
        return False

    @staticmethod
    def _interaction_prompt_label(record: dict[str, Any]) -> str:
        prompt_view = record.get("prompt_view") or {}
        canonical_prompt = (
            prompt_view.get("canonical_prompt")
            or record.get("resolved_prompt")
            or record.get("prompt")
            or "<unknown prompt>"
        )
        original_prompt = prompt_view.get("original_prompt")
        rewritten = bool(prompt_view.get("rewritten"))
        if rewritten and original_prompt and original_prompt != canonical_prompt:
            return f"{canonical_prompt} (from '{original_prompt}')"
        return str(canonical_prompt)

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "what",
            "about",
            "create",
            "summary",
            "response",
            "current",
            "closest",
            "prior",
            "session",
            "prompt",
            "run",
            "local",
            "analysis",
            "kit",
        }
        tokens = {
            token.strip(".,:;!?()[]{}'\"").lower()
            for token in text.split()
        }
        return {token for token in tokens if len(token) > 2 and token not in stop_words}

    def _dedupe_and_limit(self, items: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = " ".join(item.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
            if len(results) >= self.limit:
                break
        return results
