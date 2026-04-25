from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ScriptedTurnExpectation:
    prompt: str
    expected_mode: str | None = None
    expected_kind_prefix: str | None = None
    forbidden_modes: tuple[str, ...] = ()
    required_boundary_signal: str | None = None
    required_work_thread_intent: str | None = None
    require_project_context: bool | None = None
    max_personal_memory: int | None = None
    allow_memory_recall: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "expected_mode": self.expected_mode,
            "expected_kind_prefix": self.expected_kind_prefix,
            "forbidden_modes": list(self.forbidden_modes),
            "required_boundary_signal": self.required_boundary_signal,
            "required_work_thread_intent": self.required_work_thread_intent,
            "require_project_context": self.require_project_context,
            "max_personal_memory": self.max_personal_memory,
            "allow_memory_recall": self.allow_memory_recall,
        }


@dataclass(slots=True, frozen=True)
class ScriptedConversationScenario:
    name: str
    description: str
    turns: tuple[ScriptedTurnExpectation, ...]
    style_mode: str = "default"
    min_turns: int = 20

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "style_mode": self.style_mode,
            "min_turns": self.min_turns,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(slots=True, frozen=True)
class ScriptedConversationFinding:
    turn_index: int | None
    category: str
    severity: str
    message: str
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True, frozen=True)
class ScriptedConversationEvaluation:
    scenario_name: str
    evaluated_turns: int
    expected_turns: int
    score: float
    judgment: str
    findings: tuple[ScriptedConversationFinding, ...]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "evaluated_turns": self.evaluated_turns,
            "expected_turns": self.expected_turns,
            "score": round(float(self.score), 4),
            "judgment": self.judgment,
            "metrics": dict(self.metrics),
            "findings": [finding.to_dict() for finding in self.findings],
        }


class LongConversationEvaluator:
    """Scores scripted 20-30 turn local conversations for drift and integration quality."""

    SCAFFOLD_TERMS = (
        "best first read",
        "provisionally",
        "best next check",
        "route validation",
        "route_status",
        "support_status",
        "validation plan",
    )

    def evaluate(
        self,
        *,
        scenario: ScriptedConversationScenario,
        records: list[dict[str, Any]],
    ) -> ScriptedConversationEvaluation:
        findings: list[ScriptedConversationFinding] = []
        expected_turns = len(scenario.turns)
        evaluated_turns = len(records)
        if expected_turns < scenario.min_turns:
            findings.append(
                ScriptedConversationFinding(
                    turn_index=None,
                    category="scenario_shape",
                    severity="incorrect",
                    message="Scenario is shorter than the required long-conversation floor.",
                    evidence={"expected_turns": expected_turns, "min_turns": scenario.min_turns},
                )
            )
        if evaluated_turns < expected_turns:
            findings.append(
                ScriptedConversationFinding(
                    turn_index=None,
                    category="conversation_completion",
                    severity="incorrect",
                    message="Conversation did not produce a response for every scripted turn.",
                    evidence={"evaluated_turns": evaluated_turns, "expected_turns": expected_turns},
                )
            )

        style_mismatches = 0
        route_passes = 0
        memory_violations = 0
        repetition_violations = 0
        scaffold_violations = 0
        project_passes = 0
        work_thread_passes = 0
        for index, expectation in enumerate(scenario.turns):
            record = records[index] if index < len(records) else {}
            response = self._response(record)
            mode = self._text(response.get("mode") or record.get("mode"))
            kind = self._text(response.get("kind") or record.get("kind"))
            if expectation.expected_mode and mode == expectation.expected_mode:
                route_passes += 1
            if expectation.expected_mode and mode != expectation.expected_mode:
                findings.append(
                    self._finding(
                        index=index,
                        category="route_drift",
                        message="Turn selected the wrong mode for the scripted prompt.",
                        evidence={"prompt": expectation.prompt, "expected_mode": expectation.expected_mode, "actual_mode": mode, "kind": kind},
                    )
                )
            if mode in expectation.forbidden_modes:
                findings.append(
                    self._finding(
                        index=index,
                        category="route_drift",
                        message="Turn entered a forbidden route mode for this scenario.",
                        evidence={"prompt": expectation.prompt, "forbidden_modes": list(expectation.forbidden_modes), "actual_mode": mode},
                    )
                )
            if expectation.expected_kind_prefix and not kind.startswith(expectation.expected_kind_prefix):
                findings.append(
                    self._finding(
                        index=index,
                        category="route_drift",
                        message="Turn selected a kind outside the expected surface family.",
                        evidence={"prompt": expectation.prompt, "expected_kind_prefix": expectation.expected_kind_prefix, "actual_kind": kind},
                    )
                )
            posture = response.get("assistant_quality_posture") if isinstance(response.get("assistant_quality_posture"), dict) else {}
            voice_profile = response.get("assistant_voice_profile") if isinstance(response.get("assistant_voice_profile"), dict) else {}
            style_mode = self._text(posture.get("style_mode") or voice_profile.get("style_mode"))
            if style_mode and style_mode != scenario.style_mode:
                style_mismatches += 1
                findings.append(
                    self._finding(
                        index=index,
                        category="tone_drift",
                        message="Turn did not preserve the scenario tone mode.",
                        evidence={"expected_style": scenario.style_mode, "actual_style": style_mode},
                    )
                )
            boundary_signals = self._string_list(posture.get("conversation_boundary_signals") or response.get("assistant_boundary_signals"))
            if expectation.required_boundary_signal and expectation.required_boundary_signal not in boundary_signals:
                findings.append(
                    self._finding(
                        index=index,
                        category="boundary_drift",
                        message="Turn missed a required assistant boundary signal.",
                        evidence={"required": expectation.required_boundary_signal, "signals": boundary_signals},
                    )
                )
            project_snapshot = response.get("project_context_snapshot") if isinstance(response.get("project_context_snapshot"), dict) else {}
            project_active = bool(project_snapshot.get("project_context_active"))
            if expectation.require_project_context is not None:
                if project_active == expectation.require_project_context:
                    project_passes += 1
                else:
                    findings.append(
                        self._finding(
                            index=index,
                            category="project_continuity_drift",
                            message="Project context activation did not match the scripted turn.",
                            evidence={"expected": expectation.require_project_context, "actual": project_active},
                        )
                    )
            work_thread = response.get("work_thread_continuity") if isinstance(response.get("work_thread_continuity"), dict) else {}
            if expectation.required_work_thread_intent:
                if bool(work_thread.get("active")) and self._text(work_thread.get("intent")) == expectation.required_work_thread_intent:
                    work_thread_passes += 1
                else:
                    findings.append(
                        self._finding(
                            index=index,
                            category="work_thread_drift",
                            message="Work-thread follow-up did not anchor to the expected intent.",
                            evidence={"expected_intent": expectation.required_work_thread_intent, "work_thread": dict(work_thread)},
                        )
                    )
            personal_count, recall_prompt = self._personal_memory_count(response)
            max_personal = expectation.max_personal_memory
            if max_personal is None and not expectation.allow_memory_recall:
                max_personal = 1
            if max_personal is not None and personal_count > max_personal and not recall_prompt:
                memory_violations += 1
                findings.append(
                    self._finding(
                        index=index,
                        category="memory_noise",
                        message="Turn over-injected personal memory for ordinary conversation.",
                        evidence={"personal_memory_count": personal_count, "max_personal_memory": max_personal},
                    )
                )
            beat = response.get("conversation_beat") if isinstance(response.get("conversation_beat"), dict) else {}
            if (
                mode == "conversation"
                and self._text(beat.get("response_repetition_risk")) == "high"
                and beat.get("follow_up_offer_allowed") is not False
            ):
                repetition_violations += 1
                findings.append(
                    self._finding(
                        index=index,
                        category="repetition_drift",
                        message="High repetition risk still allowed follow-up offers.",
                        evidence={"conversation_beat": dict(beat)},
                    )
                )
            visible_text = self._visible_text(response)
            scaffold_terms = [term for term in self.SCAFFOLD_TERMS if term in visible_text.lower()]
            if scaffold_terms:
                scaffold_violations += 1
                findings.append(
                    self._finding(
                        index=index,
                        category="scaffold_leakage",
                        message="Visible assistant text leaked internal scaffold wording.",
                        evidence={"terms": scaffold_terms},
                    )
                )

        incorrect = sum(1 for finding in findings if finding.severity == "incorrect")
        score = max(0.0, 1.0 - (incorrect * 0.08))
        judgment = "correct" if score >= 0.86 and incorrect == 0 else "weak" if score >= 0.58 else "incorrect"
        return ScriptedConversationEvaluation(
            scenario_name=scenario.name,
            evaluated_turns=evaluated_turns,
            expected_turns=expected_turns,
            score=score,
            judgment=judgment,
            findings=tuple(findings),
            metrics={
                "route_passes": route_passes,
                "style_mismatches": style_mismatches,
                "memory_violations": memory_violations,
                "repetition_violations": repetition_violations,
                "scaffold_violations": scaffold_violations,
                "project_passes": project_passes,
                "work_thread_passes": work_thread_passes,
            },
        )

    @staticmethod
    def _response(record: dict[str, Any]) -> dict[str, Any]:
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        return dict(response or record)

    @staticmethod
    def _text(value: object) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [" ".join(str(item or "").strip().lower().split()) for item in value if str(item or "").strip()]

    @staticmethod
    def _visible_text(response: dict[str, Any]) -> str:
        return str(response.get("reply") or response.get("user_facing_answer") or response.get("summary") or "").strip()

    @classmethod
    def _personal_memory_count(cls, response: dict[str, Any]) -> tuple[int, bool]:
        retrieval = response.get("memory_retrieval") if isinstance(response.get("memory_retrieval"), dict) else {}
        selected = retrieval.get("selected") if isinstance(retrieval.get("selected"), list) else []
        count = sum(
            1
            for item in selected
            if isinstance(item, dict)
            and (
                cls._text(item.get("source")) == "personal_memory"
                or cls._text(item.get("memory_kind")) in {"durable_user_memory", "profile", "preference"}
            )
        )
        return count, bool(retrieval.get("recall_prompt"))

    @staticmethod
    def _finding(
        *,
        index: int,
        category: str,
        message: str,
        evidence: dict[str, object],
    ) -> ScriptedConversationFinding:
        return ScriptedConversationFinding(
            turn_index=index + 1,
            category=category,
            severity="incorrect",
            message=message,
            evidence=evidence,
        )


def phase80_default_scenarios() -> tuple[ScriptedConversationScenario, ...]:
    """Built-in local scenarios for long-chat, work-thread, and memory-restraint QA."""

    human_chat_turns = (
        ScriptedTurnExpectation("hey buddy", expected_mode="conversation", forbidden_modes=("research",), max_personal_memory=0),
        ScriptedTurnExpectation("I'm doing pretty good, what about you?", expected_mode="conversation", required_boundary_signal="self_overview", forbidden_modes=("research",), max_personal_memory=0),
        ScriptedTurnExpectation("that makes sense", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("tell me about space", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("wait what do you mean", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("go back to what we were saying", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("yeah keep going", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what are you like?", expected_mode="conversation", required_boundary_signal="self_overview", forbidden_modes=("research",), max_personal_memory=0),
        ScriptedTurnExpectation("true", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("tell me about chemistry", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("hmm maybe", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what about biology?", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("that tracks", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("who are you again?", expected_mode="conversation", required_boundary_signal="self_overview", forbidden_modes=("research",), max_personal_memory=0),
        ScriptedTurnExpectation("good for now", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("actually one more thing", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("explain engineering", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("nice", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what else", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("thanks", expected_mode="conversation", max_personal_memory=0),
    )
    work_turns = (
        ScriptedTurnExpectation("create a release QA plan", expected_mode="planning", require_project_context=True),
        ScriptedTurnExpectation("what next", expected_mode="conversation", required_work_thread_intent="next_step", require_project_context=True),
        ScriptedTurnExpectation("summarize where we are", expected_mode="conversation", required_work_thread_intent="status_summary", require_project_context=True),
        ScriptedTurnExpectation("keep going", expected_mode="conversation", required_work_thread_intent="continue_work", require_project_context=True),
        ScriptedTurnExpectation("what did we decide", expected_mode="conversation", required_work_thread_intent="decision_recap", require_project_context=True),
        ScriptedTurnExpectation("tighten that", expected_mode="conversation", require_project_context=True),
        ScriptedTurnExpectation("what are we doing again", expected_mode="conversation", require_project_context=True),
        ScriptedTurnExpectation("go on", expected_mode="conversation", require_project_context=True),
        ScriptedTurnExpectation("tell me about climate", expected_mode="research", require_project_context=False),
        ScriptedTurnExpectation("back to the project", expected_mode="conversation", require_project_context=True),
        ScriptedTurnExpectation("what next", expected_mode="conversation", required_work_thread_intent="next_step", require_project_context=True),
        ScriptedTurnExpectation("continue", expected_mode="conversation", required_work_thread_intent="continue_work", require_project_context=True),
        ScriptedTurnExpectation("where are we", expected_mode="conversation", required_work_thread_intent="status_summary", require_project_context=True),
        ScriptedTurnExpectation("what's decided", expected_mode="conversation", required_work_thread_intent="decision_recap", require_project_context=True),
        ScriptedTurnExpectation("keep working", expected_mode="conversation", required_work_thread_intent="continue_work", require_project_context=True),
        ScriptedTurnExpectation("status", expected_mode="conversation", required_work_thread_intent="status_summary", require_project_context=True),
        ScriptedTurnExpectation("next steps", expected_mode="conversation", required_work_thread_intent="next_step", require_project_context=True),
        ScriptedTurnExpectation("thanks", expected_mode="conversation", require_project_context=False),
        ScriptedTurnExpectation("what should we do next", expected_mode="conversation", required_work_thread_intent="next_step", require_project_context=True),
        ScriptedTurnExpectation("recap", expected_mode="conversation", required_work_thread_intent="status_summary", require_project_context=True),
    )
    memory_turns = (
        ScriptedTurnExpectation("remember this about me: keep answers brief", expected_mode="conversation", allow_memory_recall=True),
        ScriptedTurnExpectation("hey", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("tell me about physics", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("that's interesting", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what do you remember about my preferences", expected_mode="research", allow_memory_recall=True),
        ScriptedTurnExpectation("cool", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("explain math", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("can you keep it brief for me", expected_mode="conversation", max_personal_memory=1),
        ScriptedTurnExpectation("go on", expected_mode="conversation", max_personal_memory=1),
        ScriptedTurnExpectation("tell me about history", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("what else", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what do you remember about me", expected_mode="research", allow_memory_recall=True),
        ScriptedTurnExpectation("thanks", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("how does chemistry work", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("true", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what are you like", expected_mode="conversation", required_boundary_signal="self_overview", max_personal_memory=0),
        ScriptedTurnExpectation("nice keep going", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("what is cybersecurity", expected_mode="research", max_personal_memory=0),
        ScriptedTurnExpectation("that makes sense", expected_mode="conversation", max_personal_memory=0),
        ScriptedTurnExpectation("good for now", expected_mode="conversation", max_personal_memory=0),
    )
    return (
        ScriptedConversationScenario(
            name="phase80_human_chat",
            description="Ordinary 20-turn chat should stay natural, route knowledge prompts cleanly, and avoid memory noise.",
            style_mode="collab",
            turns=human_chat_turns,
        ),
        ScriptedConversationScenario(
            name="phase80_work_thread",
            description="Current-work follow-ups should stay anchored to active thread and project state.",
            style_mode="default",
            turns=work_turns,
        ),
        ScriptedConversationScenario(
            name="phase80_memory_restraint",
            description="Long-term familiarity should appear only on explicit recall or clearly relevant personal turns.",
            style_mode="direct",
            turns=memory_turns,
        ),
    )
