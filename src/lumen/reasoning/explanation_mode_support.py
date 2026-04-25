from __future__ import annotations

from dataclasses import dataclass
import re

from lumen.knowledge.knowledge_service import KnowledgeService
from lumen.knowledge.models import KnowledgeLookupResult
from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse, PlanningResponse, ResearchResponse
from lumen.reasoning.response_variation import ResponseVariationLayer
from lumen.routing.anchor_registry import detect_explanation_mode


@dataclass(slots=True)
class ExplanationModeMatch:
    mode: str
    normalized_prompt: str


class ExplanationModeSupport:
    """Shared explanation-mode detector and follow-up surface transformer."""

    _prompt_nlu = PromptNLU()

    @classmethod
    def detect_mode(cls, prompt: str) -> ExplanationModeMatch | None:
        understanding = cls._prompt_nlu.analyze(prompt)
        normalized = understanding.surface_views.lookup_ready_text
        if not normalized:
            return None
        mode = detect_explanation_mode(normalized)
        if mode is None:
            if understanding.structure.predicate == "break down" and "simply" in understanding.structure.modifiers:
                mode = "break_down"
            elif understanding.structure.predicate == "break down":
                mode = "break_down"
        if mode is None:
            return None
        return ExplanationModeMatch(mode=mode, normalized_prompt=normalized)

    @classmethod
    def build_follow_up_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
        active_thread: dict[str, object] | None,
        knowledge_service: KnowledgeService | None,
    ) -> dict[str, object] | None:
        match = cls.detect_mode(prompt)
        if match is None or not recent_interactions:
            return None
        latest = recent_interactions[0]
        response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        style = InteractionStylePolicy.interaction_style(interaction_profile)
        target = cls._select_follow_up_target(
            latest=latest,
            response=response,
            active_thread=active_thread,
        )

        if target == "math":
            return cls._build_math_follow_up(
                mode=match.mode,
                style=style,
                latest=latest,
                response=response,
                active_thread=active_thread,
            )
        if target == "research":
            return cls._build_research_follow_up(
                mode=match.mode,
                style=style,
                latest=latest,
                response=response,
                knowledge_service=knowledge_service,
                active_thread=active_thread,
            )
        if target == "planning":
            return cls._build_planning_follow_up(
                mode=match.mode,
                style=style,
                latest=latest,
                response=response,
                active_thread=active_thread,
            )
        return None

    @classmethod
    def _select_follow_up_target(
        cls,
        *,
        latest: dict[str, object],
        response: dict[str, object],
        active_thread: dict[str, object] | None,
    ) -> str | None:
        latest_mode = str(response.get("mode") or latest.get("mode") or "").strip()
        latest_kind = str(response.get("kind") or latest.get("kind") or "").strip()
        surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        lane = str(surface.get("lane") or "").strip()
        topic = str(surface.get("topic") or "").strip()
        if topic and lane != "math":
            return "research"
        if latest_mode == "research":
            return "research"
        if latest_mode == "planning":
            return "planning"
        if lane == "math":
            return "math"
        tool_execution = response.get("tool_execution") if isinstance(response.get("tool_execution"), dict) else {}
        if str(tool_execution.get("tool_id") or "").strip() == "math":
            return "math"
        if latest_mode == "conversation" and latest_kind.startswith("conversation.quick_math"):
            return "math"
        active_mode = str((active_thread or {}).get("mode") or "").strip()
        if active_mode == "research":
            return "research"
        if active_mode == "planning":
            return "planning"
        active_tool_context = active_thread.get("tool_context") if isinstance(active_thread, dict) else {}
        if str((active_tool_context or {}).get("tool_id") or "").strip() == "math":
            return "math"
        return None

    @classmethod
    def _build_research_follow_up(
        cls,
        *,
        mode: str,
        style: str,
        latest: dict[str, object],
        response: dict[str, object],
        knowledge_service: KnowledgeService | None,
        active_thread: dict[str, object] | None,
    ) -> dict[str, object]:
        surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        topic = (
            str(surface.get("topic") or "").strip()
            or str(response.get("resolved_prompt") or latest.get("resolved_prompt") or latest.get("prompt") or "").strip()
            or str((active_thread or {}).get("normalized_topic") or "").strip()
            or "this"
        )
        base_text = cls._latest_answer_text(latest=latest, response=response)
        lookup = knowledge_service.lookup(topic) if knowledge_service is not None and topic else None
        transformed = cls._transform_research_text(
            mode=mode,
            style=style,
            topic=topic,
            base_text=base_text,
            lookup=lookup,
            response=response,
        )
        payload = ResearchResponse(
            mode="research",
            kind="research.summary",
            summary=transformed,
            findings=[],
        ).to_dict()
        payload["reply"] = transformed
        payload["user_facing_answer"] = transformed
        payload["explanation_mode"] = mode
        payload["domain_surface"] = {
            "lane": str(surface.get("lane") or "knowledge").strip() or "knowledge",
            "topic": topic,
            "entry_id": surface.get("entry_id"),
        }
        return payload

    @classmethod
    def _build_planning_follow_up(
        cls,
        *,
        mode: str,
        style: str,
        latest: dict[str, object],
        response: dict[str, object],
        active_thread: dict[str, object] | None,
    ) -> dict[str, object]:
        topic = (
            str((active_thread or {}).get("normalized_topic") or "").strip()
            or str(latest.get("prompt") or "").strip()
            or "the current plan"
        )
        summary = cls._latest_answer_text(latest=latest, response=response)
        steps = [str(item).strip() for item in response.get("steps") or [] if str(item).strip()]
        transformed = cls._transform_planning_text(
            mode=mode,
            style=style,
            topic=topic,
            summary=summary,
            steps=steps,
        )
        payload = PlanningResponse(
            mode="planning",
            kind=str(response.get("kind") or latest.get("kind") or "planning.architecture"),
            summary=transformed,
            steps=[],
        ).to_dict()
        payload["reply"] = transformed
        payload["user_facing_answer"] = transformed
        payload["explanation_mode"] = mode
        return payload

    @classmethod
    def _build_math_follow_up(
        cls,
        *,
        mode: str,
        style: str,
        latest: dict[str, object],
        response: dict[str, object],
        active_thread: dict[str, object] | None,
    ) -> dict[str, object]:
        surface = response.get("domain_surface") if isinstance(response.get("domain_surface"), dict) else {}
        expression = str(surface.get("expression") or "").strip()
        answer = str(surface.get("answer") or "").strip()
        equation, variable = cls._tool_equation_context(response=response, active_thread=active_thread)
        if not answer:
            answer = cls._extract_solution_value(
                text=cls._latest_answer_text(latest=latest, response=response),
                variable=variable or "x",
            )
        transformed = cls._transform_math_text(
            mode=mode,
            style=style,
            expression=expression,
            answer=answer,
            equation=equation,
            variable=variable,
        )
        payload = ConversationResponse(
            mode="conversation",
            kind="conversation.quick_math_follow_up",
            summary=transformed,
            reply=transformed,
        ).to_dict()
        payload["user_facing_answer"] = transformed
        payload["explanation_mode"] = mode
        payload["domain_surface"] = {
            "lane": "math",
            "expression": expression or equation or None,
            "answer": answer or None,
            "equation": equation or None,
            "variable": variable or None,
        }
        return payload

    @staticmethod
    def _latest_answer_text(*, latest: dict[str, object], response: dict[str, object]) -> str:
        return str(
            response.get("user_facing_answer")
            or response.get("reply")
            or response.get("summary")
            or latest.get("summary")
            or ""
        ).strip()

    @staticmethod
    def _tool_equation_context(
        *,
        response: dict[str, object],
        active_thread: dict[str, object] | None,
    ) -> tuple[str, str]:
        tool_execution = response.get("tool_execution") if isinstance(response.get("tool_execution"), dict) else {}
        params = tool_execution.get("params") if isinstance(tool_execution.get("params"), dict) else {}
        equation = str(params.get("equation") or "").strip()
        variable = str(params.get("variable") or "").strip()
        if equation and variable:
            return equation, variable
        tool_context = (active_thread or {}).get("tool_context") if isinstance(active_thread, dict) else {}
        thread_params = tool_context.get("params") if isinstance(tool_context, dict) and isinstance(tool_context.get("params"), dict) else {}
        return (
            str(thread_params.get("equation") or "").strip(),
            str(thread_params.get("variable") or "").strip(),
        )

    @staticmethod
    def _extract_solution_value(*, text: str, variable: str) -> str:
        match = re.search(rf"{re.escape(variable)}\s*=\s*([^,\n.;]+)", str(text or ""))
        if match is None:
            return ""
        return str(match.group(1) or "").strip()

    @classmethod
    def _transform_research_text(
        cls,
        *,
        mode: str,
        style: str,
        topic: str,
        base_text: str,
        lookup: KnowledgeLookupResult | None,
        response: dict[str, object],
    ) -> str:
        entry = lookup.primary if lookup is not None else None
        key_points = list(getattr(entry, "key_points", []) or [])
        related = list(getattr(entry, "related_topics", []) or [])
        summary_short = str(getattr(entry, "summary_short", "") or "").strip()
        summary_medium = str(getattr(entry, "summary_medium", "") or "").strip()
        summary_deep = str(getattr(entry, "summary_deep", "") or "").strip()
        findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]

        if mode == "deeper":
            parts = [summary_deep or summary_medium or base_text]
            if key_points:
                parts.append(key_points[min(1, len(key_points) - 1)])
            if related:
                parts.append(f"It also connects to {related[0]}.")
            if parts:
                parts.insert(0, f"Deeper view of {topic}:")
            return cls._style_wrap(
                style=style,
                topic=topic,
                mode=mode,
                content=" ".join(part.strip() for part in parts if part and part.strip()),
            )

        if mode == "break_down":
            simple = summary_short or summary_medium or base_text
            chunks = [f"Plain-English version: {simple.rstrip('.')}."] if simple else []
            if key_points:
                chunks.append(f"The main thing to hold onto is {key_points[0].rstrip('.')}.")
            elif findings:
                chunks.append(f"The useful takeaway is {findings[0].rstrip('.')}.")
            if related:
                chunks.append(f"You can think of it as part of the bigger picture around {related[0]}.")
            return cls._style_wrap(
                style=style,
                topic=topic,
                mode=mode,
                content=" ".join(chunks),
            )

        steps = []
        first = summary_short or summary_medium or base_text or topic
        steps.append(f"1. Start with the core idea: {first.rstrip('.')}.")
        if key_points:
            steps.append(f"2. Next piece: {key_points[0].rstrip('.')}.")
        elif findings:
            steps.append(f"2. Next piece: {findings[0].rstrip('.')}.")
        if related:
            steps.append(f"3. Then connect it to {related[0]}.")
        else:
            steps.append(f"3. That is the part that matters most for understanding {topic}.")
        return cls._style_wrap(
            style=style,
            topic=topic,
            mode=mode,
            content="\n".join(steps),
        )

    @classmethod
    def _transform_planning_text(
        cls,
        *,
        mode: str,
        style: str,
        topic: str,
        summary: str,
        steps: list[str],
    ) -> str:
        if mode == "deeper":
            extra = steps[0] if steps else "The main structure is still the useful thing to look at."
            content = f"Deeper pass: {summary.rstrip('.')} {extra.rstrip('.')}."
        elif mode == "break_down":
            first_step = steps[0] if steps else "We are trying to make the plan easier to follow."
            base = summary.rstrip(".")
            if summary and ("first pass" in summary.lower() or "workable answer" in summary.lower()):
                base = "The simple version is to keep the design focused on the core constraints first"
            content = (
                f"Simple version: {base}. "
                f"From there, the next clear move is {first_step.rstrip('.')}."
            )
        else:
            lines = [f"1. The plan is about {topic}.", f"2. Right now the core move is {summary.rstrip('.')}."] 
            if steps:
                lines.append(f"3. Then do {steps[0].rstrip('.')}.")
            content = "\n".join(lines)
        return cls._style_wrap(style=style, topic=topic, mode=mode, content=content)

    @classmethod
    def _transform_math_text(
        cls,
        *,
        mode: str,
        style: str,
        expression: str,
        answer: str,
        equation: str,
        variable: str,
    ) -> str:
        if equation and variable and answer:
            linear = cls._parse_linear_equation(equation=equation, variable=variable)
            if linear is not None:
                coefficient, offset, right_side = linear
                remainder = right_side - offset
                if mode == "deeper":
                    content = (
                        f"Deeper view: start from {equation}. To isolate {variable}, undo the + {offset} first, "
                        f"so you get {coefficient}{variable} = {remainder}. "
                        f"Then divide both sides by {coefficient}, which gives {variable} = {answer}."
                    )
                elif mode == "break_down":
                    content = (
                        f"Simple version: think of {equation} like a balance. First remove the extra {offset} from both sides, "
                        f"so {coefficient}{variable} = {remainder}. Then split both sides by {coefficient}, "
                        f"and that leaves {variable} = {answer}."
                    )
                else:
                    content = "\n".join(
                        [
                            f"1. Start with {equation}.",
                            f"2. Subtract {offset} from both sides to get {coefficient}{variable} = {remainder}.",
                            f"3. Divide both sides by {coefficient}.",
                            f"4. So {variable} = {answer}.",
                        ]
                    )
                return cls._style_wrap(style=style, topic=equation, mode=mode, content=content)
            generic_lines = [
                f"1. Start with {equation}.",
                f"2. The goal is to isolate {variable}.",
                f"3. The solved value is {variable} = {answer}.",
            ]
            content = (
                f"The key move is to isolate {variable} in {equation}, which gives {variable} = {answer}."
                if mode != "step_by_step"
                else "\n".join(generic_lines)
            )
            return cls._style_wrap(style=style, topic=equation, mode=mode, content=content)

        target = expression or equation
        if mode == "deeper":
            content = f"Treat {target} as straightforward arithmetic. When you work through it carefully, you get {answer}."
        elif mode == "break_down":
            content = f"The simple version is just: work through {target}, and it comes out to {answer}."
        else:
            content = "\n".join(
                [
                    f"1. Start with {target}.",
                    "2. Evaluate the arithmetic directly.",
                    f"3. The result is {answer}.",
                ]
            )
        return cls._style_wrap(style=style, topic=target or "that math", mode=mode, content=content)

    @staticmethod
    def _parse_linear_equation(*, equation: str, variable: str) -> tuple[int, int, int] | None:
        normalized = str(equation or "").replace(" ", "")
        pattern = re.compile(
            rf"^([+-]?\d*){re.escape(variable)}([+-]\d+)?=([+-]?\d+)$",
            flags=re.IGNORECASE,
        )
        match = pattern.match(normalized)
        if match is None:
            return None
        coefficient_text = str(match.group(1) or "").strip()
        offset_text = str(match.group(2) or "0").strip()
        right_text = str(match.group(3) or "").strip()
        if coefficient_text in {"", "+"}:
            coefficient = 1
        elif coefficient_text == "-":
            coefficient = -1
        else:
            coefficient = int(coefficient_text)
        return coefficient, int(offset_text), int(right_text)

    @staticmethod
    def _style_wrap(*, style: str, topic: str, mode: str, content: str) -> str:
        normalized = InteractionStylePolicy.normalize_style(style)
        body = str(content or "").strip()
        if normalized == "direct":
            if mode == "deeper":
                return f"Deeper version: {body}"
            if mode == "break_down":
                return f"Simple version: {body}"
            return body
        if normalized == "collab":
            if mode == "deeper":
                return f"Let's go one layer deeper on {topic}. {body}"
            if mode == "break_down":
                return f"Here is the simpler version of {topic}. {body}"
            return f"Let's walk it through. {body}"
        if mode == "deeper":
            return f"Here is a deeper pass on {topic}. {body}"
        if mode == "break_down":
            return f"Here is the plain-English version of {topic}. {body}"
        return f"Here it is step by step. {body}"
