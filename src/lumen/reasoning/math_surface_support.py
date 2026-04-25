from __future__ import annotations

import ast
import re

from lumen.nlu.prompt_nlu import PromptNLU
from lumen.reasoning.interaction_style_policy import InteractionStylePolicy
from lumen.reasoning.response_models import ConversationResponse
from lumen.reasoning.response_variation import ResponseVariationLayer


class MathSurfaceSupport:
    """Owns bounded quick-math surfaces and lightweight follow-ups."""

    _prompt_nlu = PromptNLU()

    FOLLOW_UP_PROMPTS = {
        "why",
        "how did you get that",
        "howd you get that",
        "show me quickly",
        "show me",
        "what do you mean",
        "how so",
    }
    CARRYOVER_PREFIXES = (
        "what about ",
        "and ",
    )

    PREFIXES = (
        "what is ",
        "what's ",
        "whats ",
        "wat is ",
        "wats ",
        "calculate ",
        "calc ",
        "compute ",
        "can you do ",
        "could you do ",
        "work out ",
        "help me with ",
        "solve ",
    )

    SUFFIXES = (
        " is what",
        " equals what",
        " is how much",
        " equals",
    )

    OPERATOR_WORDS = (
        (" divided by ", "/"),
        (" times ", "*"),
        (" multiplied by ", "*"),
        (" plus ", "+"),
        (" minus ", "-"),
    )

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        follow_up = cls.build_follow_up_response(
            prompt=prompt,
            interaction_profile=interaction_profile,
            recent_interactions=recent_interactions,
        )
        if follow_up is not None:
            return follow_up

        expression = cls.extract_expression(prompt)
        if expression is None:
            return None
        value = cls.evaluate(expression)
        if value is None:
            return None

        style = InteractionStylePolicy.interaction_style(interaction_profile)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        answer = str(value)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)

        if style == "direct":
            pool = (
                answer,
                f"{answer}.",
                f"Answer: {answer}.",
                f"Result: {answer}.",
                f"It is {answer}.",
                f"That is {answer}.",
                f"{expression} = {answer}.",
                f"{answer} exactly.",
            )
        elif style == "collab":
            pool = (
                f"It's {answer}.",
                f"That comes out to {answer}.",
                f"{expression} works out to {answer}.",
                f"You get {answer}.",
                f"That gives you {answer}.",
                f"It lands at {answer}.",
                f"The result there is {answer}.",
                f"That one comes out to {answer}.",
            )
        else:
            pool = (
                f"It's {answer}.",
                f"That works out to {answer}.",
                f"The result is {answer}.",
                f"{expression} comes out to {answer}.",
                f"That evaluates to {answer}.",
                f"You get {answer}.",
                f"The answer there is {answer}.",
                f"That gives {answer}.",
            )

        reply = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[expression, style, answer, "quick_math"],
            recent_texts=recent_texts,
        )
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.quick_math",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["domain_surface"] = {
            "lane": "math",
            "expression": expression,
            "answer": answer,
        }
        return response

    @classmethod
    def build_follow_up_response(
        cls,
        *,
        prompt: str,
        interaction_profile,
        recent_interactions: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if not recent_interactions:
            return None
        normalized = cls._lookup_ready_text(prompt)
        latest = recent_interactions[0]
        latest_response = latest.get("response") if isinstance(latest.get("response"), dict) else {}
        latest_kind = str(latest_response.get("kind") or latest.get("kind") or "").strip()
        tool_execution = (
            latest_response.get("tool_execution")
            if isinstance(latest_response.get("tool_execution"), dict)
            else {}
        )
        live_math_solve = (
            str(latest_response.get("mode") or latest.get("mode") or "").strip() == "tool"
            and str(tool_execution.get("tool_id") or "").strip() == "math"
            and str(tool_execution.get("capability") or "").strip() == "solve_equation"
        )
        if (
            latest_kind not in {"conversation.quick_math", "conversation.quick_math_follow_up"}
            and not live_math_solve
        ):
            return None

        surface = latest_response.get("domain_surface") if isinstance(latest_response.get("domain_surface"), dict) else {}
        expression = str(surface.get("expression") or "").strip()
        answer = str(surface.get("answer") or "").strip()
        equation = str(surface.get("equation") or "").strip()
        variable = str(surface.get("variable") or "").strip() or "x"
        if live_math_solve:
            params = tool_execution.get("params") if isinstance(tool_execution.get("params"), dict) else {}
            equation = equation or str(params.get("equation") or "").strip()
            variable = str(params.get("variable") or variable).strip() or "x"
            if not answer:
                answer = cls._extract_solution_value(
                    text=str(
                        latest_response.get("user_facing_answer")
                        or latest_response.get("reply")
                        or latest_response.get("summary")
                        or ""
                    ),
                    variable=variable,
                )
        if not expression or not answer:
            latest_summary = str(latest_response.get("summary") or latest.get("summary") or "").strip()
            if not latest_summary:
                if not (live_math_solve and equation and answer):
                    return None
            elif not answer:
                answer = latest_summary.rstrip(".")

        carryover_expression = cls.extract_follow_up_expression(normalized)
        if carryover_expression is not None:
            value = cls.evaluate(carryover_expression)
            if value is None:
                return None
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return cls._build_result_response(
                expression=carryover_expression,
                answer=str(value),
                style=InteractionStylePolicy.interaction_style(interaction_profile),
                recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
                seed_tag="quick_math_carryover",
            )

        if normalized not in cls.FOLLOW_UP_PROMPTS:
            return None

        style = InteractionStylePolicy.interaction_style(interaction_profile)
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        if live_math_solve and equation and answer:
            lines = cls._equation_explanation_lines(
                equation=equation,
                variable=variable,
                answer=answer,
            )
            reply = cls._math_explanation_surface(
                style=style,
                equation=equation,
                answer=answer,
                lines=lines,
            )
            response = ConversationResponse(
                mode="conversation",
                kind="conversation.quick_math_follow_up",
                summary=reply,
                reply=reply,
            ).to_dict()
            response["domain_surface"] = {
                "lane": "math",
                "expression": equation,
                "answer": answer,
                "equation": equation,
                "variable": variable,
            }
            return response
        if style == "direct":
            pool = (
                f"Just evaluate {expression}. That gives {answer}.",
                f"Compute {expression} directly and you get {answer}.",
                f"It reduces straight to {answer}.",
                f"That arithmetic comes out to {answer}.",
            )
        elif style == "collab":
            pool = (
                f"Just evaluate {expression} step by step and it lands at {answer}.",
                f"Nothing tricky there. You work through {expression} and it comes out to {answer}.",
                f"It is a straight arithmetic pass: {expression} gives {answer}.",
                f"You can read it directly as {expression}, which works out to {answer}.",
            )
        else:
            pool = (
                f"Just evaluate {expression} directly and it comes out to {answer}.",
                f"It is a short arithmetic step: {expression} gives {answer}.",
                f"There is not much hidden there. {expression} works out to {answer}.",
                f"It reduces directly to {answer} once you compute {expression}.",
            )
        reply = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[normalized, expression, answer, style, "quick_math_follow_up"],
            recent_texts=recent_texts,
        )
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.quick_math_follow_up",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["domain_surface"] = {
            "lane": "math",
            "expression": expression,
            "answer": answer,
        }
        return response

    @staticmethod
    def _extract_solution_value(*, text: str, variable: str) -> str:
        match = re.search(rf"{re.escape(variable)}\s*=\s*([^,\n.;]+)", str(text or ""))
        if match is None:
            return ""
        return str(match.group(1) or "").strip()

    @classmethod
    def _equation_explanation_lines(
        cls,
        *,
        equation: str,
        variable: str,
        answer: str,
    ) -> list[str]:
        parsed = cls._parse_linear_equation(equation=equation, variable=variable)
        if parsed is None:
            return [
                f"Start with {equation}.",
                f"The goal is to isolate {variable}.",
                f"That leaves {variable} = {answer}.",
            ]
        coefficient, offset, right_side = parsed
        reduced_left = (
            f"{coefficient}{variable}"
            if coefficient not in {1, -1}
            else (variable if coefficient == 1 else f"-{variable}")
        )
        lines = [f"Start with {equation}."]
        if offset > 0:
            lines.append(
                f"Subtract {offset} from both sides to get {reduced_left} = {right_side - offset}."
            )
        elif offset < 0:
            lines.append(
                f"Add {abs(offset)} to both sides to get {reduced_left} = {right_side - offset}."
            )
        if coefficient not in {1, 0}:
            lines.append(f"Divide both sides by {coefficient} to get {variable} = {answer}.")
        elif coefficient == -1:
            lines.append(f"Divide both sides by -1 to get {variable} = {answer}.")
        elif len(lines) == 1:
            lines.append(f"That already leaves {variable} = {answer}.")
        return lines

    @classmethod
    def _math_explanation_surface(
        cls,
        *,
        style: str,
        equation: str,
        answer: str,
        lines: list[str],
    ) -> str:
        if style == "direct":
            return "\n".join(
                [f"{equation} solves to {answer}.", *lines[1:]] if lines else [f"{equation} solves to {answer}."]
            )
        opener = (
            f"Here is the short version for {equation}:"
            if style == "collab"
            else f"Here is how it works for {equation}:"
        )
        return "\n".join([opener, *lines])

    @staticmethod
    def _parse_linear_equation(*, equation: str, variable: str) -> tuple[int, int, int] | None:
        pattern = re.compile(rf"^([+-]?\d*){re.escape(variable)}([+-]\d+)?=([+-]?\d+)$", flags=re.IGNORECASE)
        normalized = str(equation or "").replace(" ", "")
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

    @classmethod
    def _build_result_response(
        cls,
        *,
        expression: str,
        answer: str,
        style: str,
        recent_texts: list[str],
        seed_tag: str,
    ) -> dict[str, object]:
        if style == "direct":
            pool = (
                answer,
                f"{answer}.",
                f"{expression} = {answer}.",
                f"Result: {answer}.",
            )
        elif style == "collab":
            pool = (
                f"It's {answer}.",
                f"{expression} comes out to {answer}.",
                f"That one lands at {answer}.",
                f"You get {answer}.",
            )
        else:
            pool = (
                f"It's {answer}.",
                f"That works out to {answer}.",
                f"{expression} comes out to {answer}.",
                f"The result is {answer}.",
            )
        reply = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[expression, style, answer, seed_tag],
            recent_texts=recent_texts,
        )
        response = ConversationResponse(
            mode="conversation",
            kind="conversation.quick_math",
            summary=reply,
            reply=reply,
        ).to_dict()
        response["domain_surface"] = {
            "lane": "math",
            "expression": expression,
            "answer": answer,
        }
        return response

    @classmethod
    def extract_follow_up_expression(cls, normalized_prompt: str) -> str | None:
        normalized = str(normalized_prompt or "").strip().lower()
        if not normalized:
            return None
        candidate = normalized
        for prefix in cls.CARRYOVER_PREFIXES:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
        if candidate.startswith("what about "):
            candidate = candidate[len("what about ") :].strip()
        if candidate.endswith("?"):
            candidate = candidate.rstrip("?").strip()
        return cls.extract_expression(candidate)

    @classmethod
    def extract_expression(cls, prompt: str) -> str | None:
        normalized = " ".join(str(prompt or "").strip().lower().split())
        if not normalized:
            return None
        for prefix in cls.PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break
        normalized = normalized.rstrip("?.!")
        for suffix in cls.SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break
        normalized = normalized.replace("×", "*")
        for source, target in cls.OPERATOR_WORDS:
            normalized = normalized.replace(source, f" {target} ")
        normalized = re.sub(r"(?<=\d)\s*[x]\s*(?=\d)", "*", normalized)
        normalized = " ".join(normalized.split())
        if not normalized:
            return None
        if len(normalized) > 48:
            return None
        if not any(op in normalized for op in "+-*/"):
            return None
        allowed = set("0123456789+-*/(). ")
        if any(ch not in allowed for ch in normalized):
            return None
        if "**" in normalized or "//" in normalized:
            return None
        if any(ch.isalpha() for ch in normalized):
            return None
        return normalized

    @staticmethod
    def evaluate(expression: str) -> int | float | None:
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = _eval(node.operand)
                return operand if isinstance(node.op, ast.UAdd) else -operand
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if right == 0:
                    raise ZeroDivisionError
                return left / right
            raise ValueError("unsupported")

        try:
            value = _eval(parsed)
        except (ValueError, ZeroDivisionError):
            return None
        return value if isinstance(value, (int, float)) else None

    @classmethod
    def _lookup_ready_text(cls, prompt: str) -> str:
        return cls._prompt_nlu.analyze(prompt).surface_views.lookup_ready_text
