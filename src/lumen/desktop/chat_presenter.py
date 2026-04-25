from __future__ import annotations

from lumen.desktop.chat_experience_support import (
    build_capability_transparency_line,
    build_tool_transparency_line,
    maybe_attach_momentum_prompt,
)


class ChatPresenter:
    """Formats assistant payloads into a simple desktop chat transcript."""

    _SCAFFOLD_PHRASES: tuple[str, ...] = (
        "here's a useful first pass",
        "here is a useful first pass",
        "here's a first pass",
        "here is a first pass",
        "here's a workable answer",
        "here is a workable answer",
        "here's the clearest current answer",
        "here is the clearest current answer",
        "here's a grounded answer using the best current assumptions",
        "here is a grounded answer using the best current assumptions",
        "here's a grounded answer, with the assumptions kept visible",
        "here is a grounded answer, with the assumptions kept visible",
        "here's a grounded answer",
        "here is a grounded answer",
        "here's the clearest answer",
        "here is the clearest answer",
        "here's the grounded explanation",
        "here is the grounded explanation",
        "here's the best first read",
        "here is the best first read",
        "best first read",
        "hold it provisionally",
        "provisionally",
        "best next check",
        "early conclusion",
        "route validation",
        "route-validation",
        "strongest local context",
        "here's a useful first plan, with the assumptions kept visible",
        "here is a useful first plan, with the assumptions kept visible",
        "best current assumptions",
        "next move",
        "next check",
        "tighten first",
        "local anchor",
        "avoid broad extrapolation",
    )
    _SCAFFOLD_LINE_PREFIXES: tuple[str, ...] = (
        "next:",
        "action:",
        "validation plan:",
        "support status:",
        "anchor evidence:",
        "route caution:",
    )

    @staticmethod
    def _render_constrained_research(response: dict[str, object], rendered: str) -> str:
        constraint = response.get("response_constraint")
        if not isinstance(constraint, dict):
            return rendered
        if str(constraint.get("level") or "").strip() != "high_level_only":
            return rendered
        note = "Keeping this high-level for safety."
        normalized_rendered = rendered.lower()
        if note.lower() in normalized_rendered:
            return rendered
        if rendered:
            return f"{rendered}\n\n{note}"
        return note

    @staticmethod
    def render(
        response: dict[str, object],
        *,
        decorate: bool = False,
        style: str = "default",
        recent_assistant_texts: list[str] | None = None,
    ) -> str:
        mode = str(response.get("mode") or "").strip()
        if response.get("user_facing_answer"):
            rendered = str(response.get("user_facing_answer") or "").strip()
        elif mode == "conversation":
            rendered = str(response.get("reply") or response.get("summary") or "").strip()
        elif mode == "clarification":
            question = str(response.get("clarification_question") or response.get("summary") or "").strip()
            options = [str(option).strip() for option in response.get("options") or [] if str(option).strip()]
            if options:
                rendered = f"{question}\n\nOptions: " + " | ".join(options)
            else:
                rendered = question
        elif mode == "safety":
            lines = [str(response.get("boundary_explanation") or response.get("summary") or "").strip()]
            redirects = [str(item).strip() for item in response.get("safe_redirects") or [] if str(item).strip()]
            if redirects:
                lines.append("Safe directions:")
                lines.extend(f"- {item}" for item in redirects)
            rendered = "\n".join(line for line in lines if line)
        elif mode == "planning":
            rendered = str(response.get("reply") or response.get("summary") or "").strip()
        elif mode == "research":
            rendered = str(response.get("reply") or response.get("summary") or "").strip()
            rendered = ChatPresenter._render_constrained_research(response, rendered)
        elif mode == "tool":
            rendered = str(
                response.get("reply")
                or response.get("summary")
                or "Tool run completed."
            ).strip()
        else:
            rendered = str(response.get("summary") or response).strip()

        user_prompt = ChatPresenter._user_prompt(response)
        rendered = ChatPresenter.normalize_for_user(
            rendered,
            response=response,
            user_prompt=user_prompt,
        )
        if mode == "research":
            rendered = ChatPresenter._render_constrained_research(response, rendered)

        if not decorate or not rendered:
            return rendered

        lines: list[str] = []
        transparency = build_tool_transparency_line(response)
        if transparency:
            lines.append(transparency)
            lines.append("")
        capability_line = build_capability_transparency_line(response)
        if capability_line:
            lines.append(capability_line)
            lines.append("")
        lines.append(rendered)
        momentum = maybe_attach_momentum_prompt(
            response,
            style=style,
            recent_assistant_texts=recent_assistant_texts,
        )
        if momentum:
            lines.extend(["", momentum])
        return "\n".join(lines).strip()

    @staticmethod
    def _render_structured(
        *,
        summary: str,
        items: list[object],
        closeout: str,
        closeout_label: str,
    ) -> str:
        lines: list[str] = []
        if summary:
            lines.append(summary)
        normalized_items = [str(item).strip() for item in items if str(item).strip()]
        if normalized_items:
            if lines:
                lines.append("")
            lines.extend(f"- {item}" for item in normalized_items)
        if closeout:
            if lines:
                lines.append("")
            lines.append(f"{closeout_label}: {closeout}")
        return "\n".join(lines).strip()

    @staticmethod
    def build_status(response: dict[str, object]) -> str:
        mode = str(response.get("mode") or "").strip() or "unknown"
        provider = response.get("provider_inference") or {}
        if isinstance(provider, dict) and provider.get("provider_id"):
            provider_id = str(provider.get("provider_id") or "").strip()
            model = str(provider.get("model") or "").strip()
            if model:
                return f"{mode} via {provider_id}:{model}"
            return f"{mode} via {provider_id}"
        return mode

    @staticmethod
    def is_internal_scaffold(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        if any(phrase in normalized for phrase in ChatPresenter._SCAFFOLD_PHRASES):
            return True
        lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith("- ")]
        if len(bullet_lines) >= 2:
            return True
        return any(
            line.startswith(prefix)
            for line in lines
            for prefix in ChatPresenter._SCAFFOLD_LINE_PREFIXES
        )

    @staticmethod
    def normalize_for_user(
        text: str,
        *,
        response: dict[str, object],
        user_prompt: str | None = None,
    ) -> str:
        rendered = str(text or "").strip()
        if not rendered:
            return rendered
        if not ChatPresenter.is_internal_scaffold(rendered):
            return rendered

        keep_structure = ChatPresenter._prompt_allows_structure(user_prompt)
        for key in ("user_facing_answer", "reply", "explanatory_body"):
            candidate = str(response.get(key) or "").strip()
            if candidate and not ChatPresenter.is_internal_scaffold(candidate):
                return ChatPresenter._clean_user_text(candidate, keep_structure=keep_structure)

        findings = [str(item).strip() for item in response.get("findings") or [] if str(item).strip()]
        steps = [str(item).strip() for item in response.get("steps") or [] if str(item).strip()]
        recommendation = str(response.get("recommendation") or "").strip()
        next_action = str(response.get("next_action") or "").strip()

        if findings:
            body = "\n".join(f"- {item}" for item in findings) if keep_structure else " ".join(findings)
            if recommendation and keep_structure:
                body = f"{body}\n\nNext: {recommendation}".strip()
            return ChatPresenter._clean_user_text(body, keep_structure=keep_structure)
        if steps:
            body = "\n".join(f"- {item}" for item in steps) if keep_structure else " ".join(steps)
            if next_action and keep_structure:
                body = f"{body}\n\nNext: {next_action}".strip()
            return ChatPresenter._clean_user_text(body, keep_structure=keep_structure)

        stripped = ChatPresenter._clean_user_text(rendered, keep_structure=keep_structure)
        if stripped and not ChatPresenter.is_internal_scaffold(stripped):
            return stripped
        return ChatPresenter._fallback_user_text(user_prompt=user_prompt)

    @staticmethod
    def _user_prompt(response: dict[str, object]) -> str | None:
        wake_resolution = response.get("wake_resolution")
        if isinstance(wake_resolution, dict):
            prompt = str(wake_resolution.get("original_prompt") or "").strip()
            if prompt:
                return prompt
        for key in ("original_prompt", "user_prompt", "prompt"):
            prompt = str(response.get(key) or "").strip()
            if prompt:
                return prompt
        return None

    @staticmethod
    def _prompt_allows_structure(user_prompt: str | None) -> bool:
        normalized = " ".join(str(user_prompt or "").strip().lower().split())
        if not normalized:
            return False
        return any(
            token in normalized
            for token in (
                "plan",
                "design",
                "steps",
                "step by step",
                "walk me through",
                "compare",
                "list",
                "outline",
                "brainstorm",
            )
        )

    @staticmethod
    def _clean_user_text(text: str, *, keep_structure: bool) -> str:
        cleaned_lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if any(phrase in lowered for phrase in ChatPresenter._SCAFFOLD_PHRASES):
                continue
            if any(lowered.startswith(prefix) for prefix in ChatPresenter._SCAFFOLD_LINE_PREFIXES):
                continue
            if lowered.startswith("- ") and not keep_structure:
                line = line[2:].strip()
            cleaned_lines.append(line)
        if not cleaned_lines:
            return ""
        if keep_structure:
            return "\n".join(cleaned_lines).strip()
        return " ".join(cleaned_lines).strip()

    @staticmethod
    def _fallback_user_text(*, user_prompt: str | None) -> str:
        normalized = " ".join(str(user_prompt or "").strip().lower().split())
        if normalized in {"good job", "nice job", "well done", "thanks", "thank you"}:
            return "Thanks. I'm glad that helped."
        if normalized.startswith("why"):
            return "Here is the direct reason."
        return "Let me answer that more directly."
