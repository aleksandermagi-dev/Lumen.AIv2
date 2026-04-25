from __future__ import annotations

from lumen.reasoning.interaction_style_policy import InteractionStylePolicy


class ModeResponseShaper:
    """Applies lightweight mode-consistent delivery shaping after reasoning is complete."""

    @staticmethod
    def apply(
        *,
        response: dict[str, object],
        interaction_profile,
    ) -> None:
        style = ModeResponseShaper._style(response=response, interaction_profile=interaction_profile)
        response["mode_nlg_profile"] = {
            "mode": style,
            "follow_up_style": (
                "exploratory" if style == "collab" else ("minimal" if style == "direct" else "balanced")
            ),
            "delivery_style": (
                "partnered" if style == "collab" else ("concise" if style == "direct" else "clear")
            ),
            **InteractionStylePolicy.voice_profile({"interaction_style": style}),
            "reasoning_depth_separate": True,
        }
        ModeResponseShaper._shape_tool_surface(response=response, style=style)
        ModeResponseShaper._attach_follow_up_offer(response=response, style=style)

    @staticmethod
    def _style(*, response: dict[str, object], interaction_profile) -> str:
        reasoning_state = response.get("reasoning_state")
        if isinstance(reasoning_state, dict):
            selected_mode = str(reasoning_state.get("selected_mode") or "").strip().lower()
            if selected_mode in {"collab", "default", "direct"}:
                return selected_mode
        style = str(getattr(interaction_profile, "interaction_style", "") or "").strip().lower()
        if style == "conversational":
            return "collab"
        if style in {"collab", "default", "direct"}:
            return style
        return "default"

    @staticmethod
    def _shape_tool_surface(*, response: dict[str, object], style: str) -> None:
        if str(response.get("mode") or "").strip() != "tool":
            return
        if response.get("tool_execution_skipped") is True:
            missing = str(response.get("tool_missing_inputs") or "required inputs").strip()
            summary = str(response.get("summary") or "").strip()
            tail = ModeResponseShaper._tail_after_first_sentence(summary)
            if style == "collab":
                lead = f"I'm ready to do that, but I need usable {missing} first."
                body = f"{lead} {tail}".strip() if tail else f"{lead} Once you give me that, I'll carry it the rest of the way."
            elif style == "direct":
                lead = f"Need usable {missing} before tool run."
                body = f"{lead} {tail}".strip() if tail else lead
            else:
                body = summary or f"I need usable {missing} before I run the tool cleanly."
            response["user_facing_answer"] = body.strip()
            response["reply"] = body.strip()
            return
        runtime = response.get("tool_runtime_status")
        if not isinstance(runtime, dict):
            return
        failure_class = str(runtime.get("failure_class") or "").strip()
        if not failure_class or failure_class == "success":
            return
        diagnostic = response.get("runtime_diagnostic")
        if not isinstance(diagnostic, dict):
            diagnostic = {}
        failure_stage = str(diagnostic.get("failure_stage") or "execution").strip()
        exception_type = str(diagnostic.get("exception_type") or "").strip()
        summary = str(response.get("summary") or response.get("reply") or "").strip()
        if not summary:
            return
        if style == "collab":
            stage = f"{failure_stage} stage" if failure_stage else "runtime"
            article = "an" if stage[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            surfaced = f"I ran into {article} {stage} issue here. {summary}"
        elif style == "direct":
            surfaced = f"{summary} ({exception_type})" if exception_type and exception_type not in summary else summary
        else:
            label = f"{failure_stage.title()} issue" if failure_stage else "Runtime issue"
            surfaced = f"{label}: {summary}"
        response["user_facing_answer"] = surfaced.strip()
        response["reply"] = surfaced.strip()

    @staticmethod
    def _attach_follow_up_offer(*, response: dict[str, object], style: str) -> None:
        if response.get("follow_up_offer"):
            return
        mode = str(response.get("mode") or "").strip()
        if mode not in {"research", "planning", "tool"}:
            return
        if mode == "tool" and (
            response.get("tool_execution_skipped") is True
            or (
                isinstance(response.get("tool_runtime_status"), dict)
                and str((response.get("tool_runtime_status") or {}).get("failure_class") or "").strip() not in {"", "success"}
            )
        ):
            return
        if style == "collab":
            response["follow_up_offer"] = "If you want, we can keep pulling on the next layer together."
        elif style == "direct":
            response["follow_up_offer"] = "I can go one layer deeper."
        else:
            response["follow_up_offer"] = "If you want, I can go one layer deeper or compare the tradeoffs."

    @staticmethod
    def _tail_after_first_sentence(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        for separator in (". ", "? ", "! "):
            if separator in cleaned:
                return cleaned.split(separator, 1)[1].strip()
        return ""
