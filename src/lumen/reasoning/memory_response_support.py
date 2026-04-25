from __future__ import annotations

from lumen.reasoning.memory_retrieval_layer import MemoryRetrievalResult


class MemoryResponseSupport:
    """Applies retrieved memory to response payloads without changing route selection."""

    @staticmethod
    def attach_memory_retrieval(
        *,
        response: dict[str, object],
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> None:
        if memory_retrieval is None:
            return
        response["memory_retrieval"] = memory_retrieval.to_dict()

    @staticmethod
    def apply_memory_recall_surface(
        *,
        response: dict[str, object],
        memory_retrieval: MemoryRetrievalResult | None,
    ) -> None:
        if memory_retrieval is None:
            return
        if memory_retrieval.project_return_prompt:
            project_hint = str(memory_retrieval.project_reply_hint or "").strip()
            if project_hint:
                response["project_memory_hint"] = project_hint
                mode = str(response.get("mode") or "").strip()
                if mode == "planning":
                    summary = str(response.get("summary") or "").strip()
                    if summary and project_hint.lower() not in summary.lower():
                        merged = f"{project_hint} {summary}".strip()
                        response["summary"] = merged
                        response["reply"] = merged
                elif mode == "research":
                    answer = str(
                        response.get("user_facing_answer")
                        or response.get("reply")
                        or response.get("summary")
                        or ""
                    ).strip()
                    if answer and project_hint.lower() not in answer.lower():
                        merged = f"{project_hint} {answer}".strip()
                        response["user_facing_answer"] = merged
                        response["summary"] = merged
                        response["reply"] = merged
        if not memory_retrieval.recall_prompt:
            return
        reply_hint = str(memory_retrieval.memory_reply_hint or "").strip()
        if not reply_hint:
            return
        response["memory_reply_hint"] = reply_hint
        response["user_facing_answer"] = reply_hint
        response["summary"] = reply_hint
        response["reply"] = reply_hint

        mode = str(response.get("mode") or "").strip()
        if mode == "research":
            response["findings"] = []
            response.pop("recommendation", None)
        if mode == "planning":
            response["steps"] = []
            response.pop("next_action", None)
        response.pop("conversation_turn", None)
        response.pop("response_intro", None)
        response.pop("response_opening", None)
