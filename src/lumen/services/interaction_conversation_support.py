from __future__ import annotations

from lumen.reasoning.response_tone_engine import ResponseToneEngine
from lumen.reasoning.response_variation import ResponseVariationLayer


class InteractionConversationSupport:
    """Conversation-specific shaping helpers extracted from InteractionService."""

    @classmethod
    def shape_reasoning_body_from_turn(
        cls,
        *,
        response: dict[str, object],
        interaction_style: str,
        allow_internal_scaffold: bool,
    ) -> None:
        if interaction_style not in {"default", "collab", "direct"}:
            return
        conversation_turn = response.get("conversation_turn") or {}
        intro = str(response.get("response_intro") or "").strip()
        opening = str(response.get("response_opening") or "").strip()
        if not intro and not opening:
            return
        mode = str(response.get("mode") or "").strip()
        if mode in {"planning", "research"} and not allow_internal_scaffold:
            return
        if mode == "planning":
            existing = list(response.get("steps") or [])
            response["steps"] = cls.prepend_conversation_items(
                existing_items=existing,
                intro=intro,
                opening=opening,
                turn=conversation_turn,
                style=interaction_style,
                allow_internal_scaffold=allow_internal_scaffold,
                limit=5 if interaction_style == "direct" else None,
            )
            next_action = str(response.get("next_action") or "").strip()
            shaped_next = cls.shape_closeout_from_turn(
                existing_closeout=next_action,
                turn=conversation_turn,
                style=interaction_style,
                label="Next move",
            )
            if shaped_next:
                response["next_action"] = shaped_next
        elif mode == "research":
            existing = list(response.get("findings") or [])
            response["findings"] = cls.prepend_conversation_items(
                existing_items=existing,
                intro=intro,
                opening=opening,
                turn=conversation_turn,
                style=interaction_style,
                allow_internal_scaffold=allow_internal_scaffold,
                limit=5 if interaction_style == "direct" else None,
            )
            recommendation = str(response.get("recommendation") or "").strip()
            shaped_recommendation = cls.shape_closeout_from_turn(
                existing_closeout=recommendation,
                turn=conversation_turn,
                style=interaction_style,
                label="Next move",
            )
            if shaped_recommendation:
                response["recommendation"] = shaped_recommendation

    @classmethod
    def prepend_conversation_items(
        cls,
        *,
        existing_items: list[str],
        intro: str,
        opening: str,
        turn: dict[str, object],
        style: str,
        allow_internal_scaffold: bool,
        limit: int | None = None,
    ) -> list[str]:
        shaped = list(existing_items)
        inserts: list[str] = []
        if intro and (not shaped or shaped[0] != intro):
            inserts.append(intro)
        if opening and opening not in shaped[:2]:
            inserts.append(opening)
        if allow_internal_scaffold:
            inserts.extend(
                cls.turn_body_items(
                    turn=turn,
                    existing_items=shaped,
                    style=style,
                )
            )
        result = inserts + shaped if inserts else shaped
        return result[:limit] if limit is not None else result

    @staticmethod
    def turn_body_items(
        *,
        turn: dict[str, object],
        existing_items: list[str],
        style: str,
    ) -> list[str]:
        items: list[str] = []
        kind = str(turn.get("kind") or "").strip()
        open_questions = list(turn.get("open_questions") or [])
        follow_ups = list(turn.get("follow_ups") or [])

        if kind == "checkpoint":
            strongest_point = str(turn.get("strongest_point") or "").strip()
            weakest_point = str(turn.get("weakest_point") or "").strip()
            current_direction = str(turn.get("current_direction") or "").strip()
            items.extend(
                ResponseToneEngine.checkpoint_body_items(
                    style=style,
                    current_direction=current_direction,
                    strongest_point=strongest_point,
                    weakest_point=weakest_point,
                    open_question=str(open_questions[0]).strip() if open_questions else "",
                    recent_texts=ResponseVariationLayer.recent_surface_texts(
                        [{"steps": existing_items}] if existing_items else []
                    ),
                )
            )
        elif kind in {"question", "challenge", "explore", "summary", "collaborate", "thread_hold"}:
            if follow_ups:
                items.append(
                    ResponseToneEngine.live_question_item(
                        style=style,
                        question=str(follow_ups[0]).strip(),
                        recent_texts=existing_items,
                    )
                )
        branch_return_hint = str(turn.get("branch_return_hint") or "").strip()
        if branch_return_hint:
            items.append(branch_return_hint)

        deduped: list[str] = []
        seen = {item.strip().lower() for item in existing_items}
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped

    @staticmethod
    def shape_closeout_from_turn(
        *,
        existing_closeout: str,
        turn: dict[str, object],
        style: str,
        label: str,
    ) -> str:
        next_move = str(turn.get("next_move") or "").strip()
        if not next_move:
            return existing_closeout
        if style == "direct" or not existing_closeout:
            return f"{label}: {next_move}"
        if next_move.lower() in existing_closeout.lower():
            return existing_closeout
        return f"{existing_closeout} {label}: {next_move}"
