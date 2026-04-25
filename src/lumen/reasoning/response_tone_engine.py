from __future__ import annotations

from lumen.reasoning.response_variation import ResponseVariationLayer


class ResponseToneEngine:
    """Owns tone shaping for conversational turn leads and thread-holding frames."""

    @staticmethod
    def _style(style: str) -> str:
        normalized = str(style or "default").strip().lower()
        if normalized == "conversational":
            return "collab"
        if normalized == "direct":
            return "direct"
        if normalized in {"direct", "default", "collab"}:
            return normalized
        return "default"

    @staticmethod
    def question_turn_lead(
        question: str,
        *,
        style: str,
        adaptive_posture: str,
        anti_spiral_active: bool,
        emotional_alignment: str = "",
        correction_detected: bool = False,
        epistemic_stance: str = "",
        stance_confidence: str = "",
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        if correction_detected:
            if style == "direct":
                return "Got it. What direction did you mean instead?"
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base="what direction were you thinking instead?",
                seed_parts=[style, question, "question", "correction"],
                recent_texts=recent_texts,
            )
        if anti_spiral_active:
            lead = ResponseVariationLayer.realize(
                parts=[
                    ("opener", ("Let's slow this down", "Let's bring the pace down", "Let's slow the pace for a second")),
                    ("connector", ("and pin down one thing first:", "and lock down one thing first:", "and clarify one thing first:")),
                ],
                seed_parts=[style, adaptive_posture, question, "question", "anti_spiral"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if style == "direct":
            lead = ResponseVariationLayer.select_from_pool(
                ("Need to pin this down:", "Need to clarify this first:", "One thing to pin down first:"),
                seed_parts=[style, adaptive_posture, question, "question", "straight"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if adaptive_posture == "step_back":
            lead = ResponseVariationLayer.realize(
                parts=[
                    ("opener", ("Before we push this further,", "Before we move too fast,", "Before we lean on this too hard,")),
                    ("connector", ("I'd want to ask:", "I'd want to pin this down:", "I'd want to clarify this first:")),
                ],
                seed_parts=[style, adaptive_posture, question, "question", "step_back"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if epistemic_stance == "assertive" and stance_confidence == "high":
            lead = ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base="I see the line you're drawing.",
                seed_parts=[style, question, "question", "assertive"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        lead = ResponseVariationLayer.select_from_pool(
            (
                "The first thing I'd want to pin down is this:",
                "The first thing I'd want to clarify is this:",
                "The first thing I'd want to lock down is this:",
            ),
            seed_parts=[style, adaptive_posture, question, "question", "default"],
            recent_texts=recent_texts,
        )
        return f"{lead} {question}"

    @staticmethod
    def challenge_turn_lead(
        question: str,
        *,
        style: str,
        adaptive_posture: str,
        anti_spiral_active: bool,
        emotional_alignment: str = "",
        correction_detected: bool = False,
        epistemic_stance: str = "",
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        if correction_detected:
            return "Got it. Let me pressure-test the new direction:"
        if anti_spiral_active:
            lead = ResponseVariationLayer.select_from_pool(
                (
                    "Before we push harder, here's the part I think we need to test:",
                    "Before we lean harder on this, here's the part I think we need to test:",
                    "Before we commit harder, here's the part I think we need to test:",
                ),
                seed_parts=[style, adaptive_posture, question, "challenge", "anti_spiral"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if style == "direct":
            lead = ResponseVariationLayer.select_from_pool(
                ("Gap to test:", "Assumption to test:", "Weak point to test:"),
                seed_parts=[style, adaptive_posture, question, "challenge", "straight"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if adaptive_posture == "push":
            lead = ResponseVariationLayer.select_from_pool(
                (
                    "This is interesting, but I think there's a gap here:",
                    "The direction is interesting, but I think there's a gap here:",
                    "I like where this is going, but I think there's a gap here:",
                ),
                seed_parts=[style, adaptive_posture, question, "challenge", "push"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        if emotional_alignment == "calm_supportive" and style != "direct":
            lead = ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base="yeah, there's a real gap here:",
                seed_parts=[style, question, "challenge", "frustrated"],
                recent_texts=recent_texts,
            )
            return f"{lead} {question}"
        lead = ResponseVariationLayer.select_from_pool(
            (
                "I like the direction, but let's test one assumption:",
                "I like the line of thought, but let's test one assumption:",
                "I like where this is going, but let's test one assumption:",
            ),
            seed_parts=[style, adaptive_posture, question, "challenge", "default"],
            recent_texts=recent_texts,
        )
        return f"{lead} {question}"

    @staticmethod
    def answer_turn_lead(
        *,
        style: str,
        deep_collaboration: bool,
        adaptive_posture: str,
        anti_spiral_active: bool,
        tone_profile: str | None = None,
        emotional_alignment: str = "",
        user_energy: str = "",
        epistemic_stance: str = "",
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        if anti_spiral_active:
            if style == "direct":
                return ResponseVariationLayer.select_from_pool(
                    ("Let's stabilize this first:", "Let's steady this first:", "Let's ground this first:"),
                    seed_parts=[style, adaptive_posture, "answer", "anti_spiral"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                (
                    "Let's slow it down and separate what's clear from what's still unresolved.",
                    "Let's slow it down and separate what's supported from what's still unresolved.",
                    "Let's bring the pace down and separate what's clear from what's still unresolved.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "anti_spiral"],
                recent_texts=recent_texts,
            )
        if style == "direct":
            if adaptive_posture == "step_back":
                return ResponseVariationLayer.select_from_pool(
                    ("Best read, provisional:", "Best read, still provisional:", "Best read, lightly held:"),
                    seed_parts=[style, adaptive_posture, "answer", "step_back"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                ("Best read:", "Best read, as it stands:", "Best read right now:")
                if not deep_collaboration
                else ("Best read. Weak point worth testing:", "Best read. One weak point worth testing:", "Best read. Main weak point to test:"),
                seed_parts=[style, adaptive_posture, str(deep_collaboration), "answer", "straight"],
                recent_texts=recent_texts,
            )
        if deep_collaboration:
            base = ResponseVariationLayer.select_from_pool(
                (
                    "Here's my read so far, and we can pressure-test the weak point together.",
                    "Here's my read so far, and we can test the weak point together.",
                    "Here's my read so far, and we can stress-test the weak point together.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "deep"],
                recent_texts=recent_texts,
            )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base=base,
                seed_parts=[adaptive_posture, "answer", "deep", "bridge"],
                recent_texts=recent_texts,
            )
        if style == "default":
            if adaptive_posture == "step_back":
                base = ResponseVariationLayer.select_from_pool(
                    (
                        "Here's the clearest grounded read I can give right now, but I'd still hold it lightly.",
                        "Here's the clearest grounded read I can give so far, but I'd still hold it lightly.",
                        "Here's the clearest grounded read so far, but I'd still keep it provisional.",
                        "Here's the cleanest grounded read I can give so far, though I'd still keep it a little provisional.",
                    ),
                    seed_parts=[style, adaptive_posture, "answer", "default_midlane_step_back"],
                    recent_texts=recent_texts,
                )
                return ResponseVariationLayer.style_bridge(
                    style=style,
                    bridge_type="thinking",
                    base=base,
                    seed_parts=[adaptive_posture, "answer", "default_step_back", "bridge"],
                    recent_texts=recent_texts,
                )
            base = ResponseVariationLayer.select_from_pool(
                (
                    "Here's the clearest grounded read so far.",
                    "Here's the clearest grounded read I can give so far.",
                    "Here's the answer in clear terms.",
                    "Alright, here's the answer in clear terms.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "default_midlane"],
                recent_texts=recent_texts,
            )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="thinking",
                base=base,
                seed_parts=[adaptive_posture, "answer", "default", "bridge"],
                recent_texts=recent_texts,
            )
        if tone_profile == "formal_explanation":
            return ResponseVariationLayer.select_from_pool(
                (
                    "Here's a direct explanation.",
                    "Here's the clearest explanation.",
                    "Here's the explanation in plain terms.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "formal_explanation"],
                recent_texts=recent_texts,
            )
        if tone_profile == "casual_explanation":
            base = ResponseVariationLayer.select_from_pool(
                (
                    "Sure. Here's the quick version.",
                    "Alright. Here's the quick shape of it.",
                    "Sure. Here's the clearest version.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "casual_explanation"],
                recent_texts=recent_texts,
            )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="thinking" if user_energy == "casual" else "acknowledgment",
                base=base,
                seed_parts=[adaptive_posture, "answer", "casual_explanation", "bridge"],
                recent_texts=recent_texts,
            )
        if adaptive_posture == "step_back":
            base = ResponseVariationLayer.select_from_pool(
                (
                    "Here's my read so far, and I'd still keep it light until we pin down the open part.",
                    "Here's my read so far, and I'd still keep it provisional until the open part is clearer.",
                    "Here's where my read lands so far, and I'd still keep it light until we pin down the open part.",
                    "Okay, here's where my read lands so far, and I'd still keep it a little light.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "step_back"],
                recent_texts=recent_texts,
            )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="softened",
                base=base,
                seed_parts=[adaptive_posture, "answer", "collab_step_back", "bridge"],
                recent_texts=recent_texts,
            )
        if adaptive_posture == "acknowledge":
            base = ResponseVariationLayer.select_from_pool(
                (
                    "That tracks. Here's my read so far.",
                    "That makes sense. Here's my read so far.",
                    "That tracks. Here's where my read lands right now.",
                    "Yeah, that tracks. Here's where I'd start with it.",
                ),
                seed_parts=[style, adaptive_posture, "answer", "acknowledge"],
                recent_texts=recent_texts,
            )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base=base,
                seed_parts=[adaptive_posture, "answer", "acknowledge", "bridge"],
                recent_texts=recent_texts,
            )
        base = ResponseVariationLayer.select_from_pool(
            (
                "Here's my read so far, and we can keep shaping it together.",
                "Here's my read so far, and we can keep refining it together.",
                "Here's my read so far, and we can keep working the shape of it together.",
                "Alright, here's where I'd start, and we can keep shaping it from there together.",
            ),
            seed_parts=[style, adaptive_posture, "answer", "default"],
            recent_texts=recent_texts,
        )
        return ResponseVariationLayer.style_bridge(
            style=style,
            bridge_type="thinking" if epistemic_stance == "exploratory" else "acknowledgment",
            base=base,
            seed_parts=[adaptive_posture, "answer", "collab", "bridge"],
            recent_texts=recent_texts,
        )

    @staticmethod
    def checkpoint_turn_lead(
        *,
        style: str,
        state_core: str,
        anti_spiral_active: bool,
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        if anti_spiral_active:
            if style == "direct":
                return ResponseVariationLayer.select_from_pool(
                    ("Current state:", "State check:", "Current picture:"),
                    seed_parts=[style, state_core, "checkpoint", "anti_spiral"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                (
                    "Let's reset the picture for a second.",
                    "Let's reset the picture for a moment.",
                    "Let's reset the frame for a second.",
                ),
                seed_parts=[style, state_core, "checkpoint", "anti_spiral"],
                recent_texts=recent_texts,
            )
        if state_core == "momentum":
            if style == "direct":
                return ResponseVariationLayer.select_from_pool(
                    ("Current state:", "State check:", "Current picture:"),
                    seed_parts=[style, state_core, "checkpoint", "momentum"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                (
                    "We're onto something. Here's where we are.",
                    "There's real movement here. Here's where we are.",
                    "We have some momentum now. Here's where we are.",
                ),
                seed_parts=[style, state_core, "checkpoint", "momentum"],
                recent_texts=recent_texts,
            )
        if state_core == "curiosity":
            if style == "direct":
                return ResponseVariationLayer.select_from_pool(
                    ("Current state:", "State check:", "Current picture:"),
                    seed_parts=[style, state_core, "checkpoint", "curiosity"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                (
                    "Here's where the idea is opening up.",
                    "Here's where the idea starts to open up.",
                    "Here's where the line of thought is opening up.",
                ),
                seed_parts=[style, state_core, "checkpoint", "curiosity"],
                recent_texts=recent_texts,
            )
        if style == "direct":
            return ResponseVariationLayer.select_from_pool(
                ("Current state:", "State check:", "Current picture:"),
                seed_parts=[style, state_core, "checkpoint", "default"],
                recent_texts=recent_texts,
            )
        base = ResponseVariationLayer.select_from_pool(
            ("Here's where we are.", "Here's the current shape of it.", "Here's where it stands right now."),
            seed_parts=[style, state_core, "checkpoint", "default"],
            recent_texts=recent_texts,
        )
        return ResponseVariationLayer.style_bridge(
            style=style,
            bridge_type="transition",
            base=base,
            seed_parts=[state_core, "checkpoint", "bridge"],
            recent_texts=recent_texts,
        )

    @staticmethod
    def thread_holding_frame(
        *,
        style: str,
        deep_collaboration: bool,
        strategy: str,
        interaction_mode: str,
        adaptive_posture: str,
        unresolved_thread_open: bool,
        branch_state: str,
        return_target: str,
        checkpoint_summary: dict[str, object] | None,
        anti_spiral_active: bool,
        tone_profile: str | None = None,
        emotional_alignment: str = "",
        user_energy: str = "",
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        if anti_spiral_active:
            if style == "direct":
                return ResponseVariationLayer.select_from_pool(
                    (
                        "Pause escalation. Keep the answer grounded in what is actually known and what is still missing.",
                        "Pause escalation. Keep this grounded in what is known and what is still missing.",
                        "Pause escalation. Keep the line anchored in what is known and what is still missing.",
                    ),
                    seed_parts=[style, strategy, interaction_mode, "frame", "anti_spiral"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.select_from_pool(
                (
                    "I want to keep this grounded, name what is actually supported, and avoid outrunning the evidence.",
                    "I want to keep this grounded, name what is actually supported, and not outrun the evidence.",
                    "I want to keep this grounded, say what is actually supported, and avoid outrunning the evidence.",
                ),
                seed_parts=[style, strategy, interaction_mode, "frame", "anti_spiral"],
                recent_texts=recent_texts,
            )
        if style == "direct":
            if branch_state == "returning_to_main":
                return "Rejoin the main thread, re-anchor the live question, then move it forward."
            if checkpoint_summary:
                return "State of play, strongest signal, weakest point, next move."
            if strategy == "challenge":
                return "Best next move is to test the weak assumption directly."
            if strategy == "ask_question":
                return "Need one answer before pushing further."
            if strategy == "expand":
                return "Open the idea, then choose the strongest line."
            if strategy == "answer":
                if adaptive_posture == "step_back":
                    return "Hold the line lightly until the open part is clearer."
                return "Best read first, then the next move."
            return "Keep the thread tight and move it forward."
        if checkpoint_summary:
            if deep_collaboration:
                base = "Let me pull the threads together so we can see what is holding, what is opening up, and what to test next."
                return ResponseVariationLayer.style_bridge(
                    style=style,
                    bridge_type="thinking",
                    base=base,
                    seed_parts=[strategy, interaction_mode, "frame", "checkpoint_deep"],
                    recent_texts=recent_texts,
                )
            base = "Let me pull the threads together so we can see what still holds and what still needs work."
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="thinking",
                base=base,
                seed_parts=[strategy, interaction_mode, "frame", "checkpoint"],
                recent_texts=recent_texts,
            )
        if branch_state == "returning_to_main":
            if return_target:
                return f"Let's re-anchor on the main thread, keep {return_target} in view, and move it forward cleanly."
            return "Let's re-anchor on the main thread and move it forward cleanly."
        if strategy == "challenge":
            return "Two minds are better than one here, so I want to push on the part most likely to break."
        if strategy == "ask_question":
            return "Before we commit too hard, I want to pin down the moving part that matters most."
        if strategy == "expand":
            if deep_collaboration:
                return "Let's widen the idea a little, then decide which line is actually worth keeping."
            return "Let's keep the thread open long enough to see which direction earns more confidence."
        if strategy == "answer":
            if deep_collaboration:
                return "I'll give you my read so far, but we should keep the live uncertainty in view."
            if style == "default":
                if adaptive_posture == "step_back":
                    return ResponseVariationLayer.style_bridge(
                        style=style,
                        bridge_type="thinking",
                        base="I'll keep this clear and grounded, but I don't want to overstate the uncertain part.",
                        seed_parts=[strategy, interaction_mode, adaptive_posture, "frame", "default_step_back"],
                        recent_texts=recent_texts,
                    )
                return ResponseVariationLayer.style_bridge(
                    style=style,
                    bridge_type="thinking",
                    base="I'll keep it clear, grounded, and easy to follow.",
                    seed_parts=[strategy, interaction_mode, adaptive_posture, "frame", "default"],
                    recent_texts=recent_texts,
                )
            if tone_profile == "formal_explanation":
                return "I'll explain it directly, keep it grounded, and point to the next useful question."
            if tone_profile == "casual_explanation":
                return "I'll keep it easy to track, give you the explanation, and leave room for a natural follow-up."
            if adaptive_posture == "step_back":
                return ResponseVariationLayer.style_bridge(
                    style=style,
                    bridge_type="softened",
                    base="I'll keep the thread open, but I don't want to overcommit before the missing piece is clearer.",
                    seed_parts=[strategy, interaction_mode, adaptive_posture, "frame", "collab_step_back"],
                    recent_texts=recent_texts,
                )
            if adaptive_posture == "acknowledge" and unresolved_thread_open:
                return ResponseVariationLayer.style_bridge(
                    style=style,
                    bridge_type="acknowledgment",
                    base="I'll acknowledge where we are, keep the open thread visible, and move it one careful step forward.",
                    seed_parts=[strategy, interaction_mode, adaptive_posture, "frame", "collab_ack"],
                    recent_texts=recent_texts,
                )
            return ResponseVariationLayer.style_bridge(
                style=style,
                bridge_type="acknowledgment",
                base="I'll give you my read, keep the thread in view, and push it one step further.",
                seed_parts=[strategy, interaction_mode, adaptive_posture, "frame", "collab"],
                recent_texts=recent_texts,
            )
        if interaction_mode == "analytical":
            return "Let's work the structure carefully and keep the reasoning anchored."
        base = "Let's work this through together and keep it grounded."
        return ResponseVariationLayer.style_bridge(
            style=style,
            bridge_type="thinking" if user_energy == "casual" else "acknowledgment",
            base=base,
            seed_parts=[strategy, interaction_mode, "frame", "default_tail"],
            recent_texts=recent_texts,
        )

    @staticmethod
    def thread_holding_next_move(
        *,
        style: str,
        turn: dict[str, object],
        checkpoint_summary: dict[str, object] | None,
        recent_texts: list[str] | None = None,
    ) -> str | None:
        if checkpoint_summary:
            return str(checkpoint_summary.get("next_step") or "").strip() or None
        follow_ups = list(turn.get("follow_ups") or [])
        if follow_ups:
            return follow_ups[0]
        if style == "direct":
            return ResponseVariationLayer.select_from_pool(
                (
                    "Use the current line, then verify the weak point.",
                    "Use the current line, then test the weak point.",
                    "Stay with the current line, then verify the weak point.",
                ),
                seed_parts=[style, str(turn.get("kind") or ""), "next_move"],
                recent_texts=recent_texts,
            )
        return None

    @staticmethod
    def pickup_bridge(
        *,
        style: str,
        category: str,
        target: str,
        recent_texts: list[str] | None = None,
    ) -> str:
        style = ResponseToneEngine._style(style)
        generic_target = target.lower().strip() in {
            "the first one",
            "first one",
            "the second one",
            "second one",
            "that one",
            "that",
        }
        if style == "direct":
            pools = {
                "direct_acceptance": ("Okay.", "Alright.", "Good."),
                "soft_acceptance": ("Okay.", "Fair.", "Alright."),
                "directional_acceptance": ("Okay.", "Good.", "Alright."),
                "hesitant_acceptance": ("Alright.", "Fair.", "Okay."),
                "collaborative_pickup": ("Okay.", "Alright.", "Good."),
            }
        elif style == "collab":
            pools = {
                "direct_acceptance": (
                    "Nice, yeah - let's go there.",
                    "Okay, I'm with you.",
                    "Yeah, let's do that.",
                    "That works - let's start there.",
                ),
                "soft_acceptance": (
                    "Yeah, that works.",
                    "Okay, that feels like a good direction.",
                    "That makes sense - let's run with it.",
                ),
                "directional_acceptance": (
                    "Perfect, that's a good direction.",
                    "Nice, let's take that one.",
                    "Yeah, let's run with that.",
                ),
                "hesitant_acceptance": (
                    "Hmm, yeah, let's take that one.",
                    "Maybe, yeah - let's go there.",
                    "Okay, I can work with that.",
                ),
                "collaborative_pickup": (
                    "Nice, I'm with you.",
                    "Okay, let's go there.",
                    "Yeah, let's stay with that.",
                ),
            }
        else:
            pools = {
                "direct_acceptance": (
                    "Alright, let's go there.",
                    "Okay, let's do that.",
                    "That works - let's start there.",
                ),
                "soft_acceptance": (
                    "Okay, that works.",
                    "Alright, that's a good direction.",
                    "That makes sense - let's go with it.",
                ),
                "directional_acceptance": (
                    "Alright, let's go with that.",
                    "Okay, let's take that direction.",
                    "That works - let's run with that.",
                ),
                "hesitant_acceptance": (
                    "Alright, let's take that one.",
                    "Okay, we can go there.",
                    "That works - let's start there.",
                ),
                "collaborative_pickup": (
                    "Okay, I'm with you.",
                    "Alright, let's go there.",
                    "That works - let's start there.",
                ),
            }
        pool = pools.get(category, pools["collaborative_pickup"])
        choice = ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style, category, target, "pickup_bridge"],
            recent_texts=recent_texts,
        )
        if generic_target:
            return choice
        return choice

    @staticmethod
    def follow_through_starter(
        *,
        style: str,
        target: str,
        recent_texts: list[str] | None = None,
    ) -> str | None:
        style = ResponseToneEngine._style(style)
        cleaned = str(target or "").strip()
        if not cleaned:
            return None
        generic_target = cleaned.lower() in {
            "the first one",
            "first one",
            "the second one",
            "second one",
            "that one",
            "that",
        }
        if style == "direct":
            pool = (
                "Start here:" if generic_target else f"First thing with {cleaned} is this:",
                "Start with this:" if generic_target else f"Start with {cleaned}:",
            )
        elif style == "collab":
            pool = (
                "Okay, let's stay with that for a second." if generic_target else f"So with {cleaned}, the key thing is this.",
                "Nice - here's where I'd start." if generic_target else f"Alright, first thing with {cleaned} is this.",
                "Let's break that open a bit." if generic_target else f"Okay, let's stay with {cleaned} for a second.",
            )
        else:
            pool = (
                "Alright, let's stay with that for a second." if generic_target else f"Alright, first thing with {cleaned} is this.",
                "Here's where I'd start." if generic_target else f"So the key thing with {cleaned} is this.",
                "Let's start there." if generic_target else f"Okay, let's start with {cleaned}.",
            )
        return ResponseVariationLayer.select_from_pool(
            pool,
            seed_parts=[style, cleaned, "follow_through_starter"],
            recent_texts=recent_texts,
        )

    @staticmethod
    def response_body_opening(*, turn: dict[str, object], style: str = "default") -> str | None:
        follow_through = str(turn.get("follow_through_starter") or "").strip()
        if follow_through:
            return follow_through
        partner_frame = str(turn.get("partner_frame") or "").strip()
        if partner_frame:
            return partner_frame
        next_move = str(turn.get("next_move") or "").strip()
        if next_move:
            normalized_style = ResponseToneEngine._style(style)
            if normalized_style == "direct":
                return f"Next move: {next_move}"
            if normalized_style == "collab":
                return f"Next step together: {next_move}"
            return f"Next move: {next_move}"
        return None

    @staticmethod
    def checkpoint_body_items(
        *,
        style: str,
        current_direction: str,
        strongest_point: str,
        weakest_point: str,
        open_question: str,
        recent_texts: list[str] | None = None,
    ) -> list[str]:
        if style == "direct":
            items: list[str] = []
            if current_direction:
                items.append(f"Current direction: {current_direction}")
            if strongest_point:
                items.append(f"Strongest point: {strongest_point}")
            if weakest_point:
                items.append(f"Weakest point: {weakest_point}")
            if open_question:
                items.append(f"Open question: {open_question}")
            return items

        direction_openers = (
            "Right now, the line we're following is",
            "Right now, the direction we're following is",
            "At the moment, the line we're following is",
        )
        strongest_openers = (
            "The strongest point so far is",
            "The strongest point right now is",
            "The strongest signal so far is",
        )
        weak_openers = (
            "The weak point is",
            "The weak point right now is",
            "The pressure point is",
        )
        question_openers = (
            "The open question is",
            "The main open question is",
            "The unresolved question is",
        )
        items = []
        if current_direction:
            opener = ResponseVariationLayer.select_from_pool(
                direction_openers,
                seed_parts=[style, current_direction, "checkpoint_body", "direction"],
                recent_texts=recent_texts,
            )
            items.append(f"{opener} {current_direction}")
        if strongest_point:
            opener = ResponseVariationLayer.select_from_pool(
                strongest_openers,
                seed_parts=[style, strongest_point, "checkpoint_body", "strongest"],
                recent_texts=recent_texts,
            )
            items.append(f"{opener} {strongest_point}")
        if weakest_point:
            opener = ResponseVariationLayer.select_from_pool(
                weak_openers,
                seed_parts=[style, weakest_point, "checkpoint_body", "weakest"],
                recent_texts=recent_texts,
            )
            items.append(f"{opener} {weakest_point}")
        if open_question:
            opener = ResponseVariationLayer.select_from_pool(
                question_openers,
                seed_parts=[style, open_question, "checkpoint_body", "question"],
                recent_texts=recent_texts,
            )
            items.append(f"{opener} {open_question}")
        return items

    @staticmethod
    def live_question_item(*, style: str, question: str, recent_texts: list[str] | None = None) -> str:
        if style == "direct":
            opener = ResponseVariationLayer.select_from_pool(
                ("Live question:", "Current live question:", "Question in play:"),
                seed_parts=[style, question, "live_question"],
                recent_texts=recent_texts,
            )
            return f"{opener} {question}"
        opener = ResponseVariationLayer.select_from_pool(
            ("The live question is", "The question still in play is", "The live question right now is"),
            seed_parts=[style, question, "live_question"],
            recent_texts=recent_texts,
        )
        return ResponseVariationLayer.style_bridge(
            style=style,
            bridge_type="thinking" if style == "default" else "transition",
            base=f"{opener} {question}",
            seed_parts=[question, "live_question", "bridge"],
            recent_texts=recent_texts,
        )

