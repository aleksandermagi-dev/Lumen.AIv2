from lumen.app.models import InteractionProfile
from lumen.reasoning.conversational_reply_realizer import (
    ConversationalReplyRealizer,
)


def test_conversational_reply_realizer_builds_state_from_lightweight_social_reply() -> None:
    profile = InteractionProfile(
        interaction_style="default",
        reasoning_depth="normal",
        selection_source="user",
    )
    response = {
        "mode": "conversation",
        "kind": "conversation.check_in",
        "summary": "I'm doing well and ready to help. What are we looking at?",
        "reply": "I'm doing well and ready to help. What are we looking at?",
    }

    state = ConversationalReplyRealizer.build_state(
        response=response,
        interaction_profile=profile,
    )

    assert state is not None
    assert state.lane == "conversational"
    assert state.intent == "conversation.check_in"
    assert state.main_content == "I'm doing well and ready to help. What are we looking at?"


def test_conversational_reply_realizer_builds_state_from_turn_and_uses_question_follow_up() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    response = {
        "mode": "conversation",
        "kind": "conversation.reply",
        "summary": "Hey. What are we looking at?",
        "reply": "Hey. What are we looking at?",
        "conversation_turn": {
            "kind": "collaborate",
            "lead": "There's truth in that, though I'd qualify one part.",
            "next_move": "What part should we pin down next?",
        },
        "stance_consistency": {
            "category": "agreement_with_qualification",
        },
    }

    state = ConversationalReplyRealizer.build_state(
        response=response,
        interaction_profile=profile,
    )

    assert state is not None
    assert state.stance == "agreement_with_qualification"
    assert state.main_content == "There's truth in that, though I'd qualify one part."
    assert state.optional_follow_up == "What part should we pin down next?"

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=[],
    )

    assert realized == (
        "There's truth in that, though I'd qualify one part. "
        "What part should we pin down next?"
    )


def test_conversational_reply_realizer_widens_collab_handoff_choices() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_turn": {
                "kind": "collaborate",
                "lead": "There's something real there.",
                "follow_ups": ["the weaker edge of the idea"],
            },
        },
        interaction_profile=profile,
    )

    assert state is not None

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=[],
    )

    assert realized.startswith("There's something real there.")
    assert any(
        phrase in realized
        for phrase in {
            "keep pulling on the weaker edge of the idea",
            "stay with the weaker edge of the idea",
            "keep going on the weaker edge of the idea",
            "poke at the weaker edge of the idea",
            "keep tugging on the weaker edge of the idea",
        }
    )


def test_conversational_reply_realizer_uses_pickup_bridge_and_follow_through_starter() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_turn": {
                "kind": "collaborate",
                "pickup_bridge": "Nice, yeah - let's go there.",
                "follow_through_starter": "So with black holes, the key thing is this.",
                "lead": "Here's my read so far, and we can keep shaping it.",
                "follow_ups": ["event horizons"],
            },
        },
        interaction_profile=profile,
    )

    assert state is not None
    assert state.pickup_bridge == "Nice, yeah - let's go there."
    assert state.main_content == "So with black holes, the key thing is this."

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=[],
    )

    assert realized.startswith("Nice, yeah - let's go there. So with black holes, the key thing is this.")
    assert "Here's my read so far" not in realized


def test_conversational_reply_realizer_ignores_non_conversation_modes() -> None:
    profile = InteractionProfile.default()
    response = {
        "mode": "research",
        "kind": "research.summary",
        "summary": "Black holes are regions where gravity is so strong that even light cannot escape.",
    }

    state = ConversationalReplyRealizer.build_state(
        response=response,
        interaction_profile=profile,
    )

    assert state is None


def test_conversational_reply_realizer_adds_continuity_presence_in_long_chat() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    recent_interactions = [
        {"mode": "conversation", "kind": "conversation.reply", "summary": "I'm with you on that."},
        {"mode": "conversation", "kind": "conversation.reply", "summary": "We can keep pulling on the weaker edge."},
        {"mode": "conversation", "kind": "conversation.reply", "summary": "Let's stay with that for a second."},
        {"mode": "conversation", "kind": "conversation.reply", "summary": "Okay, the key thing is this."},
    ]
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_turn": {
                "kind": "collaborate",
                "lead": "There's another layer to the routing question here.",
                "follow_ups": ["the memory boundary"],
            },
        },
        interaction_profile=profile,
        recent_interactions=recent_interactions,
        active_thread={"normalized_topic": "routing"},
    )

    assert state is not None
    assert state.stamina_state is not None
    assert state.stamina_state.long_chat is True

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=recent_interactions,
    )

    assert "There's another layer to the routing question here." in realized
    assert any(
        phrase in realized
        for phrase in {
            "I'm still with you on this.",
            "Yeah, let's stay with routing for a second.",
            "We can keep pulling on this from here.",
            "Alright, we're still in it.",
        }
    )


def test_conversational_reply_realizer_does_not_force_stamina_when_recent_modes_are_mixed() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    recent_interactions = [
        {"mode": "conversation", "kind": "conversation.reply", "summary": "I'm with you on that."},
        {"mode": "research", "kind": "research.summary", "summary": "Black holes curve spacetime strongly."},
        {"mode": "conversation", "kind": "conversation.reply", "summary": "Let's stay with that for a second."},
        {"mode": "planning", "kind": "planning.migration", "summary": "Split the routing helpers cleanly."},
    ]
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_turn": {
                "kind": "collaborate",
                "lead": "There's another layer to the routing question here.",
                "follow_ups": ["the memory boundary"],
            },
        },
        interaction_profile=profile,
        recent_interactions=recent_interactions,
        active_thread={"normalized_topic": "routing"},
    )

    assert state is not None
    assert state.stamina_state is not None
    assert state.stamina_state.mixed_recent_modes is True
    assert state.stamina_state.reliable_long_chat is False


def test_conversational_reply_realizer_suppresses_repeated_long_chat_follow_up_offer() -> None:
    profile = InteractionProfile(
        interaction_style="collab",
        reasoning_depth="normal",
        selection_source="user",
    )
    recent_interactions = [
        {
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "We can keep pulling on the softer edge if you want.",
        },
        {
            "mode": "conversation",
            "kind": "conversation.reply",
            "summary": "We could stay with the same thread a little longer if you want.",
        },
        {"mode": "conversation", "kind": "conversation.reply", "summary": "That makes sense."},
        {"mode": "conversation", "kind": "conversation.reply", "summary": "I am with you on that."},
    ]
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_beat": {
                "follow_up_offer_allowed": False,
                "response_repetition_risk": "high",
            },
            "conversation_turn": {
                "kind": "collaborate",
                "lead": "Yeah, that lands.",
                "follow_ups": ["the emotional edge of it"],
            },
        },
        interaction_profile=profile,
        recent_interactions=recent_interactions,
        active_thread={"normalized_topic": "human chat"},
    )

    assert state is not None

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=recent_interactions,
    )

    assert realized == "Yeah, that lands."
    assert "if you want" not in realized.lower()
    assert "keep pulling" not in realized.lower()


def test_conversational_reply_realizer_direct_mode_keeps_context_but_stays_short() -> None:
    profile = InteractionProfile(
        interaction_style="direct",
        reasoning_depth="normal",
        selection_source="user",
    )
    state = ConversationalReplyRealizer.build_state(
        response={
            "mode": "conversation",
            "kind": "conversation.reply",
            "conversation_beat": {
                "follow_up_offer_allowed": True,
                "continuity_state": "continuing",
            },
            "conversation_turn": {
                "kind": "collaborate",
                "lead": "Yes, that is the thread.",
                "follow_ups": ["the part that still feels unresolved"],
            },
        },
        interaction_profile=profile,
        recent_interactions=[
            {"mode": "conversation", "kind": "conversation.reply", "summary": "We were talking about trust."}
        ],
        active_thread={"normalized_topic": "trust"},
    )

    assert state is not None

    realized = ConversationalReplyRealizer.realize(
        state=state,
        interaction_profile=profile,
        recent_interactions=[],
    )

    assert realized.startswith("Yes, that is the thread.")
    assert len(realized.split()) <= 16
