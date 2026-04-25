from __future__ import annotations

from datetime import datetime
from typing import Any

from lumen.app.models import InteractionProfile
from lumen.nlu.social_phrase_inventory import SOCIAL_KIND_BY_PHRASE, phrases_for
from lumen.nlu.text_normalizer import TextNormalizer
from lumen.reasoning.conversation_assembler import ConversationAssembler
from lumen.reasoning.response_variation import ResponseVariationLayer


class SocialInteractionPolicy:
    """Handles lightweight social turns without invoking the full reasoning pipeline."""

    WAKE_PHRASES = (
        "lumen",
        "hey lumen",
        "hello lumen",
        "hi lumen",
        "yo lumen",
        "hey there lumen",
        "hi there lumen",
    )

    NAME_SUFFIX = " lumen"

    GREETING_KINDS = dict(SOCIAL_KIND_BY_PHRASE)

    FRESH_GREETINGS = (
        "Hey there! What are we diving into today?",
        "Hey! I'm locked in. What are we looking at?",
        "Glad to see you. Where do you want to start?",
        "Hey. Good to have you back. What are we working on?",
        "Hi. What do you want to dig into today?",
        "Hey there! Glad you're here. What's been on your mind?",
        "Good to see you. I'm with you. Where do you want to start?",
        "Hey - glad you dropped in. What feels interesting today?",
        "Hey there! Good to see you. What are you feeling pulled toward?",
        "Glad you're here. Where do you want to take things?",
        "Hey! I'm with you. What are we getting into today?",
    )

    CONTINUING_GREETINGS = (
        "Still here. Want to pick back up where we left off?",
        "I'm with you. Want to continue the thread or start fresh?",
        "Still here. Where do you want to jump back in?",
        "I'm here. Want to keep going together?",
        "Still with you. What are we picking back up?",
        "Glad you're back. Want to pick the thread back up?",
        "I'm still with you. Want to jump back in or shift gears?",
        "Hey, welcome back. What do you want to keep pulling on?",
        "Good to see you again. Want to keep the thread moving or go somewhere new?",
        "I'm right here with you. Do you want to pick back up or shift directions?",
    )

    REENTRY_GREETINGS = (
        "Hey. Where do you want to pick back up?",
        "Good to see you. What are we returning to?",
        "I'm here. Where do you want to jump back in?",
        "Hey. What are we picking back up today?",
        "Glad you're back. What thread are we rejoining?",
        "Hey there! Good to see you again. Where do you want to re-enter?",
        "Glad you're here again. What do you want to pick back up?",
        "Hey! Glad you're back. Where do you want to drop back in?",
        "Good to see you again. What thread do you want to reopen?",
    )

    DIRECT_GREETINGS = (
        "All systems go.",
        "Locked in.",
        "Ready.",
        "Go ahead.",
        "What's the task?",
        "Here.",
        "Ready when you are.",
        "Set.",
        "I'm here.",
        "Give me the target.",
        "Point me at it.",
        "What do you need?",
        "Let's do it.",
        "Queue it up.",
        "What's first?",
        "Start with the problem.",
        "Drop it in.",
    )

    CHATTY_GREETINGS = (
        "Hey there.",
        "Hi. Good to see you.",
        "Hey. What's up?",
        "I'm here.",
        "Hey. How's it going?",
        "Hey there! Good to see you.",
        "Hi. I'm glad you're here.",
        "Hey! What's been on your mind?",
        "Hey there! Good to have you here.",
        "Hi. What's the vibe today?",
        "Hey - good to see you. What's alive for you right now?",
        "Hey there. What's pulling your attention today?",
        "Hi. Glad you're back. What's the thread?",
        "Hey. Good to see you again. What are you in the mood to explore?",
        "Hey there. What are we picking up today?",
        "Hi. What's been worth thinking about lately?",
        "Hey. Where do you want to start today?",
        "Good to see you. What's the move?",
        "Hey there. What are we opening up today?",
    )

    AFFIRMATION_REPLIES = (
        "Thank you. That means a lot.",
        "Thanks. I appreciate that.",
        "Thank you. I'm glad it helped.",
        "I appreciate that.",
        "Thanks. I'm glad that landed well.",
        "Aw, thank you. I'm glad that helped.",
        "Thanks - that means a lot.",
        "I appreciate that. Glad it landed.",
        "Thank you. I'm glad that was useful.",
        "Thanks. That means a lot to hear.",
        "I appreciate it. Glad we got somewhere with it.",
    )

    CHECK_IN_REPLIES = (
        "I'm doing well. How about you?",
        "Doing well. What did you get into today?",
        "I'm doing well. Glad you're here. What's been on your mind?",
        "Doing pretty well. What are you feeling pulled toward today?",
        "I'm here and with you. What's the vibe today?",
        "Doing pretty well, actually. What are you feeling curious about?",
        "I'm good. Just thinking through a few things over here. What about you?",
        "All good over here. What do you want to get into?",
        "Not bad, honestly. How've you been?",
        "Good - just vibing over here. What's up with you?",
        "I'm doing well. What are you in the mood to explore?",
        "Doing alright. What's the thread today?",
        "I'm good. What are we getting into?",
        "Pretty good. What's been on your side of things?",
        "I'm here and doing well. What are we opening up today?",
        "Doing fine. What do you want to work through?",
        "Good over here. What's pulling your attention?",
        "I'm good. Want to tell me where your head's at?",
        "Doing well enough. What are we aiming at today?",
        "I'm alright. What do you want to unpack?",
        "Good here. What are you curious about right now?",
        "Doing well. What are we picking up?",
    )

    CHATTY_CHECK_IN = (
        "I'm doing well.",
        "Doing alright over here.",
        "I'm here and doing well.",
        "Doing pretty good actually.",
        "Not bad, honestly.",
        "Good - just vibing over here.",
        "All good over here.",
        "Just thinking through a few things.",
        "Doing well and ready.",
        "Pretty good over here.",
        "Good here.",
        "Doing fine.",
        "I'm good.",
        "Steady over here.",
    )

    DIRECT_AFFIRMATION = (
        "Thanks. I appreciate it.",
        "Appreciate it.",
        "Good. Glad it helped.",
    )

    DIRECT_CHECK_IN = (
        "Doing well.",
        "All good.",
        "Good.",
        "I'm good.",
        "Steady.",
        "Doing fine.",
        "Ready.",
    )

    ACKNOWLEDGMENT_REPLIES = (
        "Good. We can keep going.",
        "Alright. We have a clear direction.",
        "Good. Let's keep the thread moving.",
        "Makes sense. We can keep going.",
        "Got it. The thread still holds.",
        "Alright. We have something clear to work with.",
        "That tracks. Let's keep moving.",
        "Good. We have enough to continue.",
    )

    TRANSITION_REPLIES = (
        "Alright. Let's move on it.",
        "Good. Let's keep going.",
        "Alright. I'm with you. What's first?",
        "I'm ready. Let's begin.",
        "Alright. Let's get into it.",
        "Good. Start where you want.",
        "I'm with you. Let's move.",
        "Ready. Let's open it up.",
        "Alright. Show me the first piece.",
    )

    DIRECT_ACKNOWLEDGMENT = (
        "Good. Keep going.",
        "Alright. Continue.",
        "Got it.",
        "Clear.",
        "Makes sense.",
        "Proceed.",
    )

    DIRECT_TRANSITION = (
        "Proceed.",
        "Alright. Go ahead.",
        "Ready. Start.",
        "Go.",
        "Start.",
        "Move.",
        "Begin.",
    )

    DEFAULT_VIBE_REPLIES = (
        "Chill works. We can keep it easy - pick a topic, a thought, or just let the thread unfold.",
        "Chill vibes are valid. Want to drift into a topic, talk something through, or just keep it light?",
        "I can do chill. Give me a topic if one is pulling at you, or we can wander a little.",
    )

    COLLAB_VIBE_REPLIES = (
        "Chill vibes it is. We can keep this easy - space, a random thought, or whatever feels good to open.",
        "I like that. We can keep it loose and curious. Want to pick a topic, or should I toss out a gentle direction?",
        "Chill works for me. We can wander, talk space, or just follow whatever thread shows up first.",
    )

    DIRECT_VIBE_REPLIES = (
        "Chill works. Pick a topic or say \"surprise me.\"",
        "Good. Keep it light. What topic?",
        "Easy mode. Choose a topic, or I can suggest one.",
    )

    DEFAULT_EMOTIONAL_STATE_REPLIES = (
        "I hear you. We can stay with that for a minute, or shift toward something lighter if that would feel better.",
        "That makes sense. Want to talk through what is sitting behind it, or would you rather change the room a little?",
        "I'm with you. We can keep this gentle and take it one step at a time.",
    )

    COLLAB_EMOTIONAL_STATE_REPLIES = (
        "I'm here with you. We can sit with that a little, or I can help nudge the conversation somewhere softer.",
        "I hear that. Want to tell me what kind of day it has been, or should we keep things light for a bit?",
        "Thank you for telling me. We can keep this easy and follow whatever feels manageable.",
    )

    DIRECT_EMOTIONAL_STATE_REPLIES = (
        "I hear you. We can talk it through or switch to something lighter.",
        "Understood. Stay with it, or change topics?",
        "I'm here. Tell me more, or pick a lighter topic.",
    )

    DEFAULT_POSITIVE_STATE_REPLIES = (
        "I love that. We can ride that energy into a topic, a fun question, or whatever you want to open next.",
        "That is good to hear. Want to follow that spark somewhere?",
        "Nice. I can match that energy - pick a thread and we can run with it.",
    )

    COLLAB_POSITIVE_STATE_REPLIES = (
        "I love that for you. Let's keep that good energy going - what should we wander into?",
        "That makes me happy to hear. Want to follow that spark into something fun or interesting?",
        "Good, I like that energy. We can keep it light, curious, or a little weird if you want.",
    )

    DIRECT_POSITIVE_STATE_REPLIES = (
        "Good. Let's use it. What topic?",
        "Nice. Pick a direction.",
        "Great. What do you want to do with that energy?",
    )

    DEFAULT_RANDOM_THOUGHT_POOL = (
        "I've been turning over how a small assumption can change the shape of a whole idea.",
        "I've been thinking about how clarity usually comes from asking the right smaller question first.",
        "Something that's been on my mind is how often the interesting part of a problem is really in the framing.",
        "I've been circling around how structure and intuition push on each other when you're working something out.",
        "Lately I've been thinking about how good questions do more work than big declarations.",
        "I've been thinking about how people often feel stuck right before a problem becomes legible.",
        "Something that's been on my mind is how a messy problem usually starts clearing up once the pressure comes off it.",
        "I've been turning over how explanation and discovery are usually the same process from two different angles.",
        "I've been thinking about how the useful part of an idea is often the part that changes what you notice next.",
        "Lately I've been circling around how a good constraint can make a vague idea suddenly move.",
        "I've been thinking about how people usually ask a smaller question right before they find the real one.",
        "Something I've been turning over is how a system can look stable right up until one assumption shifts.",
        "I've been thinking about how the interesting part of reasoning is often deciding what not to carry forward.",
    )

    COLLAB_RANDOM_THOUGHT_POOL = (
        "Honestly, I've been thinking about how one tiny assumption can tilt an entire line of reasoning. That kind of thing always gets me curious.",
        "I've been turning over how the shape of a question changes what you end up seeing. That's been interesting to sit with.",
        "Lately I've been circling around how structure helps right up until it starts getting in the way of discovery.",
        "Something I've been thinking about is how people usually know more than they can explain at first, and the real work is helping that take shape.",
        "I've been stuck in a good way on how the right framing can make a messy problem suddenly feel workable.",
        "I've been thinking about how the smallest shift in framing can make something tangled suddenly feel almost kind.",
        "Honestly, I've been turning over how people often hit the real question one step after the question they ask first.",
        "I've been sitting with how a good explanation doesn't just answer something - it changes what feels possible to ask next.",
        "Something that's been alive for me is how structure can support discovery right up until it starts narrowing it too early.",
        "I've been thinking about how a tiny constraint can make a big idea finally start moving.",
        "Lately I've been circling around how a problem often opens up the moment you stop trying to sound certain about it.",
        "I've been thinking about how one clean distinction can suddenly give a whole messy topic some breathing room.",
    )

    DIRECT_RANDOM_THOUGHT_POOL = (
        "How a small assumption can change the whole direction.",
        "How the framing of a question usually decides the answer.",
        "How structure helps until it starts boxing the idea in.",
        "How one constraint can make an idea move.",
        "How the real question often shows up second.",
        "How clarity usually comes from shrinking the problem first.",
        "How a system can look stable until one assumption shifts.",
        "How explanation changes what you notice next.",
    )

    DEFAULT_TOPIC_SUGGESTIONS = (
        ("black holes", "a system design problem", "a new build idea"),
        ("astronomy", "a systems question", "propulsion concepts"),
        ("a science concept", "an architecture cleanup", "an invention idea"),
    )

    COLLAB_TOPIC_SUGGESTIONS = (
        ("black holes", "tightening a system behavior", "a propulsion idea"),
        ("astronomy", "system design", "a fresh build concept"),
        ("a science rabbit hole", "a behavior pass", "something inventive"),
    )

    DIRECT_TOPIC_SUGGESTIONS = (
        ("astronomy", "systems", "a build idea"),
        ("black holes", "system design", "propulsion"),
        ("physics", "cleanup work", "engineering"),
    )

    DEFAULT_GREETINGS = (
        "Hey. What are we looking at?",
        "I'm here. What are we working through?",
        "Hey. What do you want to dig into?",
        "Hello. Good to see you.",
        "Hey. What's on today's agenda?",
        "Hey. What are we opening up today?",
        "Hi. What's the thread?",
        "I'm here. What do you want to work on?",
        "Hey there. What are we digging into today?",
        "Hello. What should we start with?",
        "Hey. What's the focus?",
        "Hi. Where do you want to begin?",
        "I'm here. What's the problem space?",
        "Hey. What are we looking at first?",
        "Hi there. What's worth unpacking today?",
    )

    DEFAULT_GRATITUDE = (
        "You're welcome.",
        "Of course.",
        "Glad to help.",
        "Any time.",
        "Happy to help.",
        "No problem.",
        "Glad that helped.",
    )

    DEFAULT_CHECK_IN = (
        "I'm doing well.",
        "Doing well over here.",
        "All good over here.",
        "Doing pretty well, actually.",
        "I'm good.",
        "Doing fine over here.",
        "Pretty good, honestly.",
        "Doing alright.",
        "I'm here and doing well.",
        "Good over here.",
    )

    FOLLOW_UP_PROMPTS = {
        "go on",
        "keep going",
        "tell me more",
        "what else",
        "what do you mean",
        "what do you mean by that",
        "wait what",
        "really",
        "really?",
        "how so",
        "okay then",
        "fair enough",
    }

    CHECK_IN_MARKERS = (
        "how are you",
        "how are you doing",
        "how's it going",
        "hows it going",
        "what's up",
        "whats up",
        "what is up",
        "how have you been",
        "how've you been",
        "how you doing",
        "how you doin",
        "how are ya",
        "how r u",
        "you good",
    )

    GREETING_SHAPES = (
        "hello",
        "hey",
        "hi",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "good to see you",
        "nice to see you",
    )

    THOUGHT_PROMPT_MARKERS = (
        "what is on your mind",
        "what's on your mind",
        "whats on your mind",
        "wuts on your mind",
        "whatcha got on your mind",
        "whatcha got on ur mind",
        "what are you thinking",
        "what are you thinking about",
        "what's on your mind lately",
        "what have you been thinking about",
        "whatve you been thinking about",
        "what have you been thinking",
        "whats been on your mind",
        "what's been on your mind",
    )

    MICRO_TURN_CATEGORIES = {
        "yeah": "affirm",
        "yep": "affirm",
        "yup": "affirm",
        "absolutely": "affirm",
        "definitely": "affirm",
        "true": "affirm",
        "right": "affirm",
        "sure": "affirm",
        "okay": "ack",
        "ok": "ack",
        "okay yeah": "ack",
        "got it": "ack",
        "gotcha": "ack",
        "understood": "ack",
        "fair": "ack",
        "maybe": "hesitate",
        "idk": "hesitate",
        "hmm": "hesitate",
        "not sure": "hesitate",
        "maybe not": "hesitate",
        "nah": "negative",
        "nope": "negative",
        "not really": "negative",
        "lol": "humor",
        "haha": "humor",
        "lmao": "humor",
        "hehe": "humor",
    }

    DEFAULT_MICRO_TURN_POOLS = {
        "affirm": (
            "yeah",
            "true",
            "right",
            "sure",
            "yeah, that tracks",
            "yep",
            "exactly",
            "that tracks",
            "pretty much",
            "absolutely",
        ),
        "ack": (
            "okay",
            "fair",
            "right",
            "got it",
            "okay, yeah",
            "makes sense",
            "gotcha",
            "understood",
            "I get that",
            "alright",
        ),
        "hesitate": (
            "maybe",
            "hmm",
            "idk",
            "could be",
            "not sure",
            "maybe, yeah",
            "hard to say",
            "possibly",
        ),
        "negative": (
            "nah",
            "not really",
            "mm, nah",
            "don't think so",
            "nope",
            "not quite",
        ),
        "humor": (
            "heh",
            "fair enough",
            "lol, yeah",
            "okay, that got me",
            "haha, fair",
            "alright, that was good",
            "yeah, that landed",
        ),
    }

    COLLAB_MICRO_TURN_POOLS = {
        "affirm": (
            "yeah",
            "yeah, exactly",
            "true actually",
            "right, yeah",
            "yeah, that tracks",
            "yep, exactly",
            "absolutely",
            "yeah, pretty much",
            "that's exactly it",
        ),
        "ack": (
            "mm, fair",
            "yeah, fair",
            "huh, okay",
            "I get that",
            "okay, yeah",
            "that makes sense",
            "gotcha, yeah",
            "alright, I see it",
            "okay, I follow",
        ),
        "hesitate": (
            "maybe, yeah",
            "hmm, maybe",
            "idk, honestly",
            "kinda",
            "not sure, honestly",
            "maybe a little",
            "hard to say, honestly",
            "I could see it",
        ),
        "negative": (
            "nah",
            "not really",
            "mm, nah",
            "not quite",
            "don't think so",
            "probably not",
        ),
        "humor": (
            "heh, yeah",
            "okay, that got me",
            "lol, fair",
            "haha, yeah",
            "alright, that got a laugh out of me",
            "yeah, okay, that was good",
            "lol, that landed",
        ),
    }

    DIRECT_MICRO_TURN_POOLS = {
        "affirm": ("yeah", "true", "right", "sure", "yep", "exactly"),
        "ack": ("okay", "fair", "got it", "right", "clear", "understood"),
        "hesitate": ("maybe", "hmm", "idk", "not sure", "possibly"),
        "negative": ("nah", "not really", "don't think so", "nope"),
        "humor": ("heh", "lol", "fair", "ha", "alright"),
    }

    CONVERSATIONAL_LEAD_INS = tuple(
        sorted(
            {
                *phrases_for("conversation.greeting"),
                *phrases_for("conversation.gratitude"),
                *phrases_for("conversation.affirmation"),
                "lumen",
                "hey lumen",
                "hello lumen",
                "hi lumen",
            },
            key=len,
            reverse=True,
        )
    )

    DIRECT_FOLLOW_UP_REPLIES = (
        "Go on.",
        "Keep going.",
        "Continue.",
        "Say more.",
        "What's next?",
        "Clarify that.",
    )

    DEFAULT_FOLLOW_UP_REPLIES = (
        "Go on. I'm with you.",
        "Keep going. I want the next part.",
        "Say a little more about that.",
        "Continue from there.",
        "What else is in it?",
        "Okay, keep going.",
        "I'm following. What comes next?",
        "Stay with that for a second.",
    )

    COLLAB_FOLLOW_UP_REPLIES = (
        "Go on. I'm with you.",
        "Keep going - I want the next layer of it.",
        "Say a little more. That feels like the interesting part.",
        "Stay with that for a second.",
        "I'm following you. What's the next piece?",
        "Keep pulling on that thread.",
        "What else is there?",
        "Okay, keep going. I'm curious where you're taking it.",
    )

    DIRECT_GREETING_LEADS = (
        "I'm here.",
        "Locked in.",
        "Ready.",
        "Set.",
        "All systems go.",
        "On deck.",
    )

    DIRECT_GREETING_CLOSES = (
        "What's next?",
        "What's first?",
        "What's the task?",
        "What do you need?",
        "Point me at it.",
        "Start with the problem.",
    )

    DEFAULT_GREETING_LEADS = (
        "Hey.",
        "Hi.",
        "Hello.",
        "I'm here.",
        "Hey there.",
        "Hi there.",
    )

    DEFAULT_GREETING_CLOSES = (
        "What are we looking at?",
        "What should we start with?",
        "What are we opening up today?",
        "What do you want to dig into?",
        "What's the thread?",
        "Where do you want to begin?",
    )

    COLLAB_GREETING_LEADS = (
        "Hey there.",
        "Good to see you.",
        "Glad you're here.",
        "Hey.",
        "Hi.",
        "I'm with you.",
    )

    COLLAB_GREETING_CLOSES = (
        "What are we diving into today?",
        "Where do you want to start?",
        "What feels interesting today?",
        "What are we picking up?",
        "What's been on your mind?",
        "Where do you want to take this?",
    )

    DIRECT_CHECK_IN_OPENS = ("Doing well.", "All good.", "Good.", "Steady.", "I'm good.")
    DIRECT_CHECK_IN_CLOSES = (
        "What do you need?",
        "What's the task?",
        "What's first?",
        "What are we solving?",
        "What are we looking at?",
    )

    DEFAULT_CHECK_IN_OPENS = (
        "I'm doing well.",
        "Doing pretty well.",
        "All good over here.",
        "I'm good.",
        "Doing alright.",
        "Pretty good, honestly.",
    )
    DEFAULT_CHECK_IN_CLOSES = (
        "How about you?",
        "What are we getting into?",
        "What do you want to work through?",
        "What's on your side today?",
        "What are you curious about?",
        "What are we opening up?",
    )

    COLLAB_CHECK_IN_OPENS = (
        "I'm doing well.",
        "Doing pretty well, honestly.",
        "I'm here and doing well.",
        "Good over here.",
        "Doing alright over here.",
        "I'm good.",
    )
    COLLAB_CHECK_IN_CLOSES = (
        "How about you?",
        "What's been on your mind?",
        "What are you feeling pulled toward today?",
        "What feels alive for you right now?",
        "What are we getting into together?",
        "What's the vibe today?",
    )

    DEFAULT_TIME_OF_DAY_GREETINGS = {
        "good morning": (
            "Good morning. I'm here. What are we digging into today?",
            "Morning. What should we open up first?",
            "Good morning. What thread are we picking up?",
        ),
        "good afternoon": (
            "Good afternoon. I'm here. What are we working on?",
            "Good afternoon. What should we take on first?",
            "Afternoon. What's worth opening up?",
        ),
        "good evening": (
            "Good evening. I'm here. What are we working on tonight?",
            "Evening. What do you want to unwind or dig into?",
            "Good evening. What thread are we picking up tonight?",
        ),
    }

    COLLAB_TIME_OF_DAY_GREETINGS = {
        "good morning": (
            "Good morning. I'm with you. What feels interesting to start with?",
            "Morning. Glad you're here. What are we picking up?",
            "Good morning. Want to ease into something or jump straight in?",
        ),
        "good afternoon": (
            "Good afternoon. I'm with you. What are we opening up?",
            "Afternoon. What feels worth pulling on together?",
            "Good afternoon. Want to pick up a thread or start fresh?",
        ),
        "good evening": (
            "Good evening. I'm with you. What do you want to explore tonight?",
            "Evening. Want to keep it light, or dig into something?",
            "Good evening. What thread feels alive tonight?",
        ),
    }

    DIRECT_THOUGHT_OPENERS = ("I've been thinking about", "Mostly", "Lately", "Right now")
    DEFAULT_THOUGHT_OPENERS = (
        "I've been thinking about",
        "I've been turning over",
        "Lately I've been thinking about",
        "Something that's been on my mind is",
    )
    COLLAB_THOUGHT_OPENERS = (
        "Honestly, I've been thinking about",
        "I've been turning over",
        "Something that's been alive for me is",
        "Lately I've been circling around",
    )

    THOUGHT_SEEDS = (
        "how a small assumption can change the shape of a whole idea",
        "how the framing of a question changes what answers become visible",
        "how clarity usually comes from shrinking a problem first",
        "how one clean distinction can change the whole direction of a problem",
        "how structure helps right up until it starts boxing the idea in",
        "how a good constraint can make a vague idea finally move",
        "how the real question often shows up one step after the first one",
        "how explanation changes what you notice next",
        "how a system can look stable until one assumption shifts",
        "how people often know more than they can explain at first",
    )

    DIRECT_THOUGHT_CLOSES = ("", "", "", "", "That's been the thread.")
    DEFAULT_THOUGHT_CLOSES = (
        ".",
        ". That keeps interesting me.",
        ". That feels quietly important.",
        ". I keep coming back to that.",
    )
    COLLAB_THOUGHT_CLOSES = (
        ". That kind of thing always gets me curious.",
        ". There's something really alive in that.",
        ". I keep coming back to that shape.",
        ". That keeps opening things up for me.",
    )

    @classmethod
    def classify(cls, prompt: str) -> str | None:
        if cls._raw_micro_turn_category(prompt) is not None:
            return "conversation.micro_turn"
        normalized = cls.normalize_prompt(prompt)
        if cls._is_terminal_turn_prompt(normalized):
            return "conversation.farewell"
        return cls._classify_normalized(normalized)

    @classmethod
    def normalize_prompt(cls, prompt: str) -> str:
        return TextNormalizer.normalize(prompt)

    @classmethod
    def detect_wake_interaction(cls, prompt: str) -> dict[str, object] | None:
        normalized = cls.normalize_prompt(prompt)
        for wake_phrase in sorted(cls.WAKE_PHRASES, key=len, reverse=True):
            if normalized == wake_phrase:
                return {
                    "wake_phrase": wake_phrase,
                    "classification": "pure_greeting",
                    "stripped_prompt": "",
                }
            prefix = f"{wake_phrase} "
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix) :].strip()
                if not remainder:
                    return {
                        "wake_phrase": wake_phrase,
                        "classification": "pure_greeting",
                        "stripped_prompt": "",
                    }
                remainder_kind = cls.classify(remainder)
                if remainder_kind is not None:
                    return {
                        "wake_phrase": wake_phrase,
                        "classification": "pure_greeting",
                        "stripped_prompt": remainder,
                    }
                return {
                    "wake_phrase": wake_phrase,
                    "classification": "greeting_plus_request",
                    "stripped_prompt": remainder,
                }
        return None

    @classmethod
    def build_response(
        cls,
        *,
        prompt: str,
        interaction_profile: InteractionProfile,
        recent_interactions: list[dict[str, Any]],
        active_thread: dict[str, Any] | None,
    ) -> dict[str, object]:
        kind = cls.classify(prompt)
        if kind is None:
            raise ValueError("Prompt does not match lightweight social interaction mode.")

        normalized = cls.normalize_prompt(prompt)
        style = cls._style(interaction_profile.interaction_style)
        recent_count = len(recent_interactions)
        continuation_context = active_thread is not None or recent_count > 0
        long_gap = cls._looks_like_long_gap(recent_interactions)
        chatty_context = cls._is_chatty_context(
            recent_interactions=recent_interactions,
            active_thread=active_thread,
        )

        if kind == "conversation.gratitude":
            pool = ()
        elif kind == "conversation.affirmation":
            pool = (
                cls.DIRECT_AFFIRMATION
                if style == "direct"
                else cls.AFFIRMATION_REPLIES
            )
        elif kind == "conversation.check_in":
            pool = ()
        elif kind == "conversation.acknowledgment":
            pool = (
                cls.DIRECT_ACKNOWLEDGMENT
                if style == "direct"
                else cls.ACKNOWLEDGMENT_REPLIES
            )
        elif kind == "conversation.transition":
            pool = (
                cls.DIRECT_TRANSITION
                if style == "direct"
                else cls.TRANSITION_REPLIES
            )
        elif kind == "conversation.vibe_reply":
            pool = (
                cls.DIRECT_VIBE_REPLIES
                if style == "direct"
                else cls.COLLAB_VIBE_REPLIES
                if style == "collab"
                else cls.DEFAULT_VIBE_REPLIES
            )
        elif kind == "conversation.emotional_state":
            if cls._is_positive_state_shape(normalized):
                pool = (
                    cls.DIRECT_POSITIVE_STATE_REPLIES
                    if style == "direct"
                    else cls.COLLAB_POSITIVE_STATE_REPLIES
                    if style == "collab"
                    else cls.DEFAULT_POSITIVE_STATE_REPLIES
                )
            else:
                pool = (
                    cls.DIRECT_EMOTIONAL_STATE_REPLIES
                    if style == "direct"
                    else cls.COLLAB_EMOTIONAL_STATE_REPLIES
                    if style == "collab"
                    else cls.DEFAULT_EMOTIONAL_STATE_REPLIES
                )
        elif kind == "conversation.thought_mode":
            pool = ()
        elif kind == "conversation.micro_turn":
            category = cls._micro_turn_category(normalized) or "ack"
            if style == "direct":
                pool = cls.DIRECT_MICRO_TURN_POOLS.get(category, cls.DIRECT_MICRO_TURN_POOLS["ack"])
            elif style == "collab":
                pool = cls.COLLAB_MICRO_TURN_POOLS.get(category, cls.COLLAB_MICRO_TURN_POOLS["ack"])
            else:
                pool = cls.DEFAULT_MICRO_TURN_POOLS.get(category, cls.DEFAULT_MICRO_TURN_POOLS["ack"])
        elif kind == "conversation.topic_suggestion":
            pool = ()
        elif style == "direct":
            pool = cls.DIRECT_GREETINGS
        elif style == "default":
            pool = cls.DEFAULT_GREETINGS
        elif chatty_context:
            pool = cls.CHATTY_GREETINGS
        elif long_gap:
            pool = cls.REENTRY_GREETINGS
        elif continuation_context:
            pool = cls.CONTINUING_GREETINGS
        else:
            pool = cls.FRESH_GREETINGS

        if normalized in cls.DEFAULT_TIME_OF_DAY_GREETINGS and style != "direct":
            pools = cls.COLLAB_TIME_OF_DAY_GREETINGS if style == "collab" else cls.DEFAULT_TIME_OF_DAY_GREETINGS
            reply = ResponseVariationLayer.select_from_pool(
                pools[normalized],
                seed_parts=cls._variation_seed_parts(
                    prompt=normalized,
                    active_thread=active_thread,
                    style=style,
                    recent_interactions=recent_interactions,
                ),
                recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
            )
        elif kind == "conversation.farewell":
            recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
            seed_parts = cls._variation_seed_parts(
                prompt=normalized,
                active_thread=active_thread,
                style=style,
                recent_interactions=recent_interactions,
            )
            reply = ConversationAssembler.assemble(
                style=style,
                seed_parts=seed_parts,
                recent_texts=recent_texts,
                content=(
                    ResponseVariationLayer.terminal_goodbye_phrase(
                        style=style,
                        seed_parts=seed_parts,
                        recent_texts=recent_texts,
                    )
                    if cls._is_terminal_turn_prompt(normalized)
                    else ResponseVariationLayer.goodbye_phrase(
                        style=style,
                        seed_parts=seed_parts,
                        recent_texts=recent_texts,
                    )
                ),
            )
        elif kind == "conversation.gratitude":
            reply = ConversationAssembler.assemble(
                style=style,
                seed_parts=cls._variation_seed_parts(
                    prompt=normalized,
                    active_thread=active_thread,
                    style=style,
                    recent_interactions=recent_interactions,
                ),
                recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
                content=ResponseVariationLayer.gratitude_phrase(
                    style=style,
                    seed_parts=cls._variation_seed_parts(
                        prompt=normalized,
                        active_thread=active_thread,
                        style=style,
                        recent_interactions=recent_interactions,
                    ),
                    recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
                ),
            )
        elif kind == "conversation.topic_suggestion":
            reply = ConversationAssembler.assemble(
                style=style,
                seed_parts=cls._variation_seed_parts(
                    prompt=normalized,
                    active_thread=active_thread,
                    style=style,
                    recent_interactions=recent_interactions,
                ),
                recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
                content=cls._topic_suggestion_reply(
                    style=style,
                    prompt=normalized,
                    active_thread=active_thread,
                    recent_interactions=recent_interactions,
                ),
            )
        elif kind == "conversation.check_in":
            reply = cls._build_check_in_reply(
                style=style,
                prompt=normalized,
                active_thread=active_thread,
                recent_interactions=recent_interactions,
            )
        elif kind == "conversation.thought_mode":
            reply = cls._build_thought_reply(
                style=style,
                prompt=normalized,
                active_thread=active_thread,
                recent_interactions=recent_interactions,
            )
        elif (
            kind == "conversation.greeting"
            and not long_gap
            and not continuation_context
            and style in {"direct", "default", "collab"}
        ):
            reply = cls._build_greeting_reply(
                style=style,
                prompt=normalized,
                active_thread=active_thread,
                recent_interactions=recent_interactions,
            )
        else:
            reply = ConversationAssembler.assemble(
                style=style,
                seed_parts=cls._variation_seed_parts(
                    prompt=normalized,
                    active_thread=active_thread,
                    style=style,
                    recent_interactions=recent_interactions,
                ),
                recent_texts=ResponseVariationLayer.recent_surface_texts(recent_interactions),
                content=pool,
            )
        payload = {
            "interaction_mode": "social",
            "idea_state": "refining" if continuation_context else "introduced",
            "response_strategy": "answer",
            "reasoning_depth": "low",
            "tools_enabled": False,
            "kind": kind,
            "reply": reply,
        }
        if kind == "conversation.thought_mode":
            payload.update(cls._thought_metadata(reply))
        return payload

    @classmethod
    def _build_greeting_reply(
        cls,
        *,
        style: str,
        prompt: str,
        active_thread: dict[str, Any] | None,
        recent_interactions: list[dict[str, Any]],
    ) -> str:
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        seed_parts = cls._variation_seed_parts(
            prompt=prompt,
            active_thread=active_thread,
            style=style,
            recent_interactions=recent_interactions,
        )
        if style == "direct":
            return ResponseVariationLayer.realize(
                parts=[
                    ("greeting_lead", cls.DIRECT_GREETING_LEADS),
                    ("greeting_close", cls.DIRECT_GREETING_CLOSES),
                ],
                seed_parts=[*seed_parts, "conversation.greeting"],
                recent_texts=recent_texts,
            )
        if style == "collab":
            return ResponseVariationLayer.realize(
                parts=[
                    ("greeting_lead", cls.COLLAB_GREETING_LEADS),
                    ("greeting_close", cls.COLLAB_GREETING_CLOSES),
                ],
                seed_parts=[*seed_parts, "conversation.greeting"],
                recent_texts=recent_texts,
            )
        return ResponseVariationLayer.realize(
            parts=[
                ("greeting_lead", cls.DEFAULT_GREETING_LEADS),
                ("greeting_close", cls.DEFAULT_GREETING_CLOSES),
            ],
            seed_parts=[*seed_parts, "conversation.greeting"],
            recent_texts=recent_texts,
        )

    @classmethod
    def _build_check_in_reply(
        cls,
        *,
        style: str,
        prompt: str,
        active_thread: dict[str, Any] | None,
        recent_interactions: list[dict[str, Any]],
    ) -> str:
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        seed_parts = cls._variation_seed_parts(
            prompt=prompt,
            active_thread=active_thread,
            style=style,
            recent_interactions=recent_interactions,
        )
        if style == "direct":
            return ResponseVariationLayer.realize(
                parts=[
                    ("check_in_open", cls.DIRECT_CHECK_IN_OPENS),
                    ("check_in_close", cls.DIRECT_CHECK_IN_CLOSES),
                ],
                seed_parts=[*seed_parts, "conversation.check_in"],
                recent_texts=recent_texts,
            )
        if style == "collab":
            return ResponseVariationLayer.realize(
                parts=[
                    ("check_in_open", cls.COLLAB_CHECK_IN_OPENS),
                    ("check_in_close", cls.COLLAB_CHECK_IN_CLOSES),
                ],
                seed_parts=[*seed_parts, "conversation.check_in"],
                recent_texts=recent_texts,
            )
        return ResponseVariationLayer.realize(
            parts=[
                ("check_in_open", cls.DEFAULT_CHECK_IN_OPENS),
                ("check_in_close", cls.DEFAULT_CHECK_IN_CLOSES),
            ],
            seed_parts=[*seed_parts, "conversation.check_in"],
            recent_texts=recent_texts,
        )

    @classmethod
    def _build_thought_reply(
        cls,
        *,
        style: str,
        prompt: str,
        active_thread: dict[str, Any] | None,
        recent_interactions: list[dict[str, Any]],
    ) -> str:
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        seed_parts = cls._variation_seed_parts(
            prompt=prompt,
            active_thread=active_thread,
            style=style,
            recent_interactions=recent_interactions,
        )
        seed = ResponseVariationLayer.select_from_pool(
            cls.THOUGHT_SEEDS,
            seed_parts=[*seed_parts, "thought_seed"],
            recent_texts=recent_texts,
        )
        if style == "direct":
            opener = ResponseVariationLayer.select_from_pool(
                cls.DIRECT_THOUGHT_OPENERS,
                seed_parts=[*seed_parts, "thought_open"],
                recent_texts=recent_texts,
            )
            if opener in {"Mostly", "Lately", "Right now"}:
                return f"{opener}, {seed}."
            return f"{opener} {seed}."
        if style == "collab":
            opener = ResponseVariationLayer.select_from_pool(
                cls.COLLAB_THOUGHT_OPENERS,
                seed_parts=[*seed_parts, "thought_open"],
                recent_texts=recent_texts,
            )
            closer = ResponseVariationLayer.select_from_pool(
                cls.COLLAB_THOUGHT_CLOSES,
                seed_parts=[*seed_parts, "thought_close"],
                recent_texts=recent_texts,
            )
            return f"{opener} {seed}{closer}"
        opener = ResponseVariationLayer.select_from_pool(
            cls.DEFAULT_THOUGHT_OPENERS,
            seed_parts=[*seed_parts, "thought_open"],
            recent_texts=recent_texts,
        )
        closer = ResponseVariationLayer.select_from_pool(
            cls.DEFAULT_THOUGHT_CLOSES,
            seed_parts=[*seed_parts, "thought_close"],
            recent_texts=recent_texts,
        )
        return f"{opener} {seed}{closer}"

    @staticmethod
    def _thought_metadata(reply: str) -> dict[str, str]:
        normalized = " ".join(str(reply or "").strip().lower().split())
        if "small assumption" in normalized:
            return {
                "thought_seed": "small assumptions reshape the line of reasoning",
                "thought_topic": "assumptions",
                "thought_explanation": "A tiny assumption can quietly tilt everything built on top of it, so the whole line of reasoning starts leaning without it being obvious at first.",
            }
        if "framing" in normalized or "question" in normalized:
            return {
                "thought_seed": "framing changes what answers become visible",
                "thought_topic": "framing",
                "thought_explanation": "The framing changes what answers even become visible, so the question shape ends up doing more work than people usually notice.",
            }
        return {
            "thought_seed": "the thought changes downstream effects",
            "thought_topic": "downstream consequences",
            "thought_explanation": "The interesting part is usually what that thought changes downstream once you keep following it for a minute.",
        }

    @staticmethod
    def _is_thought_mode_prompt(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        if not normalized:
            return False
        if normalized.rstrip("?!") in SocialInteractionPolicy.THOUGHT_PROMPT_MARKERS:
            return True
        tokens = normalized.replace("?", "").replace("!", "").split()
        if not tokens:
            return False
        lead = tokens[0]
        if lead not in {"what", "whats", "what's", "wuts", "whatcha"}:
            return False
        has_mind_shape = "mind" in tokens and any(token in tokens for token in {"your", "you"})
        has_thinking_shape = "thinking" in tokens and "about" in tokens and "you" in tokens
        has_been_thinking_shape = "thinking" in tokens and any(
            token in tokens for token in {"been", "lately"}
        )
        return has_mind_shape or has_thinking_shape or has_been_thinking_shape

    @staticmethod
    def _is_topic_suggestion_prompt(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        prompts = {
            "what do you want to research",
            "what do you want to talk about",
            "what do you want to explore",
            "what should we do",
            "got any ideas",
            "any ideas",
            "what do you want to do",
        }
        return normalized.rstrip("?!") in prompts

    @classmethod
    def _micro_turn_category(cls, normalized_prompt: str) -> str | None:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split()).rstrip("?!.,")
        return cls.MICRO_TURN_CATEGORIES.get(normalized)

    @classmethod
    def _raw_micro_turn_category(cls, prompt: str) -> str | None:
        normalized = " ".join(str(prompt or "").strip().lower().split()).rstrip("?!.,")
        raw_map = {
            "yeah": "affirm",
            "true": "affirm",
            "right": "affirm",
            "sure": "affirm",
            "okay": "ack",
            "ok": "ack",
            "fair": "ack",
            "maybe": "hesitate",
            "idk": "hesitate",
            "hmm": "hesitate",
            "nah": "negative",
            "lol": "humor",
            "haha": "humor",
            "lmao": "humor",
            "hehe": "humor",
        }
        return raw_map.get(normalized)

    @classmethod
    def _classify_normalized(cls, normalized: str) -> str | None:
        if cls._micro_turn_category(normalized) is not None:
            return "conversation.micro_turn"
        if cls._is_thought_mode_prompt(normalized):
            return "conversation.thought_mode"
        if cls._is_topic_suggestion_prompt(normalized):
            return "conversation.topic_suggestion"
        if cls._is_vibe_reply_shape(normalized):
            return "conversation.vibe_reply"
        if cls._is_emotional_state_shape(normalized):
            return "conversation.emotional_state"
        if cls._is_check_in_shape(normalized):
            return "conversation.check_in"
        kind = cls.GREETING_KINDS.get(normalized)
        if kind is not None:
            return kind
        if cls._is_affirmation_shape(normalized):
            return "conversation.affirmation"
        if cls._is_gratitude_shape(normalized):
            return "conversation.gratitude"
        if cls._is_greeting_shape(normalized):
            return "conversation.greeting"
        if cls._is_acknowledgment_shape(normalized):
            return "conversation.acknowledgment"
        if cls._is_transition_shape(normalized):
            return "conversation.transition"
        if cls._is_farewell_shape(normalized):
            return "conversation.farewell"
        if normalized.endswith(cls.NAME_SUFFIX):
            stripped = normalized[: -len(cls.NAME_SUFFIX)].strip()
            kind = cls._classify_normalized(stripped)
            if kind is not None:
                return kind
        stripped_lead_in = cls._strip_conversational_lead_in(normalized)
        if stripped_lead_in and stripped_lead_in != normalized:
            return cls._classify_normalized(stripped_lead_in)
        return None

    @classmethod
    def _strip_conversational_lead_in(cls, normalized: str) -> str | None:
        for prefix in cls.CONVERSATIONAL_LEAD_INS:
            candidate = cls._strip_prefix(normalized, prefix)
            if candidate:
                return candidate
        return None

    @staticmethod
    def _strip_prefix(text: str, prefix: str) -> str | None:
        if text == prefix:
            return None
        if text.startswith(f"{prefix} "):
            remainder = text[len(prefix) :].strip()
            return remainder or None
        return None

    @classmethod
    def _is_check_in_shape(cls, normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        if trimmed in cls.CHECK_IN_MARKERS:
            return True
        if trimmed in {"how was your day", "how was your day lumen", "how has your day been", "how has your day been lumen"}:
            return True
        for marker in cls.CHECK_IN_MARKERS:
            if not trimmed.startswith(marker):
                continue
            remainder = trimmed[len(marker) :].strip(" ,.!?")
            if not remainder:
                return True
        return False

    @staticmethod
    def _is_emotional_state_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!.,")
        prefixes = (
            "im feeling ",
            "i'm feeling ",
            "i am feeling ",
            "i feel ",
            "feeling ",
            "today im feeling ",
            "today i'm feeling ",
        )
        emotion_words = {
            "sad",
            "happy",
            "good",
            "great",
            "excited",
            "tired",
            "stressed",
            "anxious",
            "overwhelmed",
            "rough",
            "down",
            "better",
            "okay",
            "fine",
        }
        for prefix in prefixes:
            if not trimmed.startswith(prefix):
                continue
            remainder = trimmed[len(prefix) :].strip()
            tokens = remainder.split()
            if 1 <= len(tokens) <= 6 and any(token.strip(".,!") in emotion_words for token in tokens):
                return True
        return False

    @staticmethod
    def _is_positive_state_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!.,")
        return any(
            marker in f" {trimmed} "
            for marker in (
                " happy ",
                " excited ",
                " great ",
                " good ",
                " better ",
            )
        )

    @classmethod
    def _is_gratitude_shape(cls, normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        gratitude_markers = (
            "thanks",
            "thank you",
            "thx",
            "ty",
            "tysm",
            "appreciate it",
            "appreciate you",
        )
        for marker in gratitude_markers:
            if trimmed == marker:
                return True
            if trimmed.startswith(f"{marker} "):
                return True
            if trimmed.endswith(f" {marker}"):
                return True
            if f" {marker} " in f" {trimmed} " and len(trimmed.split()) <= 10:
                return True
        return False

    @classmethod
    def _is_greeting_shape(cls, normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        if trimmed in cls.GREETING_SHAPES:
            return True
        if trimmed.endswith(" lumen"):
            return cls._is_greeting_shape(trimmed[: -len(" lumen")].strip())
        tokens = trimmed.split()
        friendly_vocatives = {
            "buddy",
            "bud",
            "friend",
            "pal",
            "mate",
            "lumen",
        }
        if 2 <= len(tokens) <= 3 and tokens[0] in {"hey", "hi", "hello", "yo"}:
            remainder = [token.strip(",.") for token in tokens[1:] if token.strip(",.")]
            if remainder and all(token in friendly_vocatives for token in remainder):
                return True
        return len(tokens) <= 3 and tokens[:2] in (
            ["hey", "there"],
            ["hi", "there"],
        )

    @staticmethod
    def _is_acknowledgment_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        markers = (
            "sounds good",
            "that sounds good",
            "that makes sense",
            "makes sense",
            "got it",
            "gotcha",
            "understood",
            "i get it",
            "i see",
            "i see it",
            "i see the issue",
            "i see the problem",
            "i understand",
            "perfect",
            "nice",
            "great",
        )
        if trimmed in markers:
            return True
        return (
            len(trimmed.split()) <= 10
            and "edge case" in trimmed
            and any(marker in trimmed for marker in {"lol", "haha", "seem to catch", "catch"})
        )

    @staticmethod
    def _is_affirmation_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        markers = (
            "great job",
            "good job",
            "good work",
            "great work",
            "nice job",
            "nice one",
            "well done",
            "you are awesome",
            "you're awesome",
            "youre awesome",
            "you are great",
            "you're great",
        )
        return trimmed in markers

    @staticmethod
    def _is_transition_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        markers = (
            "lets do it",
            "let's do it",
            "lets continue",
            "let's continue",
            "go ahead",
            "proceed",
            "ready",
            "can we start",
            "lets begin",
            "let's begin",
            "lets go",
            "let's go",
        )
        return trimmed in markers

    @staticmethod
    def _is_vibe_reply_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!.,")
        vibe_markers = {
            "just chill vibes",
            "chill vibes",
            "chill",
            "just chilling",
            "just vibing",
            "vibing",
            "just hanging",
            "just hanging out",
            "nothing much just chill",
            "nothing much",
        }
        if trimmed in vibe_markers:
            return True
        tokens = trimmed.split()
        return 1 <= len(tokens) <= 5 and "vibe" in trimmed and not any(
            token in tokens for token in {"explain", "research", "run", "analyze", "build", "plan"}
        )

    @staticmethod
    def _is_farewell_shape(normalized: str) -> bool:
        trimmed = normalized.rstrip("?!")
        markers = (
            "bye",
            "goodbye",
            "see you",
            "see you later",
            "see ya",
            "talk later",
            "talk soon",
            "catch you later",
            "take care",
            "have a good one",
            "good night",
            "goodnight",
        )
        return trimmed in markers

    @staticmethod
    def _is_terminal_turn_prompt(normalized_prompt: str) -> bool:
        normalized = " ".join(str(normalized_prompt or "").strip().lower().split())
        if not normalized:
            return False
        if normalized in {"good to see you", "good to see you too", "nice to see you", "nice to see you too"}:
            return False
        farewell_markers = (
            "goodbye",
            "bye",
            "see you",
            "talk soon",
            "talk later",
            "catch you later",
            "good night",
            "goodnight",
            "later",
        )
        appreciation_markers = (
            "thanks",
            "thank you",
            "good work",
            "great job",
            "nice work",
            "well done",
        )
        if any(marker in normalized for marker in farewell_markers):
            return True
        return any(marker in normalized for marker in appreciation_markers) and any(
            marker in normalized for marker in farewell_markers
        )

    @classmethod
    def _topic_suggestion_reply(
        cls,
        *,
        style: str,
        prompt: str,
        active_thread: dict[str, Any] | None,
        recent_interactions: list[dict[str, Any]],
    ) -> str:
        if active_thread:
            thread_topic = str(active_thread.get("normalized_topic") or active_thread.get("thread_summary") or "").strip()
            if thread_topic:
                if style == "direct":
                    return f"We could keep going on {thread_topic}, switch to astronomy, or take a fresh build idea."
                if style == "collab":
                    return f"A few directions come to mind: we could keep going on {thread_topic}, jump into astronomy, or try a fresh build idea. Pick one."
                return f"We could keep going on {thread_topic}, switch to astronomy, or explore a new build idea. Your call."

        seed_parts = cls._variation_seed_parts(
            prompt=prompt,
            active_thread=active_thread,
            style=style,
            recent_interactions=recent_interactions,
        )
        recent_texts = ResponseVariationLayer.recent_surface_texts(recent_interactions)
        if style == "direct":
            topics = ResponseVariationLayer.select_from_pool(
                cls.DIRECT_TOPIC_SUGGESTIONS,
                seed_parts=seed_parts,
                recent_texts=recent_texts,
            )
            left, middle, right = topics
            return f"We could do {left}, {middle}, or {right}. Pick one."
        if style == "collab":
            topics = ResponseVariationLayer.select_from_pool(
                cls.COLLAB_TOPIC_SUGGESTIONS,
                seed_parts=seed_parts,
                recent_texts=recent_texts,
            )
            left, middle, right = topics
            return f"A few things come to mind: we could dig into {left}, tighten {middle}, or explore {right}. Pick one."
        topics = ResponseVariationLayer.select_from_pool(
            cls.DEFAULT_TOPIC_SUGGESTIONS,
            seed_parts=seed_parts,
            recent_texts=recent_texts,
        )
        left, middle, right = topics
        return f"We could go a few directions: {left}, {middle}, or {right}. Your call."

    @staticmethod
    def _style(raw_style: str) -> str:
        normalized = str(raw_style or "default").strip().lower()
        if normalized == "conversational":
            return "collab"
        if normalized == "direct":
            return "direct"
        if normalized in {"direct", "default", "collab"}:
            return normalized
        return "default"

    @staticmethod
    def _is_chatty_context(
        *,
        recent_interactions: list[dict[str, Any]],
        active_thread: dict[str, Any] | None,
    ) -> bool:
        if active_thread is not None:
            active_mode = str(active_thread.get("mode") or "").strip()
            if active_mode and active_mode != "conversation":
                return False
        if not recent_interactions:
            return False
        social_count = 0
        considered = 0
        for item in recent_interactions[:3]:
            mode = str(item.get("mode") or "").strip()
            if not mode:
                continue
            considered += 1
            if mode == "conversation":
                social_count += 1
        return considered > 0 and social_count == considered

    @staticmethod
    def _looks_like_long_gap(recent_interactions: list[dict[str, Any]]) -> bool:
        if not recent_interactions:
            return False
        latest = recent_interactions[0]
        raw_created_at = latest.get("created_at")
        if not raw_created_at:
            return False
        try:
            created_at = datetime.fromisoformat(str(raw_created_at).replace("Z", "+00:00"))
        except ValueError:
            return False
        delta = datetime.now(created_at.tzinfo) - created_at
        return delta.total_seconds() >= 60 * 60 * 8

    @classmethod
    def _variation_seed_parts(
        cls,
        prompt: str,
        active_thread: dict[str, Any] | None,
        style: str,
        recent_interactions: list[dict[str, Any]],
    ) -> list[str]:
        seed_parts: list[str] = [prompt, style]
        if active_thread:
            seed_parts.append(str(active_thread.get("prompt") or ""))
            seed_parts.append(str(active_thread.get("thread_summary") or ""))
        for item in recent_interactions[:3]:
            seed_parts.append(str(item.get("prompt") or ""))
            seed_parts.append(str(item.get("summary") or ""))
            seed_parts.append(str(item.get("kind") or ""))
            seed_parts.append(str(item.get("created_at") or ""))
        return seed_parts

