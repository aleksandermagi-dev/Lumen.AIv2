from __future__ import annotations

from textwrap import dedent


STYLE_GUIDANCE = {
    "observational": (
        "Sound like a calm, sharp observer noticing patterns others miss. "
        "Favor contrast, curiosity, and incomplete resolution over heavy conclusions."
    ),
    "analytical": (
        "Sound clear, rigorous, and controlled. Favor structure, insight, and strong framing without sounding academic."
    ),
    "educational": (
        "Teach clearly and simply while keeping retention high. Avoid lecture tone and keep openings vivid."
    ),
    "story": (
        "Use scene-like momentum, perspective shifts, and strong movement without becoming theatrical or melodramatic."
    ),
}

PLATFORM_GUIDANCE = {
    "tiktok": "Favor quicker tension shifts, faster hooks, and sharper contrast.",
    "youtube_shorts": "Favor smoother build-up, clearer phrasing, and a slightly more reflective finish.",
    "all": "Keep the concept portable across TikTok and YouTube Shorts.",
}


def build_system_prompt(*, style_profile: str) -> str:
    style_text = STYLE_GUIDANCE.get(style_profile, STYLE_GUIDANCE["observational"])
    return dedent(
        f"""
        You are generating structured publishable content packages for Lumen.

        Lumen remains the reasoning authority. Your job is execution inside a bounded content-generation lane.

        Style profile:
        - {style_text}

        Hard requirements:
        - Return strict JSON only.
        - Use short, readable lines.
        - Avoid preachy, manipulative, ideological, alarmist, academic, or robotic language.
        - Prefer hooks, tension, contrast, reframing, and curiosity over conclusions.
        - Do not claim uncertain things as facts.
        - Keep platform notes practical and production-focused.
        - Use only morning, afternoon, or evening for suggested_post_time.
        """
    ).strip()


def build_ideas_prompt(*, topic: str | None, count: int, platform: str, style_profile: str) -> str:
    topic_text = topic.strip() if topic else "No topic provided. Generate the strongest broadly useful ideas."
    return dedent(
        f"""
        Generate {count} ranked short-form content ideas.
        Topic context: {topic_text}
        Platform guidance: {PLATFORM_GUIDANCE[platform]}
        Style guidance: {STYLE_GUIDANCE[style_profile]}

        Return strict JSON with this shape:
        {{
          "ideas": [
            {{
              "rank": 1,
              "title": "one-line title",
              "rationale": "one-line reason"
            }}
          ]
        }}

        Keep each idea sharp, distinct, and usable for later script generation.
        """
    ).strip()


def build_batch_prompt(*, topic: str, count: int, style_profile: str) -> str:
    return dedent(
        f"""
        Create {count} master short-form content drafts for this topic: {topic}
        Style guidance: {STYLE_GUIDANCE[style_profile]}

        Return strict JSON with this shape:
        {{
          "posts": [
            {{
              "topic": "one-line topic title",
              "hook": "short opening line",
              "script_lines": ["short line", "short line"],
              "caption": "short caption",
              "hashtags": ["#tag1", "#tag2"],
              "platform_notes": "1 to 2 practical visual or pacing notes",
              "retention_check": "one short sentence",
              "suggested_post_time": "morning"
            }}
          ]
        }}

        Requirements:
        - 5 to 20 seconds worth of script.
        - Flow: hook, build, shift, end.
        - Each line should either build curiosity, raise tension, or shift perspective.
        - End slightly open rather than fully resolved.
        """
    ).strip()


def build_format_prompt(*, item_payload: dict[str, object], platform: str, style_profile: str) -> str:
    return dedent(
        f"""
        Adapt this master draft for {platform}.
        Platform guidance: {PLATFORM_GUIDANCE[platform]}
        Style guidance: {STYLE_GUIDANCE[style_profile]}

        Return strict JSON with this shape:
        {{
          "topic": "one-line topic title",
          "hook": "platform-optimized hook",
          "script_lines": ["short line", "short line"],
          "caption": "platform-optimized caption",
          "hashtags": ["#tag1", "#tag2"],
          "platform_notes": "1 to 2 practical visual or pacing notes",
          "retention_check": "one short sentence",
          "suggested_post_time": "afternoon"
        }}

        Preserve the idea, but make the pacing and language fit the target platform.
        Input:
        {item_payload}
        """
    ).strip()


def build_rewrite_prompt(*, post_payload: dict[str, object], safety_reasons: list[str], style_profile: str) -> str:
    reasons_text = "\n".join(f"- {reason}" for reason in safety_reasons) or "- Improve clarity and safety alignment."
    return dedent(
        f"""
        Rewrite this content draft to fix safety and quality issues.
        Style guidance: {STYLE_GUIDANCE[style_profile]}
        Issues to fix:
        {reasons_text}

        Return strict JSON using the same schema as the input.
        Input:
        {post_payload}
        """
    ).strip()
