from __future__ import annotations

import json

from lumen.content_generation.models import GeneratedContentDraft, GeneratedContentVariant, GeneratedIdea


def format_ideas_text(ideas: list[GeneratedIdea]) -> str:
    return "\n\n".join(f"{item.rank}. {item.title}\n{item.rationale}" for item in ideas)


def format_draft_text(draft: GeneratedContentDraft | GeneratedContentVariant) -> str:
    tags = " ".join(draft.hashtags) if draft.hashtags else "None"
    platform_notes = draft.platform_notes or "None"
    script = "\n".join(draft.script_lines)
    return (
        f"TOPIC:\n{draft.topic}\n\n"
        f"HOOK:\n{draft.hook}\n\n"
        f"SCRIPT:\n{script}\n\n"
        f"CAPTION:\n{draft.caption}\n\n"
        f"HASHTAGS:\n{tags}\n\n"
        f"SUGGESTED POST TIME:\n{draft.suggested_post_time}\n\n"
        f"PLATFORM NOTES:\n{platform_notes}\n\n"
        f"RETENTION CHECK:\n{draft.retention_check}"
    )


def format_drafts_markdown(items: list[GeneratedContentDraft | GeneratedContentVariant]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        tags = " ".join(item.hashtags) if item.hashtags else "None"
        script_block = "  \n".join(item.script_lines)
        blocks.append(
            "\n".join(
                [
                    f"## Item {index}: {item.topic}",
                    "",
                    f"**Platform**  \n{item.platform}",
                    "",
                    f"**Hook**  \n{item.hook}",
                    "",
                    f"**Script**  \n{script_block}",
                    "",
                    f"**Caption**  \n{item.caption}",
                    "",
                    f"**Hashtags**  \n{tags}",
                    "",
                    f"**Retention Check**  \n{item.retention_check}",
                    "",
                    f"**Platform Notes**  \n{item.platform_notes or 'None'}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_json(payload: object) -> str:
    return json.dumps(payload, indent=2)
