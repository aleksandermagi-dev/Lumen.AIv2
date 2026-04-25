from pathlib import Path

from lumen.content_generation.models import GeneratedContentDraft
from lumen.content_generation.service import ContentGenerationService


class StubTransport:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)

    def __call__(self, *, instructions: str, input_text: str, metadata: dict[str, object], model: str | None) -> str:
        assert instructions
        assert input_text
        assert metadata
        return self.payloads.pop(0)


def test_content_generation_service_generates_ideas_from_stub_transport(tmp_path: Path) -> None:
    service = ContentGenerationService(
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "ideas": [
                    {"title": "Black holes hide timing clues", "rationale": "Turns an astronomy concept into a hookable tension point"},
                    {"title": "Why small bugs drain big teams", "rationale": "Strong contrast and broad usefulness"}
                  ]
                }
                """
            ]
        ),
    )

    ideas = service.generate_ideas(
        topic="astronomy hooks",
        count=2,
        platform="tiktok",
        style_profile="observational",
    )

    assert len(ideas) == 2
    assert ideas[0].rank == 1
    assert ideas[0].title == "Black holes hide timing clues"


def test_content_generation_service_generates_batch_and_preserves_drafts(tmp_path: Path) -> None:
    service = ContentGenerationService(
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "posts": [
                    {
                      "topic": "Why black holes distort timing",
                      "hook": "Black holes bend more than light.",
                      "script_lines": ["They also scramble how time feels.", "But the weird part is how that changes what we can measure."],
                      "caption": "Time behaves differently near gravity wells.",
                      "hashtags": ["#space", "#physics"],
                      "platform_notes": "Use calm pacing and highlight the shift line.",
                      "retention_check": "The second line re-opens the question.",
                      "suggested_post_time": "afternoon"
                    }
                  ]
                }
                """
            ]
        ),
    )

    batch = service.generate_batch(
        topic="black holes",
        count=1,
        style_profile="observational",
    )

    assert len(batch.items) == 1
    assert not batch.discarded_items
    assert batch.items[0].safety.decision == "PASS"


def test_content_generation_service_formats_platform_variant(tmp_path: Path) -> None:
    service = ContentGenerationService(
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "topic": "Why black holes distort timing",
                  "hook": "Black holes bend more than light.",
                  "script_lines": ["They also bend your sense of timing.", "But that is what makes them measurable."],
                  "caption": "Gravity changes clocks too.",
                  "hashtags": ["#space", "#science"],
                  "platform_notes": "Keep the cut quick and stress the second line.",
                  "retention_check": "The pivot happens on line two.",
                  "suggested_post_time": "evening"
                }
                """
            ]
        ),
    )
    draft = GeneratedContentDraft.from_mapping(
        {
            "id": "draft01",
            "topic": "Why black holes distort timing",
            "hook": "Black holes bend more than light.",
            "script_lines": ["They also scramble how time feels.", "But the weird part is how that changes what we can measure."],
            "caption": "Time behaves differently near gravity wells.",
            "hashtags": ["#space", "#physics"],
            "platform_notes": "Use calm pacing and highlight the shift line.",
            "retention_check": "The second line re-opens the question.",
            "suggested_post_time": "afternoon",
            "style_profile": "observational",
            "platform": "master",
        }
    )

    variant = service.format_variant(
        draft=draft,
        platform="tiktok",
        style_profile="observational",
    )

    assert variant.platform == "tiktok"
    assert variant.source_draft_id == draft.id
    assert variant.safety.decision == "PASS"
