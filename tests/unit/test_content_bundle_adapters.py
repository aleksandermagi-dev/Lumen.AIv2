from pathlib import Path

from lumen.tools.registry_types import BundleManifest, CapabilityManifest, ToolRequest

from tool_bundles.content.adapters.format_platform_adapter import FormatPlatformAdapter
from tool_bundles.content.adapters.generate_batch_adapter import GenerateBatchAdapter
from tool_bundles.content.adapters.generate_ideas_adapter import GenerateIdeasAdapter
from tool_bundles.content.adapters._content_adapter_utils import classify_generation_error


class StubTransport:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)

    def __call__(self, *, instructions: str, input_text: str, metadata: dict[str, object], model: str | None) -> str:
        return self.payloads.pop(0)


def _manifest() -> BundleManifest:
    return BundleManifest(
        id="content",
        name="Content Generation Tools",
        version="0.1.0",
        entrypoint="bundle.py",
        capabilities=[
            CapabilityManifest(id="generate_ideas", adapter="generate_ideas_adapter"),
            CapabilityManifest(id="generate_batch", adapter="generate_batch_adapter"),
            CapabilityManifest(id="format_platform", adapter="format_platform_adapter"),
        ],
    )


def test_generate_ideas_adapter_writes_artifacts(tmp_path: Path) -> None:
    adapter = GenerateIdeasAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "ideas": [
                    {"title": "The hook people skip", "rationale": "Useful for TikTok packaging"},
                    {"title": "Why short scripts drift", "rationale": "Good batch follow-up"}
                  ]
                }
                """
            ]
        ),
    )
    request = ToolRequest(
        tool_id="content",
        capability="generate_ideas",
        params={"topic": "hooks", "count": 2, "platform": "tiktok"},
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "ok"
    assert result.structured_data["count"] == 2
    assert any(artifact.name == "ideas.txt" for artifact in result.artifacts)


def test_generate_ideas_adapter_reports_missing_provider_configuration_cleanly(tmp_path: Path) -> None:
    def _missing_provider_transport(**kwargs) -> str:
        raise RuntimeError(
            "Hosted content generation is not configured. "
            "Set OPENAI_API_KEY and an OpenAI Responses model or configure a hosted inference provider."
        )

    adapter = GenerateIdeasAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=_missing_provider_transport,
    )
    request = ToolRequest(
        tool_id="content",
        capability="generate_ideas",
        params={"topic": "hooks", "count": 3},
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "error"
    assert result.structured_data["failure_category"] == "missing_provider_config"
    assert result.structured_data["result_quality"] == "capability_unavailable"
    assert result.structured_data["failure_reason"] == "Hosted content generation is not configured for this runtime."
    assert result.structured_data["runtime_diagnostics"]["provider_status"] == "missing_provider_config"
    assert result.structured_data["runtime_diagnostics"]["runtime_ready"] is False
    assert "provider" in result.summary.lower() or "configured" in result.summary.lower()


def test_classify_generation_error_returns_runtime_diagnostics() -> None:
    failure = classify_generation_error(
        RuntimeError(
            "Hosted content generation is not configured. "
            "Set OPENAI_API_KEY and an OpenAI Responses model or configure a hosted inference provider."
        )
    )

    assert failure["failure_category"] == "missing_provider_config"
    assert failure["failure_reason"] == "Hosted content generation is not configured for this runtime."
    assert failure["runtime_diagnostics"]["provider_status"] == "missing_provider_config"
    assert failure["runtime_diagnostics"]["runtime_ready"] is False


def test_generate_batch_adapter_packages_master_drafts(tmp_path: Path) -> None:
    adapter = GenerateBatchAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "posts": [
                    {
                      "topic": "Why routing drift happens",
                      "hook": "Most pipelines drift in small places first.",
                      "script_lines": ["The first miss looks harmless.", "But that is what compounds later."],
                      "caption": "Small drift creates big cleanup.",
                      "hashtags": ["#systems", "#architecture"],
                      "platform_notes": "Use a slow first beat and stress the pivot.",
                      "retention_check": "The second sentence reframes the problem.",
                      "suggested_post_time": "morning"
                    }
                  ]
                }
                """
            ]
        ),
    )
    request = ToolRequest(
        tool_id="content",
        capability="generate_batch",
        params={"topic": "architecture drift", "count": 1},
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "ok"
    assert result.structured_data["generated_count"] == 1
    assert (result.run_dir / "outputs" / "artifact_manifest.json").exists()


def test_generate_batch_adapter_returns_structured_error_when_topic_is_missing(tmp_path: Path) -> None:
    adapter = GenerateBatchAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport([]),
    )
    request = ToolRequest(
        tool_id="content",
        capability="generate_batch",
        params={},
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "error"
    assert result.structured_data["result_quality"] == "missing_inputs"
    assert result.structured_data["missing_inputs"] == ["topic"]
    assert "needs a topic" in result.summary.lower()


def test_format_platform_adapter_returns_variant_and_package(tmp_path: Path) -> None:
    adapter = FormatPlatformAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "topic": "Why routing drift happens",
                  "hook": "Routing drift rarely starts loud.",
                  "script_lines": ["It starts with one tiny mismatch.", "But that is what spreads through the system."],
                  "caption": "The first mismatch matters most.",
                  "hashtags": ["#systems", "#debugging"],
                  "platform_notes": "Open tight and emphasize the second line.",
                  "retention_check": "The pivot lands on the contrast line.",
                  "suggested_post_time": "evening"
                }
                """
            ]
        ),
    )
    request = ToolRequest(
        tool_id="content",
        capability="format_platform",
        params={
            "platform": "tiktok",
            "draft": {
                "id": "draft01",
                "topic": "Why routing drift happens",
                "hook": "Most pipelines drift in small places first.",
                "script_lines": ["The first miss looks harmless.", "But that is what compounds later."],
                "caption": "Small drift creates big cleanup.",
                "hashtags": ["#systems", "#architecture"],
                "platform_notes": "Use a slow first beat and stress the pivot.",
                "retention_check": "The second sentence reframes the problem.",
                "suggested_post_time": "morning",
                "style_profile": "observational",
                "platform": "master"
            },
        },
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "ok"
    assert result.structured_data["variant"]["platform"] == "tiktok"
    assert (result.run_dir / "outputs" / "artifact_manifest.json").exists()


def test_format_platform_adapter_accepts_plain_source_text(tmp_path: Path) -> None:
    adapter = FormatPlatformAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport(
            [
                """
                {
                  "topic": "Routing drift",
                  "hook": "Routing drift starts small.",
                  "script_lines": ["One tiny mismatch slips in.", "Then it spreads through the system."],
                  "caption": "Small drift adds up fast.",
                  "hashtags": ["#systems", "#architecture"],
                  "platform_notes": "Keep the pacing tight.",
                  "retention_check": "The contrast line lands in the second beat.",
                  "suggested_post_time": "afternoon"
                }
                """,
                """
                {
                  "topic": "Routing drift",
                  "hook": "Routing drift starts small.",
                  "script_lines": ["One tiny mismatch slips in.", "Then it spreads through the system."],
                  "caption": "Small drift adds up fast.",
                  "hashtags": ["#systems", "#architecture"],
                  "platform_notes": "Keep the pacing tight.",
                  "retention_check": "The contrast line lands in the second beat.",
                  "suggested_post_time": "afternoon"
                }
                """
            ]
        ),
    )
    request = ToolRequest(
        tool_id="content",
        capability="format_platform",
        params={
            "platform": "tiktok",
            "source_text": "Routing drift starts with one small mismatch.",
        },
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "ok"
    assert result.structured_data["variant"]["platform"] == "tiktok"
    assert result.structured_data["draft"]["platform"] == "master"


def test_format_platform_adapter_returns_structured_error_when_inputs_are_missing(tmp_path: Path) -> None:
    adapter = FormatPlatformAdapter(
        manifest=_manifest(),
        repo_root=tmp_path,
        transport=StubTransport([]),
    )
    request = ToolRequest(
        tool_id="content",
        capability="format_platform",
        params={},
        session_id="content-tests",
    )

    result = adapter.execute(request)

    assert result.status == "error"
    assert result.structured_data["result_quality"] == "missing_inputs"
    assert "platform" in result.structured_data["missing_inputs"]
