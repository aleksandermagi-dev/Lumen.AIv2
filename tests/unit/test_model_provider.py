from pathlib import Path

from lumen.app.settings import AppSettings
from lumen.providers.factory import build_model_provider
from lumen.providers.local_provider import LocalOnlyProvider
from lumen.providers.models import InferenceRequest
from lumen.providers.openai_responses_provider import OpenAIResponsesProvider


def test_provider_factory_defaults_to_local_only(tmp_path: Path) -> None:
    settings = AppSettings.from_repo_root(tmp_path)

    provider = build_model_provider(settings)

    assert isinstance(provider, LocalOnlyProvider)
    assert provider.provider_id == "local"
    assert provider.deployment_mode == "local_only"
    assert provider.capabilities().to_dict() == {
        "supports_sync": False,
        "supports_streaming": False,
        "supports_async": False,
        "supports_background": False,
    }


def test_provider_factory_builds_openai_responses_provider(tmp_path: Path) -> None:
    (tmp_path / "lumen.toml").write_text(
        "\n".join(
            [
                "[app]",
                'deployment_mode = "hybrid"',
                'inference_provider = "openai_responses"',
                'openai_api_base = "https://api.openai.com/v1"',
                'openai_responses_model = "gpt-5"',
                "provider_timeout_seconds = 45",
            ]
        ),
        encoding="utf-8",
    )
    settings = AppSettings.from_repo_root(tmp_path)

    provider = build_model_provider(settings)

    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.provider_id == "openai_responses"
    assert provider.deployment_mode == "hybrid"
    assert provider.api_base == "https://api.openai.com/v1"
    assert provider.default_model == "gpt-5"
    assert provider.timeout_seconds == 45
    assert provider.capabilities().to_dict() == {
        "supports_sync": True,
        "supports_streaming": True,
        "supports_async": True,
        "supports_background": True,
    }


def test_provider_factory_uses_openai_env_defaults_when_key_is_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_RESPONSES_MODEL", "gpt-test")

    settings = AppSettings.from_repo_root(tmp_path)
    provider = build_model_provider(settings)

    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.deployment_mode == "hybrid"
    assert provider.default_model == "gpt-test"


def test_openai_responses_provider_builds_responses_payload() -> None:
    provider = OpenAIResponsesProvider(
        deployment_mode_value="hosted",
        api_base="https://api.openai.com/v1",
        default_model="gpt-5",
    )

    payload = provider.build_payload(
        request=InferenceRequest(
            model=None,
            instructions="Be concise.",
            input_text="Summarize the session.",
            metadata={"session_id": "default"},
            temperature=0.2,
            max_output_tokens=400,
        )
    )

    assert payload == {
        "model": "gpt-5",
        "input": "Summarize the session.",
        "instructions": "Be concise.",
        "metadata": {"session_id": "default"},
        "temperature": 0.2,
        "max_output_tokens": 400,
    }


def test_openai_responses_provider_infer_returns_text(monkeypatch) -> None:
    provider = OpenAIResponsesProvider(
        deployment_mode_value="hybrid",
        api_base="https://api.openai.com/v1",
        default_model="gpt-test",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return (
                b'{"model":"gpt-test","status":"completed","output":[{"content":[{"text":"Black holes warp spacetime."}]}]}'
            )

    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = http_request.data
        captured["authorization"] = http_request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("lumen.providers.openai_responses_provider.urllib_request.urlopen", fake_urlopen)

    result = provider.infer(
        InferenceRequest(
            model=None,
            instructions="Answer clearly.",
            input_text="Tell me about black holes.",
            metadata={"session_id": "desktop"},
            temperature=0.4,
            max_output_tokens=500,
        )
    )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["timeout"] == 30
    assert captured["authorization"] == "Bearer test-key"
    assert b'"model": "gpt-test"' in captured["body"]
    assert result.provider_id == "openai_responses"
    assert result.model == "gpt-test"
    assert result.output_text == "Black holes warp spacetime."
