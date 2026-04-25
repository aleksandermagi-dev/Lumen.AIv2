from __future__ import annotations

from lumen.app.settings import AppSettings
from lumen.providers.base import ModelProvider
from lumen.providers.local_provider import LocalOnlyProvider
from lumen.providers.openai_responses_provider import OpenAIResponsesProvider


def build_model_provider(settings: AppSettings) -> ModelProvider:
    provider_id = settings.inference_provider
    if settings.deployment_mode == "local_only":
        return LocalOnlyProvider()
    if provider_id == "openai_responses":
        return OpenAIResponsesProvider(
            deployment_mode_value=settings.deployment_mode,
            api_base=settings.openai_api_base,
            default_model=settings.openai_responses_model,
            timeout_seconds=settings.provider_timeout_seconds,
        )
    if provider_id == "local":
        return LocalOnlyProvider()
    raise ValueError(f"Unsupported inference provider: {provider_id}")
