from __future__ import annotations

from lumen.providers.base import ModelProvider
from lumen.providers.models import InferenceRequest, InferenceResult, ProviderCapabilities


class LocalOnlyProvider(ModelProvider):
    @property
    def provider_id(self) -> str:
        return "local"

    @property
    def deployment_mode(self) -> str:
        return "local_only"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_sync=False,
            supports_streaming=False,
            supports_async=False,
            supports_background=False,
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        raise RuntimeError(
            "LocalOnlyProvider does not supply hosted inference. "
            "Lumen orchestration remains local in local_only mode."
        )
