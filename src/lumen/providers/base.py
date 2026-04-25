from __future__ import annotations

from abc import ABC, abstractmethod

from lumen.providers.models import InferenceRequest, InferenceResult, ProviderCapabilities


class ModelProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def deployment_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError

    def stream(self, request: InferenceRequest) -> object:
        raise NotImplementedError(f"{self.provider_id} does not implement streaming inference")

    def submit_background(self, request: InferenceRequest) -> object:
        raise NotImplementedError(f"{self.provider_id} does not implement background inference")
