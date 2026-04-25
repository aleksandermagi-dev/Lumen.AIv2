from __future__ import annotations

from array import array
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import math
from typing import Any
import warnings


class SemanticEmbeddingService:
    """Optional local sentence-transformers embedding service for memory-item content."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, *, model_name: str | None = None):
        self.model_name = str(model_name or self.MODEL_NAME).strip() or self.MODEL_NAME
        self._model: Any | None = None
        self._import_error: str | None = None

    def is_available(self) -> bool:
        try:
            self._load_model()
        except RuntimeError:
            return False
        return True

    def availability_status(self) -> dict[str, object]:
        available = self.is_available()
        return {
            "available": available,
            "model_name": self.model_name,
            "error": None if available else self._import_error,
        }

    def normalize_text(self, text: str | None) -> str:
        return " ".join(str(text or "").strip().split())

    def content_hash(self, text: str | None) -> str:
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def embed_text(self, text: str | None) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def embed_texts(self, texts: list[str | None]) -> list[list[float]]:
        model = self._load_model()
        normalized = [self.normalize_text(item) for item in texts]
        with self._quiet_runtime():
            vectors = model.encode(
                normalized,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return [self._vector_to_list(vector) for vector in vectors]

    @staticmethod
    def pack_embedding(vector: list[float]) -> bytes:
        payload = array("f", [float(value) for value in vector])
        return payload.tobytes()

    @staticmethod
    def unpack_embedding(blob: bytes | bytearray | memoryview | None) -> list[float]:
        if blob is None:
            return []
        payload = array("f")
        payload.frombytes(bytes(blob))
        return [float(value) for value in payload]

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(math.fsum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(math.fsum(float(b) * float(b) for b in right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            with self._quiet_runtime():
                from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on local optional runtime
            self._import_error = str(exc)
            raise RuntimeError("sentence-transformers runtime is unavailable") from exc
        with self._quiet_runtime():
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _quiet_runtime():
        stack = ExitStack()
        stack.enter_context(redirect_stdout(StringIO()))
        stack.enter_context(redirect_stderr(StringIO()))
        warning_context = warnings.catch_warnings()
        stack.enter_context(warning_context)
        warnings.filterwarnings("ignore", module="huggingface_hub")
        warnings.filterwarnings("ignore", module="sentence_transformers")
        warnings.filterwarnings("ignore", module="transformers")
        return stack

    @staticmethod
    def _vector_to_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            values = vector.tolist()
            if isinstance(values, list):
                return [float(item) for item in values]
        if isinstance(vector, list):
            return [float(item) for item in vector]
        return [float(item) for item in vector]
