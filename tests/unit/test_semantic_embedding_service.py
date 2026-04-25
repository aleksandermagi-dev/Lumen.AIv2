import sys
from types import ModuleType

from lumen.semantic.embedding_service import SemanticEmbeddingService


def test_semantic_embedding_service_normalizes_and_hashes_stably() -> None:
    service = SemanticEmbeddingService()

    assert service.normalize_text("  sqlite   schema   migration  ") == "sqlite schema migration"
    assert service.content_hash("sqlite schema migration") == service.content_hash(" sqlite   schema migration ")


def test_semantic_embedding_service_pack_unpack_and_cosine_similarity() -> None:
    vector = [1.0, 0.0, 0.0]
    blob = SemanticEmbeddingService.pack_embedding(vector)
    unpacked = SemanticEmbeddingService.unpack_embedding(blob)

    assert unpacked == vector
    assert SemanticEmbeddingService.cosine_similarity([1.0, 0.0], [0.9, 0.1]) > 0.9
    assert SemanticEmbeddingService.cosine_similarity([1.0, 0.0], [0.0, 1.0]) < 0.1


def test_semantic_embedding_service_suppresses_runtime_stream_noise(monkeypatch, capsys) -> None:
    fake_module = ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            print(f"loading {model_name}")
            print("runtime warning stream", file=sys.stderr)

        def encode(
            self,
            texts,
            *,
            normalize_embeddings: bool,
            convert_to_numpy: bool,
        ):
            print("encoding progress")
            print("encoding stderr", file=sys.stderr)
            return [[1.0, 0.0] for _ in texts]

    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    service = SemanticEmbeddingService()

    assert service.embed_text("hello world") == [1.0, 0.0]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
