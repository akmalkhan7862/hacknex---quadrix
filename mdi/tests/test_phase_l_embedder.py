"""
Phase L Unit Tests — Embedding Abstraction

Covers:
  - EmbeddingProvider ABC interface & embed_image NotImplementedError
  - DeterministicFallbackEmbedder: text and batch embeddings, normalization
  - SentenceTransformerEmbedder: neural embeddings with graceful offline fallback
  - MockEmbedder: custom dictionary mappings and vector dimension
  - get_embedder() factory & global_embedder singleton
  - Interchangeability with VectorRetriever (dependency injection)
"""

import pytest
import numpy as np

from backend.app.models.embedder import (
    EmbeddingProvider,
    DeterministicFallbackEmbedder,
    SentenceTransformerEmbedder,
    MockEmbedder,
    get_embedder,
    global_embedder,
)
from backend.app.index.vector_store import VectorStore
from backend.app.query.retriever import VectorRetriever


class TestEmbeddingProviderABC:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_embed_image_hook_raises(self):
        embedder = DeterministicFallbackEmbedder(dim=128)
        with pytest.raises(NotImplementedError) as exc_info:
            embedder.embed_image("crops/test.png")
        assert "not required" in str(exc_info.value).lower()


class TestDeterministicFallbackEmbedder:

    def setup_method(self):
        self.embedder = DeterministicFallbackEmbedder(dim=384)

    def test_embed_text_dimension_and_norm(self):
        vec = self.embedder.embed_text("production efficiency in Q4")
        assert isinstance(vec, list)
        assert len(vec) == 384
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self):
        vecs = self.embedder.embed_batch(["text one", "text two"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384
        assert len(vecs[1]) == 384

    def test_properties(self):
        assert self.embedder.dimension == 384
        assert "deterministic-fallback" in self.embedder.model_name


class TestSentenceTransformerEmbedder:

    def test_embed_text(self):
        embedder = SentenceTransformerEmbedder()
        vec = embedder.embed_text("sample input text")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_batch(self):
        embedder = SentenceTransformerEmbedder()
        vecs = embedder.embed_batch(["batch item 1", "batch item 2"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_graceful_fallback_on_invalid_model(self):
        embedder = SentenceTransformerEmbedder(model_name="non_existent_model_xyz_123")
        vec = embedder.embed_text("fallback test")
        assert isinstance(vec, list)
        assert len(vec) == 384


class TestMockEmbedder:

    def test_custom_mapping(self):
        custom_mapping = {
            "revenue": [1.0, 0.0, 0.0, 0.0],
            "cost": [0.0, 1.0, 0.0, 0.0],
        }
        embedder = MockEmbedder(dim=4, mapping=custom_mapping)

        v_rev = embedder.embed_text("Total revenue in 2023")
        assert v_rev == [1.0, 0.0, 0.0, 0.0]

        v_cost = embedder.embed_text("Operating cost decreased")
        assert v_cost == [0.0, 1.0, 0.0, 0.0]

        # Non-mapped fallback
        v_other = embedder.embed_text("unrelated text")
        assert len(v_other) == 4
        assert sum(v_other) == 1.0


class TestVectorRetrieverInterchangeability:

    def test_inject_mock_embedder_into_vector_retriever(self):
        mock_emb = MockEmbedder(dim=4, mapping={"Dallas": [1.0, 0.0, 0.0, 0.0]})
        store = VectorStore()
        doc = {
            "element_id": "e1",
            "doc_id": "d1",
            "document_title": "Doc",
            "page": 1,
            "page_label": "P1",
            "section_path": "Sec",
            "element_type": "text",
            "content": "Dallas Plant output.",
        }
        store.add_item([1.0, 0.0, 0.0, 0.0], doc)

        retriever = VectorRetriever(embedding_provider=mock_emb, vector_store=store)
        results = retriever.retrieve("Dallas query", top_k=1)
        assert len(results) == 1
        assert results[0].element_id == "e1"

    def test_global_singleton(self):
        assert global_embedder is not None
        assert isinstance(global_embedder, EmbeddingProvider)
