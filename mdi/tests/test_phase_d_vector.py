"""
Phase D Unit Tests — Vector Retrieval

Covers:
  - EmbeddingProvider: embed_text, embed_batch, dimension, model_name, is_fallback
  - EmbeddingProvider: embed_image raises NotImplementedError
  - get_embedding_provider() singleton behaviour
  - VectorRetriever: retrieve() on empty store → []
  - VectorRetriever: retrieve() returns EvidenceUnit with full provenance
  - VectorRetriever: nearest-neighbour ordering is correct
  - VectorRetriever: metadata filter — document_id
  - VectorRetriever: metadata filter — content_type / element_type
  - VectorRetriever: metadata filter — section substring
  - VectorRetriever: metadata filter — period keyword in content
  - VectorRetriever: metadata filter — page exact match
  - VectorRetriever: retrieve_raw() returns dicts
  - VectorRetriever: custom injected provider + store (DI)
  - backward-compatibility: hybrid_retrieve() still works
  - generate_embedding / generate_embeddings unchanged
"""

import pytest
import numpy as np
from typing import List

from backend.app.index.embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
    generate_embedding,
    generate_embeddings,
)
from backend.app.index.vector_store import VectorStore
from backend.app.query.retriever import (
    VectorRetriever,
    hybrid_retrieve,
    _apply_metadata_filters,
    global_vector_retriever,
)
from backend.app.query.schemas import EvidenceUnit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_unit_dict(
    element_id: str,
    content: str,
    element_type: str = "text",
    doc_id: str = "doc1",
    page: int = 1,
    section_path: str = "Section 1",
) -> dict:
    return {
        "element_id": element_id,
        "doc_id": doc_id,
        "document_title": "Test Document",
        "page": page,
        "page_label": f"Page {page}",
        "section_path": section_path,
        "element_type": element_type,
        "content": content,
        "table_json": None,
        "table_markdown": None,
    }


def _deterministic_vec(seed: int, dim: int = 8) -> List[float]:
    """Small deterministic unit vector for testing."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_store(*entries) -> VectorStore:
    """
    Build a VectorStore from (element_id, content, vector, extra_fields) tuples.
    Each entry is (dict_of_metadata, List[float] vector).
    """
    store = VectorStore()
    for meta, vec in entries:
        store.add_item(vec, meta)
    return store


class _FixedEmbedder(EmbeddingProvider):
    """Test embedding provider that returns a fixed vector for any text."""

    def __init__(self, fixed_vec: List[float]):
        self._fixed = fixed_vec
        self._dim = len(fixed_vec)
        self._raw_model = "test-fixed"

    def embed_text(self, text: str) -> List[float]:
        return list(self._fixed)

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self):
        return self._dim

    @property
    def model_name(self):
        return "test-fixed"

    @property
    def is_fallback(self):
        return False

    def _detect_dimension(self):
        return self._dim


# ---------------------------------------------------------------------------
# EmbeddingProvider tests
# ---------------------------------------------------------------------------

class TestEmbeddingProvider:

    def test_instantiates(self):
        p = EmbeddingProvider()
        assert p is not None

    def test_embed_text_returns_list(self):
        p = EmbeddingProvider()
        vec = p.embed_text("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_embed_text_dimension(self):
        p = EmbeddingProvider()
        vec = p.embed_text("test sentence")
        assert len(vec) == p.dimension

    def test_embed_batch_returns_list_of_lists(self):
        p = EmbeddingProvider()
        vecs = p.embed_batch(["sentence one", "sentence two"])
        assert isinstance(vecs, list)
        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)

    def test_embed_batch_empty(self):
        p = EmbeddingProvider()
        assert p.embed_batch([]) == []

    def test_dimension_property(self):
        p = EmbeddingProvider()
        assert isinstance(p.dimension, int)
        assert p.dimension == 384  # all-MiniLM-L6-v2 / fallback both produce 384

    def test_model_name_is_string(self):
        p = EmbeddingProvider()
        assert isinstance(p.model_name, str)
        assert len(p.model_name) > 0

    def test_is_fallback_is_bool(self):
        p = EmbeddingProvider()
        assert isinstance(p.is_fallback, bool)

    def test_embed_image_raises(self):
        p = EmbeddingProvider()
        with pytest.raises(NotImplementedError):
            p.embed_image("some/path.png")


class TestGetEmbeddingProvider:

    def test_returns_instance(self):
        p = get_embedding_provider()
        assert isinstance(p, EmbeddingProvider)

    def test_singleton(self):
        p1 = get_embedding_provider()
        p2 = get_embedding_provider()
        assert p1 is p2


# ---------------------------------------------------------------------------
# generate_embedding / generate_embeddings backward compat
# ---------------------------------------------------------------------------

class TestModuleLevelFunctions:

    def test_generate_embedding_returns_list(self):
        vec = generate_embedding("test text")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_generate_embeddings_batch(self):
        vecs = generate_embeddings(["one", "two", "three"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_generate_embedding_normalised(self):
        """All-MiniLM vectors are already normalised; fallback normalises too."""
        vec = generate_embedding("normalisation check")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01


# ---------------------------------------------------------------------------
# VectorRetriever tests
# ---------------------------------------------------------------------------

class TestVectorRetrieverEmpty:

    def test_empty_store_returns_empty(self):
        store = VectorStore()
        embedder = _FixedEmbedder(_deterministic_vec(0))
        retriever = VectorRetriever(embedding_provider=embedder, vector_store=store)
        results = retriever.retrieve("any question")
        assert results == []

    def test_empty_question_returns_empty(self):
        store = VectorStore()
        embedder = _FixedEmbedder(_deterministic_vec(0))
        retriever = VectorRetriever(embedding_provider=embedder, vector_store=store)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []


class TestVectorRetrieverProvenance:

    def setup_method(self):
        # Build a small store with 2 entries pointing in orthogonal directions
        dim = 8
        self.vec_a = _deterministic_vec(1, dim)
        self.vec_b = _deterministic_vec(2, dim)

        meta_a = _make_unit_dict("id-a", "Production efficiency Q4 was 92%.", page=3)
        meta_b = _make_unit_dict("id-b", "Revenue increased in H1.", page=5)

        self.store = _make_store((meta_a, self.vec_a), (meta_b, self.vec_b))
        self.embedder = _FixedEmbedder(self.vec_a)  # query aligned with vec_a
        self.retriever = VectorRetriever(
            embedding_provider=self.embedder,
            vector_store=self.store,
        )

    def test_returns_evidence_units(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=2)
        assert all(isinstance(r, EvidenceUnit) for r in results)

    def test_top_result_is_closest_vector(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=2)
        assert results[0].element_id == "id-a"

    def test_provenance_element_id_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].element_id == "id-a"

    def test_provenance_doc_id_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].doc_id == "doc1"

    def test_provenance_document_title_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].document_title == "Test Document"

    def test_provenance_page_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].page == 3

    def test_provenance_page_label_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].page_label == "Page 3"

    def test_provenance_section_path_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].section_path == "Section 1"

    def test_provenance_element_type_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].element_type == "text"

    def test_provenance_content_preserved(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert "Q4" in results[0].content

    def test_score_populated(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert results[0].score is not None
        assert results[0].score > 0.0

    def test_metadata_contains_vector_score(self):
        results = self.retriever.retrieve("Q4 efficiency", top_k=1)
        assert "vector_score" in results[0].metadata


class TestVectorRetrieverFilters:

    def setup_method(self):
        dim = 8
        shared_vec = _deterministic_vec(5, dim)

        self.metas = [
            {**_make_unit_dict("text-1", "Q4 production efficiency was 92%.", doc_id="doc_a"), "element_type": "text"},
            {**_make_unit_dict("table-1", "| Q4 | 92% | production |", doc_id="doc_a"), "element_type": "table"},
            {**_make_unit_dict("text-2", "Revenue grew in Q2 of 2023.", doc_id="doc_b", page=2, section_path="2. Revenue"), "element_type": "text"},
        ]
        self.store = VectorStore()
        for m in self.metas:
            self.store.add_item(shared_vec, m)

        self.embedder = _FixedEmbedder(shared_vec)
        self.retriever = VectorRetriever(
            embedding_provider=self.embedder,
            vector_store=self.store,
        )

    def test_filter_by_document_id(self):
        results = self.retriever.retrieve("efficiency", top_k=10, filters={"document_id": "doc_b"})
        assert all(r.doc_id == "doc_b" for r in results)
        assert len(results) == 1

    def test_filter_by_content_type_table(self):
        results = self.retriever.retrieve("efficiency", top_k=10, filters={"content_type": "table"})
        assert all(r.element_type == "table" for r in results)
        assert len(results) == 1

    def test_filter_by_content_type_text(self):
        results = self.retriever.retrieve("efficiency", top_k=10, filters={"content_type": "text"})
        assert all(r.element_type == "text" for r in results)

    def test_filter_by_section_substring(self):
        results = self.retriever.retrieve("revenue", top_k=10, filters={"section": "Revenue"})
        assert len(results) == 1
        assert "Revenue" in results[0].section_path

    def test_filter_by_period(self):
        results = self.retriever.retrieve("period", top_k=10, filters={"period": "Q2"})
        assert all("Q2" in r.content.upper() for r in results)

    def test_filter_by_page(self):
        results = self.retriever.retrieve("something", top_k=10, filters={"page": 2})
        assert all(r.page == 2 for r in results)

    def test_no_filter_returns_all(self):
        results = self.retriever.retrieve("efficiency", top_k=10)
        assert len(results) == 3

    def test_overly_restrictive_filter_returns_empty(self):
        results = self.retriever.retrieve("efficiency", top_k=10, filters={"document_id": "nonexistent_doc"})
        assert results == []


class TestVectorRetrieverRetrieveRaw:

    def test_retrieve_raw_returns_dicts(self):
        dim = 8
        vec = _deterministic_vec(7, dim)
        meta = _make_unit_dict("raw-1", "Some content here.")
        store = _make_store((meta, vec))
        embedder = _FixedEmbedder(vec)
        retriever = VectorRetriever(embedding_provider=embedder, vector_store=store)
        results = retriever.retrieve_raw("some content")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_raw_contains_element_id(self):
        dim = 8
        vec = _deterministic_vec(7, dim)
        meta = _make_unit_dict("raw-id", "Content text.")
        store = _make_store((meta, vec))
        embedder = _FixedEmbedder(vec)
        retriever = VectorRetriever(embedding_provider=embedder, vector_store=store)
        results = retriever.retrieve_raw("content")
        assert results[0]["element_id"] == "raw-id"


class TestVectorRetrieverProperties:

    def test_embedding_provider_property(self):
        p = _FixedEmbedder(_deterministic_vec(0))
        r = VectorRetriever(embedding_provider=p)
        assert r.embedding_provider is p

    def test_vector_store_property(self):
        s = VectorStore()
        r = VectorRetriever(vector_store=s)
        assert r.vector_store is s

    def test_global_singleton_exists(self):
        assert global_vector_retriever is not None
        assert isinstance(global_vector_retriever, VectorRetriever)


# ---------------------------------------------------------------------------
# _apply_metadata_filters tests
# ---------------------------------------------------------------------------

class TestApplyMetadataFilters:

    def _items(self):
        return [
            {"element_id": "a", "doc_id": "d1", "element_type": "text", "section_path": "Intro", "content": "Q4 data", "page": 1},
            {"element_id": "b", "doc_id": "d2", "element_type": "table", "section_path": "Summary", "content": "Revenue Q2", "page": 2},
        ]

    def test_no_filter(self):
        assert len(_apply_metadata_filters(self._items(), {})) == 2

    def test_doc_id_filter(self):
        out = _apply_metadata_filters(self._items(), {"document_id": "d1"})
        assert len(out) == 1 and out[0]["element_id"] == "a"

    def test_content_type_filter(self):
        out = _apply_metadata_filters(self._items(), {"content_type": "table"})
        assert len(out) == 1 and out[0]["element_id"] == "b"

    def test_section_filter(self):
        out = _apply_metadata_filters(self._items(), {"section": "intro"})
        assert len(out) == 1 and out[0]["element_id"] == "a"

    def test_period_filter(self):
        out = _apply_metadata_filters(self._items(), {"period": "Q2"})
        assert len(out) == 1 and "Q2" in out[0]["content"]

    def test_page_filter(self):
        out = _apply_metadata_filters(self._items(), {"page": 2})
        assert len(out) == 1 and out[0]["page"] == 2


# ---------------------------------------------------------------------------
# Backward compatibility — hybrid_retrieve still works
# ---------------------------------------------------------------------------

class TestHybridRetrieveBackwardCompat:

    def test_hybrid_retrieve_returns_list(self):
        result = hybrid_retrieve("test question", top_k=5)
        assert isinstance(result, list)

    def test_hybrid_retrieve_empty_store(self):
        # Global store may be empty if no docs ingested — should return []
        result = hybrid_retrieve("production efficiency", top_k=5)
        assert isinstance(result, list)
