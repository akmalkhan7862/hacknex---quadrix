"""
Phase F Unit Tests — Hybrid Retrieval & Metadata Filtering

Covers:
  - reciprocal_rank_fusion() mathematical correctness and ranking
  - Vector-only retrieval via HybridRetriever.retrieve_vector_only()
  - BM25-only retrieval via HybridRetriever.retrieve_bm25_only()
  - Hybrid retrieval with RRF rank fusion boosting dual-match items
  - Metadata filtering:
      * document_id
      * period
      * content_type
      * section
      * entity
  - Empty query and empty store edge cases
  - Backward compatibility of hybrid_retrieve()
"""

import pytest
from typing import List, Dict, Any

from backend.app.index.embeddings import EmbeddingProvider
from backend.app.index.vector_store import VectorStore
from backend.app.index.bm25 import BM25Index
from backend.app.query.retriever import (
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    reciprocal_rank_fusion,
    global_hybrid_retriever,
    hybrid_retrieve,
)
from backend.app.query.schemas import EvidenceUnit


class _MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing."""
    def __init__(self, mapping: Dict[str, List[float]], default_dim: int = 4):
        self.mapping = mapping
        self._dim = default_dim
        self._raw_model = "mock"

    def embed_text(self, text: str) -> List[float]:
        for key, vec in self.mapping.items():
            if key.lower() in text.lower():
                return list(vec)
        return [0.1] * self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def is_fallback(self) -> bool:
        return False

    def _detect_dimension(self) -> int:
        return self._dim


def _create_test_data():
    docs = [
        {
            "element_id": "doc1-p1-text-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 1,
            "page_label": "Page 1",
            "section_path": "Executive Overview",
            "element_type": "text",
            "content": "Overall corporate revenue reached record levels in North America.",
            "table_json": None,
            "table_markdown": None,
        },
        {
            "element_id": "doc1-p3-table-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 3,
            "page_label": "Page 3",
            "section_path": "Manufacturing > Metrics",
            "element_type": "table",
            "content": "| Quarter | Efficiency | Facility |\n| Q4 | 91.5% | Dallas Plant |",
            "table_json": '[["Quarter", "Efficiency", "Facility"], ["Q4", "91.5%", "Dallas Plant"]]',
            "table_markdown": "| Quarter | Efficiency | Facility |\n| Q4 | 91.5% | Dallas Plant |",
        },
        {
            "element_id": "doc2-p5-text-02",
            "doc_id": "doc2",
            "document_title": "Quarterly Efficiency Review",
            "page": 5,
            "page_label": "Page 5",
            "section_path": "Operations > Dallas",
            "element_type": "text",
            "content": "Dallas Plant maintained high efficiency throughout Q4 with zero downtime.",
            "table_json": None,
            "table_markdown": None,
        },
    ]

    # Pre-calculated 4D unit vectors
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    v3 = [0.0, 0.707, 0.707, 0.0]

    vector_store = VectorStore()
    vector_store.add_item(v1, docs[0])
    vector_store.add_item(v2, docs[1])
    vector_store.add_item(v3, docs[2])

    bm25_index = BM25Index()
    bm25_index.build_index(docs)

    embedder = _MockEmbeddingProvider(
        mapping={
            "efficiency": [0.0, 1.0, 0.0, 0.0],
            "revenue": [1.0, 0.0, 0.0, 0.0],
        },
        default_dim=4
    )

    vec_retriever = VectorRetriever(embedding_provider=embedder, vector_store=vector_store)
    bm25_retriever = BM25Retriever(index=bm25_index)
    hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever, bm25_retriever=bm25_retriever, rrf_k=60)

    return docs, hybrid_retriever


class TestReciprocalRankFusion:

    def test_rrf_scoring_and_boost(self):
        list1 = [
            {"element_id": "item1", "vector_score": 0.9},
            {"element_id": "item2", "vector_score": 0.5},
        ]
        list2 = [
            {"element_id": "item2", "bm25_score": 5.0},
            {"element_id": "item3", "bm25_score": 3.0},
        ]

        fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_n=3)
        assert len(fused) == 3

        # item2 appears in both (rank 2 in list1, rank 1 in list2)
        # RRF score for item2 = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.032522
        # RRF score for item1 = 1/(60+1) = ~0.016393
        # item2 should rank #1 due to reciprocal rank fusion boost
        assert fused[0]["element_id"] == "item2"
        assert fused[0]["score"] > fused[1]["score"]
        assert "rrf_score" in fused[0]


class TestHybridRetrieverModes:

    def setup_method(self):
        self.docs, self.hybrid = _create_test_data()

    def test_vector_only_retrieval(self):
        units = self.hybrid.retrieve_vector_only("revenue", top_k=1)
        assert len(units) == 1
        assert units[0].element_id == "doc1-p1-text-01"
        assert "revenue" in units[0].content.lower()

    def test_bm25_only_retrieval(self):
        units = self.hybrid.retrieve_bm25_only("Dallas Plant", top_k=2)
        assert len(units) >= 1
        assert any("Dallas Plant" in u.content for u in units)

    def test_hybrid_retrieval_combines_streams(self):
        units = self.hybrid.retrieve("Q4 Dallas Plant efficiency", top_k=3)
        assert len(units) >= 2
        # doc1-p3-table-01 and doc2-p5-text-02 both contain Q4, Dallas, efficiency
        top_ids = [u.element_id for u in units]
        assert "doc1-p3-table-01" in top_ids
        assert "doc2-p5-text-02" in top_ids
        assert all(isinstance(u, EvidenceUnit) for u in units)
        assert all(u.score is not None for u in units)


class TestMetadataFiltering:

    def setup_method(self):
        self.docs, self.hybrid = _create_test_data()

    def test_filter_by_document_id(self):
        units = self.hybrid.retrieve("Dallas", top_k=5, filters={"document_id": "doc2"})
        assert len(units) == 1
        assert units[0].doc_id == "doc2"

    def test_filter_by_period(self):
        units = self.hybrid.retrieve("Dallas Plant", top_k=5, filters={"period": "Q4"})
        assert len(units) >= 1
        assert all("Q4" in u.content for u in units)

    def test_filter_by_content_type(self):
        units = self.hybrid.retrieve("Dallas Plant", top_k=5, filters={"content_type": "table"})
        assert len(units) == 1
        assert units[0].element_type == "table"
        assert units[0].element_id == "doc1-p3-table-01"

    def test_filter_by_section(self):
        units = self.hybrid.retrieve("Dallas", top_k=5, filters={"section": "Manufacturing"})
        assert len(units) == 1
        assert "Manufacturing" in units[0].section_path

    def test_filter_by_entity(self):
        units = self.hybrid.retrieve("efficiency", top_k=5, filters={"entity": "Dallas Plant"})
        assert len(units) >= 1
        assert all("Dallas Plant" in u.content for u in units)

    def test_combined_filters(self):
        units = self.hybrid.retrieve(
            "efficiency",
            top_k=5,
            filters={
                "document_id": "doc1",
                "content_type": "table",
                "period": "Q4"
            }
        )
        assert len(units) == 1
        assert units[0].element_id == "doc1-p3-table-01"


class TestHybridEdgeCasesAndCompat:

    def setup_method(self):
        self.docs, self.hybrid = _create_test_data()

    def test_empty_query(self):
        assert self.hybrid.retrieve("") == []
        assert self.hybrid.retrieve("   ") == []
        assert self.hybrid.retrieve_vector_only("") == []
        assert self.hybrid.retrieve_bm25_only("") == []

    def test_global_singleton(self):
        assert global_hybrid_retriever is not None
        assert isinstance(global_hybrid_retriever, HybridRetriever)

    def test_legacy_hybrid_retrieve_function(self):
        results = hybrid_retrieve("revenue", top_k=2)
        assert isinstance(results, list)
