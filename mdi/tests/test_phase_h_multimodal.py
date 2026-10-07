"""
Phase H Unit Tests — Multimodal Retrieval & EvidencePacket Generation

Covers:
  - MultimodalRetriever specialized paths:
      * retrieve_text() -> element_type="text"
      * retrieve_tables() -> element_type="table"
      * retrieve_visuals() -> element_type="chart" / "image"
  - MultimodalRetriever.retrieve(plan):
      * table-specific query plan -> table evidence retrieved
      * chart-specific query plan -> visual evidence retrieved
      * complex multimodal query plan -> heterogeneous evidence packet
  - EvidencePacket structure & helper methods:
      * question, plan, strategy, evidence_units, retrieval_metadata
      * get_unit_by_id()
      * filter_by_modality()
  - Provenance preservation across modalities
  - Empty query handling
"""

import pytest
from typing import List, Dict, Any

from backend.app.index.embeddings import EmbeddingProvider
from backend.app.index.vector_store import VectorStore
from backend.app.index.bm25 import BM25Index
from backend.app.query.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from backend.app.query.rerank import DeterministicScoreFusionReranker
from backend.app.query.multimodal import MultimodalRetriever, global_multimodal_retriever
from backend.app.query.schemas import Plan, EvidencePacket, EvidenceUnit, RetrievalStrategy


class _MockEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 4):
        self._dim = dim
        self._raw_model = "mock"

    def embed_text(self, text: str) -> List[float]:
        # Simple hash-based deterministic unit vector
        v = [0.1] * self._dim
        v[abs(hash(text)) % self._dim] = 1.0
        return v

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


def _create_multimodal_corpus():
    items = [
        {
            "element_id": "doc1-p1-text-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 1,
            "page_label": "Page 1",
            "section_path": "Summary",
            "element_type": "text",
            "content": "Production efficiency reached 92% in Q4 across automated assembly lines.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": None,
        },
        {
            "element_id": "doc1-p3-table-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 3,
            "page_label": "Page 3",
            "section_path": "Metrics",
            "element_type": "table",
            "content": "| Quarter | Line Efficiency | Output |\n| Q4 | 92.4% | 15000 |",
            "table_json": '[["Quarter", "Line Efficiency", "Output"], ["Q4", "92.4%", "15000"]]',
            "table_markdown": "| Quarter | Line Efficiency | Output |\n| Q4 | 92.4% | 15000 |",
            "visual_ref": None,
        },
        {
            "element_id": "doc1-p4-chart-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 4,
            "page_label": "Page 4",
            "section_path": "Visualizations",
            "element_type": "chart",
            "content": "Chart showing quarterly efficiency trends from Q1 to Q4 demonstrating steady growth.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": "data/processed/crops/doc1-p4-chart-01.png",
        },
        {
            "element_id": "doc1-p5-img-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 5,
            "page_label": "Page 5",
            "section_path": "Facility Photos",
            "element_type": "image",
            "content": "Photograph of the assembly floor equipment and quality control station.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": "data/processed/crops/doc1-p5-img-01.png",
        },
    ]

    embedder = _MockEmbedder()
    vstore = VectorStore()
    for item in items:
        vec = embedder.embed_text(item["content"])
        vstore.add_item(vec, item)

    bm25 = BM25Index()
    bm25.build_index(items)

    vec_retriever = VectorRetriever(embedding_provider=embedder, vector_store=vstore)
    bm25_retriever = BM25Retriever(index=bm25)
    hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever, bm25_retriever=bm25_retriever)
    reranker = DeterministicScoreFusionReranker()

    retriever = MultimodalRetriever(hybrid_retriever=hybrid_retriever, reranker=reranker)
    return items, retriever


class TestMultimodalSpecializedPaths:

    def setup_method(self):
        self.items, self.retriever = _create_multimodal_corpus()

    def test_retrieve_text_only(self):
        units = self.retriever.retrieve_text("production efficiency", top_k=5)
        assert len(units) >= 1
        assert all(u.element_type == "text" for u in units)

    def test_retrieve_tables_only(self):
        units = self.retriever.retrieve_tables("Line Efficiency", top_k=5)
        assert len(units) >= 1
        assert all(u.element_type == "table" for u in units)
        assert units[0].element_id == "doc1-p3-table-01"

    def test_retrieve_visuals_chart_and_image(self):
        visuals = self.retriever.retrieve_visuals("quarterly efficiency trends chart", top_k=5)
        assert len(visuals) >= 1
        assert all(v.element_type in ("chart", "image") for v in visuals)
        assert visuals[0].element_id == "doc1-p4-chart-01"
        assert visuals[0].visual_ref is not None


class TestMultimodalRetrievePlan:

    def setup_method(self):
        self.items, self.retriever = _create_multimodal_corpus()

    def test_table_plan_retrieval(self):
        plan = Plan(
            question="What was production efficiency in Q4?",
            intent="fact_lookup",
            modalities=["text", "table"],
            periods=["Q4"],
        )
        packet = self.retriever.retrieve(plan, top_k=5)
        assert isinstance(packet, EvidencePacket)
        assert packet.question == plan.question
        assert packet.strategy == RetrievalStrategy.HYBRID

        # Should contain table unit
        table_units = packet.filter_by_modality("table")
        assert len(table_units) >= 1
        assert table_units[0].element_id == "doc1-p3-table-01"

    def test_chart_plan_retrieval(self):
        plan = Plan(
            question="What trend does the production efficiency chart show?",
            intent="visual_analysis",
            modalities=["chart"],
            query_type="visual",
        )
        packet = self.retriever.retrieve(plan, top_k=5)
        assert isinstance(packet, EvidencePacket)
        assert packet.strategy == RetrievalStrategy.MULTIMODAL

        chart_units = packet.filter_by_modality("chart")
        assert len(chart_units) >= 1
        assert chart_units[0].element_id == "doc1-p4-chart-01"
        assert chart_units[0].visual_ref == "data/processed/crops/doc1-p4-chart-01.png"

    def test_complex_multimodal_plan(self):
        plan = Plan(
            question="Compare Q2 and Q4 production efficiency using tables and charts.",
            intent="comparison",
            modalities=["text", "table", "chart"],
            periods=["Q2", "Q4"],
        )
        packet = self.retriever.retrieve(plan, top_k=5)
        assert isinstance(packet, EvidencePacket)
        assert len(packet.evidence_units) >= 2

        # Check metadata contains counts
        meta = packet.retrieval_metadata
        assert "searched_paths" in meta
        assert "visual" in meta["searched_paths"]
        assert "table" in meta["searched_paths"]

    def test_evidence_packet_helpers(self):
        plan = Plan(question="Test question", modalities=["text", "table"])
        packet = self.retriever.retrieve(plan, top_k=5)

        # Lookup by ID
        found = packet.get_unit_by_id("doc1-p3-table-01")
        assert found is not None
        assert found.element_type == "table"

        not_found = packet.get_unit_by_id("invalid-id")
        assert not_found is None

    def test_empty_question(self):
        plan = Plan(question="")
        packet = self.retriever.retrieve(plan)
        assert len(packet.evidence_units) == 0
        assert packet.retrieval_metadata["status"] == "empty_question"

    def test_global_singleton(self):
        assert global_multimodal_retriever is not None
        assert isinstance(global_multimodal_retriever, MultimodalRetriever)
