"""
Phase N Unit Tests — End-to-End Retrieval Pipeline (fetch_evidence)

Covers:
  - fetch_evidence(plan) public contract
  - Dynamic routing across all strategies:
      * VECTOR
      * HYBRID
      * MULTIMODAL
      * CROSS_DOCUMENT
  - Targeted sub-question overriding (sub_question parameter)
  - Provenance integrity on returned EvidencePacket
  - Dependency injection of test retrievers
  - Empty query handling
"""

import pytest
from typing import List, Dict, Any

from backend.app.query.pipeline import fetch_evidence
from backend.app.query.schemas import Plan, EvidencePacket, EvidenceUnit, RetrievalStrategy
from backend.app.index.vector_store import VectorStore
from backend.app.index.bm25 import BM25Index
from backend.app.models.embedder import MockEmbedder
from backend.app.query.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from backend.app.query.multimodal import MultimodalRetriever
from backend.app.query.graph import GraphRetriever
from backend.app.query.rerank import DeterministicScoreFusionReranker


def _build_test_pipeline():
    docs = [
        {
            "element_id": "doc1-p1-text-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 1,
            "page_label": "Page 1",
            "section_path": "Executive Summary",
            "element_type": "text",
            "content": "Overall corporate revenue reached $50M in 2023.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": None,
        },
        {
            "element_id": "doc1-p3-tbl-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 3,
            "page_label": "Page 3",
            "section_path": "Operations > Metrics",
            "element_type": "table",
            "content": "| Quarter | Line Efficiency | Output |\n| Q4 | 94.2% | 15000 |",
            "table_json": '[["Quarter", "Line Efficiency", "Output"], ["Q4", "94.2%", "15000"]]',
            "table_markdown": "| Quarter | Line Efficiency | Output |\n| Q4 | 94.2% | 15000 |",
            "visual_ref": None,
        },
        {
            "element_id": "doc1-p4-chart-01",
            "doc_id": "doc1",
            "document_title": "Annual Operations 2023",
            "page": 4,
            "page_label": "Page 4",
            "section_path": "Trends",
            "element_type": "chart",
            "content": "Line chart showing production efficiency growth from Q1 to Q4.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": "crops/chart1.png",
        },
    ]

    embedder = MockEmbedder(dim=4, mapping={
        "revenue": [1.0, 0.0, 0.0, 0.0],
        "efficiency": [0.0, 1.0, 0.0, 0.0],
    })

    vstore = VectorStore()
    for d in docs:
        vec = embedder.embed_text(d["content"])
        vstore.add_item(vec, d)

    bm25 = BM25Index()
    bm25.build_index(docs)

    vec_retriever = VectorRetriever(embedding_provider=embedder, vector_store=vstore)
    bm25_retriever = BM25Retriever(index=bm25)
    hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever, bm25_retriever=bm25_retriever)
    reranker = DeterministicScoreFusionReranker()
    multimodal_retriever = MultimodalRetriever(hybrid_retriever=hybrid_retriever, reranker=reranker)

    graph_retriever = GraphRetriever()
    graph_retriever.add_mock_edge(
        source_entity="Annual Operations 2023",
        target_entity="Operations Report",
        relation="cross_references",
        doc_id="doc2",
        evidence_content="Referenced data aligns across annual and quarterly documents.",
    )

    return {
        "vector_retriever": vec_retriever,
        "hybrid_retriever": hybrid_retriever,
        "multimodal_retriever": multimodal_retriever,
        "graph_retriever": graph_retriever,
        "reranker": reranker,
    }


class TestFetchEvidencePipeline:

    def setup_method(self):
        self.components = _build_test_pipeline()

    def test_vector_strategy_dispatch(self):
        plan = Plan(
            question="What was corporate revenue in 2023?",
            intent="fact_lookup",
            modalities=["text"],
        )
        packet = fetch_evidence(plan=plan, **self.components)

        assert isinstance(packet, EvidencePacket)
        assert packet.strategy == RetrievalStrategy.VECTOR
        assert len(packet.evidence_units) >= 1
        assert packet.evidence_units[0].element_id == "doc1-p1-text-01"
        assert packet.evidence_units[0].doc_id == "doc1"
        assert packet.evidence_units[0].page == 1

    def test_hybrid_strategy_dispatch(self):
        plan = Plan(
            question="What was production efficiency in Q4?",
            intent="fact_lookup",
            modalities=["text", "table"],
            periods=["Q4"],
        )
        packet = fetch_evidence(plan=plan, **self.components)

        assert isinstance(packet, EvidencePacket)
        assert packet.strategy == RetrievalStrategy.HYBRID
        assert len(packet.evidence_units) >= 1
        # Table item should be retrieved
        assert any(u.element_type == "table" for u in packet.evidence_units)

    def test_multimodal_strategy_dispatch(self):
        plan = Plan(
            question="What trend does the production efficiency chart show?",
            intent="visual_analysis",
            modalities=["chart"],
            query_type="visual",
        )
        packet = fetch_evidence(plan=plan, **self.components)

        assert isinstance(packet, EvidencePacket)
        assert packet.strategy == RetrievalStrategy.MULTIMODAL
        assert any(u.element_type == "chart" for u in packet.evidence_units)
        chart_unit = packet.filter_by_modality("chart")[0]
        assert chart_unit.visual_ref == "crops/chart1.png"

    def test_cross_document_strategy_dispatch(self):
        plan = Plan(
            question="Compare the annual report and operations report.",
            intent="cross_doc_synthesis",
            cross_document=True,
            modalities=["text"],
        )
        packet = fetch_evidence(plan=plan, **self.components)

        assert isinstance(packet, EvidencePacket)
        assert packet.strategy == RetrievalStrategy.CROSS_DOCUMENT
        assert len(packet.evidence_units) >= 1

    def test_sub_question_parameter_override(self):
        plan = Plan(
            question="Broad complex query about factory operations",
            intent="fact_lookup",
            modalities=["text"],
        )
        # Targeted sub-question
        packet = fetch_evidence(
            plan=plan,
            sub_question="corporate revenue",
            **self.components
        )
        assert packet.question == "corporate revenue"
        assert packet.retrieval_metadata["sub_question_used"] is True
        assert packet.evidence_units[0].element_id == "doc1-p1-text-01"

    def test_empty_query_returns_safe_packet(self):
        plan = Plan(question="")
        packet = fetch_evidence(plan=plan, **self.components)
        assert isinstance(packet, EvidencePacket)
        assert len(packet.evidence_units) == 0
        assert packet.retrieval_metadata["status"] == "empty_query"
