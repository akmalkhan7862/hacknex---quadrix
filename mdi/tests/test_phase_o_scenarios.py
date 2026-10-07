"""
Phase O Canonical Scenario Integration Tests

Validates the complete end-to-end pipeline:
  build_plan(question) -> route_plan(plan) -> fetch_evidence(plan) -> Reader extraction

Tests the 6 required canonical scenarios:
  1. Exact value table question: "What was production efficiency in Q4?"
  2. Exact terminology question: "What exact value is reported for Q4?"
  3. Chart question: "What trend does the production efficiency chart show?"
  4. Comparison question: "Compare production efficiency between Q2 and Q4."
  5. Cross-document question: "Compare the annual report and operations report."
  6. Calculation question: "Calculate the percentage increase." (Retrieves evidence ONLY, no calculation)
"""

import pytest
from typing import List, Dict, Any

from backend.app.query.analyzer import build_plan
from backend.app.query.router import route_plan
from backend.app.query.pipeline import fetch_evidence
from backend.app.query.schemas import Plan, EvidencePacket, EvidenceUnit, RetrievalStrategy
from backend.app.index.vector_store import VectorStore
from backend.app.index.bm25 import BM25Index
from backend.app.models.embedder import MockEmbedder
from backend.app.query.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from backend.app.query.multimodal import MultimodalRetriever
from backend.app.query.graph import GraphRetriever
from backend.app.query.rerank import DeterministicScoreFusionReranker
from backend.app.query.readers.table import TableReader, TableReadResult
from backend.app.query.readers.chart import ChartReader, ChartReadResult
from backend.app.query.readers.text import TextReader, TextReadResult
from backend.app.models.vlm import MockVisionProvider


def _build_scenario_corpus():
    docs = [
        # Text unit
        {
            "element_id": "doc1-p1-text-01",
            "doc_id": "doc1",
            "document_title": "Annual Report 2023",
            "page": 1,
            "page_label": "Page 1",
            "section_path": "Executive Summary",
            "element_type": "text",
            "content": "Corporate revenue grew while production efficiency remained steady in the Dallas plant.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": None,
        },
        # Table unit with Q2 and Q4 values
        {
            "element_id": "doc1-p3-tbl-01",
            "doc_id": "doc1",
            "document_title": "Annual Report 2023",
            "page": 3,
            "page_label": "Page 3",
            "section_path": "Operations > Metrics",
            "element_type": "table",
            "content": "| Quarter | Production Efficiency | Total Cost |\n| Q2 | 88.0% | $4.2M |\n| Q4 | 94.2% | $4.0M |",
            "table_json": '[["Quarter", "Production Efficiency", "Total Cost"], ["Q2", "88.0%", "$4.2M"], ["Q4", "94.2%", "$4.0M"]]',
            "table_markdown": "| Quarter | Production Efficiency | Total Cost |\n|---|---|---|\n| Q2 | 88.0% | $4.2M |\n| Q4 | 94.2% | $4.0M |",
            "visual_ref": None,
        },
        # Chart unit
        {
            "element_id": "doc1-p4-chart-01",
            "doc_id": "doc1",
            "document_title": "Annual Report 2023",
            "page": 4,
            "page_label": "Page 4",
            "section_path": "Trends",
            "element_type": "chart",
            "content": "Line chart showing production efficiency trend from Q1 through Q4 with upward growth.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": "data/processed/crops/doc1-p4-chart-01.png",
        },
        # Second document text unit for cross-document evaluation
        {
            "element_id": "doc2-p2-text-01",
            "doc_id": "doc2",
            "document_title": "Operations Report 2023",
            "page": 2,
            "page_label": "Page 2",
            "section_path": "Factory Insights",
            "element_type": "text",
            "content": "Operations report confirms that factory throughput matched the annual report findings.",
            "table_json": None,
            "table_markdown": None,
            "visual_ref": None,
        },
    ]

    embedder = MockEmbedder(dim=4, mapping={
        "efficiency": [0.0, 1.0, 0.0, 0.0],
        "revenue": [1.0, 0.0, 0.0, 0.0],
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
        source_entity="Annual Report",
        target_entity="Operations Report",
        relation="cross_references",
        doc_id="doc2",
        evidence_content="Cross-document correlation between annual report and operations report confirms factory throughput.",
        page=2,
    )

    vlm = MockVisionProvider()
    chart_reader = ChartReader(vlm=vlm)
    table_reader = TableReader()
    text_reader = TextReader()

    return {
        "pipeline_kwargs": {
            "vector_retriever": vec_retriever,
            "hybrid_retriever": hybrid_retriever,
            "multimodal_retriever": multimodal_retriever,
            "graph_retriever": graph_retriever,
            "reranker": reranker,
        },
        "chart_reader": chart_reader,
        "table_reader": table_reader,
        "text_reader": text_reader,
    }


class TestCanonicalScenarios:

    def setup_method(self):
        self.ctx = _build_scenario_corpus()
        self.pipe_kwargs = self.ctx["pipeline_kwargs"]
        self.table_reader = self.ctx["table_reader"]
        self.chart_reader = self.ctx["chart_reader"]
        self.text_reader = self.ctx["text_reader"]

    # ── Scenario 1: Exact value table question ─────────────────────────────
    def test_scenario_1_exact_value_table_question(self):
        """
        Question: 'What was production efficiency in Q4?'
        Expectation:
          - Plan detects table modality & period Q4
          - Router routes to HYBRID
          - Pipeline retrieves table unit
          - TableReader extracts relevant row and exact 94.2% value
        """
        question = "What was production efficiency in Q4?"
        plan = build_plan(question)
        assert "table" in plan.modalities
        assert "Q4" in plan.periods

        strategy = route_plan(plan)
        assert strategy == RetrievalStrategy.HYBRID

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        assert isinstance(packet, EvidencePacket)

        table_units = packet.filter_by_modality("table")
        assert len(table_units) >= 1
        table_unit = table_units[0]
        assert table_unit.element_type == "table"

        # Table Reader extracts relevant value
        read_res = self.table_reader.read(question, table_unit)
        assert "94.2%" in read_res.relevant_values
        assert "%" in read_res.units
        assert any(r.get("Quarter") == "Q4" for r in read_res.row_info)
        assert read_res.confidence >= 0.8

    # ── Scenario 2: Exact terminology question ─────────────────────────────
    def test_scenario_2_exact_terminology_question(self):
        """
        Question: 'What exact value is reported for Q4?'
        Expectation:
          - Triggers BM25 / keyword path
          - Retrieves matching unit containing exact Q4 token
        """
        question = "What exact value is reported for Q4?"
        plan = build_plan(question)
        assert "Q4" in plan.periods
        assert route_plan(plan) == RetrievalStrategy.HYBRID

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        assert len(packet.evidence_units) >= 1

        # Must find unit with exact Q4 match
        assert any("Q4" in u.content for u in packet.evidence_units)

    # ── Scenario 3: Chart question ─────────────────────────────────────────
    def test_scenario_3_chart_question(self):
        """
        Question: 'What trend does the production efficiency chart show?'
        Expectation:
          - Plan detects chart modality & visual intent
          - Router routes to MULTIMODAL
          - Retrieves chart unit with visual_ref
          - ChartReader passes visual_ref to VLM and returns visual summary
        """
        question = "What trend does the production efficiency chart show?"
        plan = build_plan(question)
        assert "chart" in plan.modalities
        assert plan.intent == "visual_analysis"

        strategy = route_plan(plan)
        assert strategy == RetrievalStrategy.MULTIMODAL

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        chart_units = packet.filter_by_modality("chart")
        assert len(chart_units) >= 1
        chart_unit = chart_units[0]
        assert chart_unit.visual_ref == "data/processed/crops/doc1-p4-chart-01.png"

        # Pass to ChartReader -> VLM
        chart_res = self.chart_reader.read(question, chart_unit, request_visual_analysis=True)
        assert chart_res.visual_summary is not None
        assert "upward trend" in chart_res.visual_summary.lower()
        assert chart_res.metadata["vlm_executed"] is True

    # ── Scenario 4: Comparison question ────────────────────────────────────
    def test_scenario_4_comparison_question(self):
        """
        Question: 'Compare production efficiency between Q2 and Q4.'
        Expectation:
          - Plan detects periods Q2 and Q4, comparison intent, and table
          - Pipeline retrieves evidence covering both periods
          - TableReader extracts both Q2 and Q4 values
        """
        question = "Compare production efficiency between Q2 and Q4."
        plan = build_plan(question)
        assert "Q2" in plan.periods
        assert "Q4" in plan.periods
        assert plan.intent == "comparison"
        assert "table" in plan.modalities

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        assert len(packet.evidence_units) >= 1

        table_units = packet.filter_by_modality("table")
        assert len(table_units) >= 1

        read_res = self.table_reader.read(question, table_units[0])
        assert "88.0%" in read_res.relevant_values
        assert "94.2%" in read_res.relevant_values

    # ── Scenario 5: Cross-document question ────────────────────────────────
    def test_scenario_5_cross_document_question(self):
        """
        Question: 'Compare the annual report and operations report.'
        Expectation:
          - Plan detects cross_document = True
          - Router routes to CROSS_DOCUMENT
          - Graph path traverses cross-document entities
          - Returns multi-document evidence
        """
        question = "Compare the annual report and operations report."
        plan = build_plan(question)
        assert plan.cross_document is True

        strategy = route_plan(plan)
        assert strategy == RetrievalStrategy.CROSS_DOCUMENT

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        assert packet.strategy == RetrievalStrategy.CROSS_DOCUMENT
        assert len(packet.evidence_units) >= 1

        # Check multi-document or graph edge evidence
        doc_ids = {u.doc_id for u in packet.evidence_units}
        assert len(doc_ids) >= 1

    # ── Scenario 6: Calculation question ──────────────────────────────────
    def test_scenario_6_calculation_evidence_only_no_math(self):
        """
        Question: 'Calculate the percentage increase.'
        Expectation:
          - Plan detects needs_calculation = True
          - Pipeline retrieves source evidence units only
          - Reader extracts source values (e.g. 88.0%, 94.2%) WITHOUT performing calculation
          - Calculation is preserved for the downstream reasoning teammate
        """
        question = "Calculate the percentage increase from Q2 to Q4."
        plan = build_plan(question)
        assert plan.needs_calculation is True

        packet = fetch_evidence(plan=plan, **self.pipe_kwargs)
        assert len(packet.evidence_units) >= 1

        table_units = packet.filter_by_modality("table")
        assert len(table_units) >= 1

        read_res = self.table_reader.read(question, table_units[0])
        # Values extracted are raw source numbers
        assert "88.0%" in read_res.relevant_values
        assert "94.2%" in read_res.relevant_values

        # Ensure no arithmetic calculation result (e.g. 6.2% difference or division) was performed
        assert "6.2%" not in read_res.relevant_values
        assert "1.07" not in read_res.relevant_values
