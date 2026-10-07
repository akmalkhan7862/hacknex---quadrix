"""
Phase M Unit Tests — Graph RAG Interface & Extension Point

Covers:
  - GraphRetriever extension point interface
  - Unconfigured behavior: returns empty list without raising exceptions
  - status() method reporting
  - Mock edge configuration and entity-linked EvidenceUnit retrieval
  - Provenance integrity on graph evidence units
  - Integration with router.py's GraphRAGExtensionPoint
  - Non-breaking behavior in cross-document routing
"""

import pytest

from backend.app.query.graph import GraphRetriever, global_graph_retriever
from backend.app.query.router import GraphRAGExtensionPoint, QueryRouter
from backend.app.query.schemas import EvidenceUnit


class TestGraphRetrieverExtensionPoint:

    def test_unconfigured_returns_empty_list(self):
        retriever = GraphRetriever(enabled=False)
        assert retriever.retrieve("Compare Annual Report and Operations Report") == []
        assert retriever.retrieve_raw("Compare Annual Report and Operations Report") == []

    def test_status_unconfigured(self):
        retriever = GraphRetriever(enabled=False)
        assert retriever.status() == "unconfigured"

    def test_configured_mock_graph_retrieval(self):
        retriever = GraphRetriever()
        retriever.add_mock_edge(
            source_entity="Dallas Plant",
            target_entity="Production Efficiency",
            relation="exhibits",
            doc_id="doc1",
            evidence_content="Dallas Plant reached 94.2% efficiency in Q4.",
            page=3,
        )

        assert "configured_mock_graph" in retriever.status()

        units = retriever.retrieve("What was the Dallas Plant production efficiency?", top_k=2)
        assert len(units) == 1
        assert isinstance(units[0], EvidenceUnit)
        assert units[0].element_id == "graph-edge-1"
        assert units[0].doc_id == "doc1"
        assert units[0].page == 3
        assert "Dallas Plant" in units[0].content
        assert "Production Efficiency" in units[0].content
        assert units[0].score is not None

    def test_filtering_by_document_id(self):
        retriever = GraphRetriever()
        retriever.add_mock_edge(
            source_entity="A", target_entity="B", relation="rel", doc_id="doc1", evidence_content="C1"
        )
        retriever.add_mock_edge(
            source_entity="A", target_entity="B", relation="rel", doc_id="doc2", evidence_content="C2"
        )

        units = retriever.retrieve("A B", filters={"document_id": "doc2"})
        assert len(units) == 1
        assert units[0].doc_id == "doc2"

    def test_global_singleton_available(self):
        assert global_graph_retriever is not None
        assert isinstance(global_graph_retriever, GraphRetriever)


class TestRouterGraphIntegration:

    def test_router_extension_point_unconfigured(self):
        ext = GraphRAGExtensionPoint()
        res = ext.retrieve({"question": "Compare Annual and Operations reports"})
        assert isinstance(res, list)
        assert len(res) == 0

    def test_router_extension_point_with_mock_graph(self):
        retriever = GraphRetriever()
        retriever.add_mock_edge(
            source_entity="Q4",
            target_entity="Revenue",
            relation="generated",
            doc_id="doc_fin",
            evidence_content="Generated $10M in revenue.",
        )
        ext = GraphRAGExtensionPoint(retriever=retriever)
        res = ext.retrieve({"question": "Q4 Revenue"})
        assert len(res) == 1
        assert res[0]["doc_id"] == "doc_fin"

    def test_query_router_does_not_break(self):
        router = QueryRouter()
        engines = router.route_query({
            "question": "Compare the annual report and operations report across documents",
            "cross_document": True
        })
        assert "graph" in engines
