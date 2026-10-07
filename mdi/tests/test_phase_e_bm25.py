"""
Phase E Unit Tests — BM25 Keyword Retrieval

Covers:
  - BM25Index: build_index, add_items, clear
  - BM25Index: exact keyword matches ("Q4", "production", "efficiency")
  - BM25Index: get_by_id O(1) resolution
  - BM25Index: search_ids returns (element_id, score) pairs without duplicating units
  - BM25Index: metadata filtering (document_id, content_type, section, period, page)
  - BM25Retriever: typed retrieve() returning EvidenceUnit with complete provenance
  - BM25Retriever: retrieve_ids() reference retrieval
  - BM25Retriever: retrieve_raw() returning dicts
  - BM25Retriever: dependency injection of custom index
  - BM25Retriever: empty query and empty index edge cases
  - Backward compatibility: global_bm25_index and existing search()
"""

import pytest
from typing import List, Dict, Any

from backend.app.index.bm25 import BM25Index, global_bm25_index
from backend.app.query.retriever import BM25Retriever, global_bm25_retriever
from backend.app.query.schemas import EvidenceUnit


def _sample_docs() -> List[Dict[str, Any]]:
    return [
        {
            "element_id": "doc1-p1-text-01",
            "doc_id": "doc1",
            "document_title": "Annual Report 2023",
            "page": 1,
            "page_label": "Page 1",
            "section_path": "Executive Summary",
            "element_type": "text",
            "content": "Overall corporate revenue grew while production efficiency remained steady.",
            "table_json": None,
            "table_markdown": None,
        },
        {
            "element_id": "doc1-p4-table-01",
            "doc_id": "doc1",
            "document_title": "Annual Report 2023",
            "page": 4,
            "page_label": "Page 4",
            "section_path": "Operations > Metrics",
            "element_type": "table",
            "content": "| Quarter | Production Efficiency | Output |\n| Q4 | 94.2% | 12000 |",
            "table_json": '[["Quarter", "Production Efficiency", "Output"], ["Q4", "94.2%", "12000"]]',
            "table_markdown": "| Quarter | Production Efficiency | Output |\n| Q4 | 94.2% | 12000 |",
        },
        {
            "element_id": "doc2-p2-text-01",
            "doc_id": "doc2",
            "document_title": "Operations Report",
            "page": 2,
            "page_label": "Page 2",
            "section_path": "Factory Performance",
            "element_type": "text",
            "content": "In Q2, factory throughput increased and operating costs decreased by 15%.",
            "table_json": None,
            "table_markdown": None,
        },
    ]


class TestBM25IndexCore:

    def test_build_and_search_exact_keywords(self):
        index = BM25Index()
        index.build_index(_sample_docs())

        # Exact match for Q4 and production efficiency
        results = index.search("Q4 production efficiency", top_k=2)
        assert len(results) >= 1
        top = results[0]
        assert top["element_id"] == "doc1-p4-table-01"
        assert "Q4" in top["content"]
        assert "Production Efficiency" in top["content"]
        assert top.get("bm25_score", 0.0) > 0.0

    def test_search_individual_keywords(self):
        index = BM25Index()
        index.build_index(_sample_docs())

        # Match Q2
        res_q2 = index.search("Q2 operating costs", top_k=1)
        assert len(res_q2) == 1
        assert res_q2[0]["element_id"] == "doc2-p2-text-01"

    def test_get_by_id_resolution(self):
        index = BM25Index()
        index.build_index(_sample_docs())

        doc = index.get_by_id("doc1-p4-table-01")
        assert doc is not None
        assert doc["doc_id"] == "doc1"
        assert doc["page"] == 4

        missing = index.get_by_id("non-existent-id")
        assert missing is None

    def test_search_ids_returns_references(self):
        index = BM25Index()
        index.build_index(_sample_docs())

        id_results = index.search_ids("Q4 production efficiency", top_k=2)
        assert isinstance(id_results, list)
        assert len(id_results) >= 1
        elem_id, score = id_results[0]
        assert elem_id == "doc1-p4-table-01"
        assert isinstance(score, float)
        assert score > 0.0

    def test_add_items_incremental(self):
        index = BM25Index()
        docs = _sample_docs()
        index.build_index(docs[:2])
        assert len(index.documents) == 2

        index.add_items([docs[2]])
        assert len(index.documents) == 3
        assert index.get_by_id("doc2-p2-text-01") is not None

    def test_clear_index(self):
        index = BM25Index()
        index.build_index(_sample_docs())
        assert len(index.documents) == 3

        index.clear()
        assert len(index.documents) == 0
        assert index.search("revenue") == []
        assert index.get_by_id("doc1-p1-text-01") is None

    def test_empty_query_or_empty_index(self):
        index = BM25Index()
        assert index.search("revenue") == []
        assert index.search_ids("revenue") == []

        index.build_index(_sample_docs())
        assert index.search("") == []
        assert index.search("   ") == []


class TestBM25IndexFilters:

    def setup_method(self):
        self.index = BM25Index()
        self.index.build_index(_sample_docs())

    def test_filter_by_document_id(self):
        results = self.index.search("revenue", top_k=5, filters={"document_id": "doc1"})
        assert all(r["doc_id"] == "doc1" for r in results)

    def test_filter_by_content_type(self):
        results = self.index.search("production", top_k=5, filters={"content_type": "table"})
        assert all(r["element_type"] == "table" for r in results)
        assert len(results) == 1
        assert results[0]["element_id"] == "doc1-p4-table-01"

    def test_filter_by_section(self):
        results = self.index.search("throughput", top_k=5, filters={"section": "Factory Performance"})
        assert len(results) == 1
        assert results[0]["element_id"] == "doc2-p2-text-01"

    def test_filter_by_period(self):
        results = self.index.search("efficiency", top_k=5, filters={"period": "Q4"})
        assert len(results) == 1
        assert "Q4" in results[0]["content"].upper()

    def test_filter_by_page(self):
        results = self.index.search("efficiency", top_k=5, filters={"page": 4})
        assert len(results) == 1
        assert results[0]["page"] == 4


class TestBM25Retriever:

    def setup_method(self):
        self.index = BM25Index()
        self.index.build_index(_sample_docs())
        self.retriever = BM25Retriever(index=self.index)

    def test_retrieve_returns_evidence_units(self):
        units = self.retriever.retrieve("Q4 production efficiency", top_k=2)
        assert len(units) >= 1
        top = units[0]
        assert isinstance(top, EvidenceUnit)
        assert top.element_id == "doc1-p4-table-01"
        assert top.doc_id == "doc1"
        assert top.document_title == "Annual Report 2023"
        assert top.page == 4
        assert top.element_type == "table"
        assert top.score is not None
        assert top.score > 0.0

    def test_retrieve_ids_returns_tuples(self):
        refs = self.retriever.retrieve_ids("Q2 operating costs", top_k=2)
        assert len(refs) >= 1
        elem_id, score = refs[0]
        assert elem_id == "doc2-p2-text-01"
        assert score > 0.0

    def test_retrieve_raw_returns_dicts(self):
        raw = self.retriever.retrieve_raw("operating costs", top_k=1)
        assert len(raw) == 1
        assert isinstance(raw[0], dict)
        assert raw[0]["element_id"] == "doc2-p2-text-01"
        assert "bm25_score" in raw[0]

    def test_retrieve_with_filters(self):
        units = self.retriever.retrieve(
            "efficiency",
            top_k=5,
            filters={"content_type": "text"}
        )
        assert all(u.element_type == "text" for u in units)

    def test_empty_query(self):
        assert self.retriever.retrieve("") == []
        assert self.retriever.retrieve_ids("") == []
        assert self.retriever.retrieve_raw("") == []

    def test_global_bm25_retriever_singleton(self):
        assert global_bm25_retriever is not None
        assert isinstance(global_bm25_retriever, BM25Retriever)
