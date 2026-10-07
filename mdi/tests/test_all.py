"""
Comprehensive Unit & Integration Test Suite for MDI Architecture
"""

import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.ingest.classify import classify_page
from backend.app.ingest.sections import SectionTracker, extract_section_path
from backend.app.ingest.provenance import create_provenance_record
from backend.app.index.bm25 import BM25Index
from backend.app.index.vector_store import VectorStore
from backend.app.query.analyzer import analyze_query
from backend.app.query.router import global_router
from backend.app.query.retriever import hybrid_retrieve
from backend.app.query.reranker import rerank_evidence
from backend.app.query.context_builder import build_context
from backend.app.reasoning.llm import generate_llm_answer
from backend.app.evidence.validator import validate_claims_and_evidence
from backend.app.evidence.engine import global_evidence_engine

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_page_classification():
    text_digital = "This is a digital page containing ample text content for extraction and analysis."
    c1 = classify_page(text_digital, page_num=1)
    assert c1["page"] == 1
    assert c1["page_type"] == "digital"
    assert c1["quality_score"] > 0.5

    text_scanned = ""
    c2 = classify_page(text_scanned, page_num=2)
    assert c2["page"] == 2
    assert c2["page_type"] == "scanned"
    assert c2["quality_score"] == 0.2

def test_section_detection():
    tracker = SectionTracker()
    sec1 = tracker.process_text_line("1. Executive Summary")
    assert sec1 == "1. Executive Summary"
    
    sec2 = tracker.process_text_line("1.1 Background & Context")
    assert sec2 == "1. Executive Summary > 1.1 Background & Context"
    
    sec3 = tracker.process_text_line("Regular text line here.")
    assert sec3 == "1. Executive Summary > 1.1 Background & Context"

def test_provenance_record():
    record = create_provenance_record(
        doc_id="test123",
        document_title="Sample.pdf",
        page=5,
        section_path="1. Overview",
        element_type="text",
        element_id="test123-p5-text-01",
        content="Provenance text content test."
    )
    assert record["doc_id"] == "test123"
    assert record["page"] == 5
    assert record["page_label"] == "Page 5"
    assert record["section_path"] == "1. Overview"
    assert record["element_id"] == "test123-p5-text-01"

def test_bm25_retrieval():
    index = BM25Index()
    docs = [
        {"element_id": "id1", "content": "Operating expenses decreased by 15% in Q3."},
        {"element_id": "id2", "content": "Total revenue reached $5 million."}
    ]
    index.build_index(docs)
    results = index.search("expenses decreased", top_k=1)
    assert len(results) >= 1
    assert results[0]["element_id"] == "id1"

def test_vector_store():
    store = VectorStore()
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    p1 = {"element_id": "vec1", "content": "First item"}
    p2 = {"element_id": "vec2", "content": "Second item"}
    store.add_item(v1, p1)
    store.add_item(v2, p2)
    
    res = store.search([0.9, 0.1, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0]["element_id"] == "vec1"

def test_query_analyzer():
    res = analyze_query("What is the total revenue in the table?")
    assert res["needs_text"] is True
    assert res["needs_table"] is True
    assert "revenue" in res["keywords"]

def test_query_router():
    # A bare minimal dict with no table/chart/calc signals → VECTOR route.
    engines = global_router.route_query({"question": "sample"})
    assert "vector" in engines

    # A table-bearing query → HYBRID (both vector + bm25).
    table_engines = global_router.route_query({
        "question": "What was the revenue?",
        "modalities": ["text", "table"],
        "needs_calculation": False,
        "cross_document": False,
        "intent": "fact_lookup",
    })
    assert "vector" in table_engines
    assert "bm25" in table_engines

def test_reranker():
    candidates = [
        {"element_id": "e1", "vector_score": 0.9, "bm25_score": 0.1},
        {"element_id": "e2", "vector_score": 0.2, "bm25_score": 0.8}
    ]
    reranked = rerank_evidence(candidates, top_n=2)
    assert len(reranked) == 2
    assert "score" in reranked[0]

def test_context_builder():
    evidence = [
        {
            "document_title": "Report.pdf",
            "page": 1,
            "section_path": "Summary",
            "element_type": "text",
            "element_id": "e1",
            "content": "Sample content."
        }
    ]
    ctx = build_context(evidence)
    assert "DOCUMENT: Report.pdf" in ctx
    assert "PAGE: 1" in ctx
    assert "EVIDENCE_ID: e1" in ctx

def test_evidence_validator_and_not_found():
    llm_out = {"answer": "Sample answer", "claims": [{"text": "Claim 1", "evidence_ids": ["non_existent_id"]}]}
    retrieved = [{"element_id": "valid_id", "content": "Content"}]
    
    ans, claims, ids = validate_claims_and_evidence(llm_out, retrieved)
    assert ans == "Not found in the provided documents."
    assert len(claims) == 0

def test_insufficient_evidence_behavior():
    res = global_evidence_engine.process(
        {"answer": "Not found in the provided documents.", "claims": []},
        []
    )
    assert res["answer"] == "Not found in the provided documents."
    assert res["claims"] == []
    assert res["sources"] == []
