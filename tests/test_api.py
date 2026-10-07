"""
Pytest Suite for VERITAS FastAPI Endpoints (Specs Section 10, 13, 14, 17)
Includes tests for 405 bugfix, page overlays, query RAG, evidence unit crops, and eval runs.
"""

import pytest
import httpx
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_documents_ingest_no_trailing_slash():
    """Verify POST /documents returns 202 with document_id and job_id."""
    response = client.post(
        "/documents",
        files={"file": ("test_doc.pdf", b"%PDF-1.4 test document content", "application/pdf")}
    )
    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert "job_id" in data
    assert data["status"] == "PROCESSING"

def test_documents_ingest_with_trailing_slash_bugfix():
    """Verify POST /documents/ returns 202 (fixing 405 Method Not Allowed error)."""
    response = client.post(
        "/documents/",
        files={"file": ("test_doc.pdf", b"%PDF-1.4 test document content", "application/pdf")}
    )
    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert "job_id" in data

def test_documents_status_endpoint():
    response = client.get("/documents/doc-q2-eff/status")
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "stage" in data
    assert "progress_pct" in data

def test_page_overlay_endpoint():
    response = client.get("/documents/doc-q2-eff/pages/1?format=json")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-q2-eff"
    assert data["page_number"] == 1
    assert "bboxes" in data

def test_query_endpoint():
    payload = {
        "question": "Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4.",
        "document_ids": ["doc-q2-eff", "doc-q4-eff"],
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data

def test_evidence_unit_endpoint():
    response = client.get("/evidence/doc-q2-eff-p1-table-01")
    assert response.status_code == 200
    data = response.json()
    assert data["element_id"] == "doc-q2-eff-p1-table-01"

def test_eval_run_endpoint():
    response = client.post("/eval/run")
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
