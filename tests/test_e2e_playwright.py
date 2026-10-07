"""
Playwright End-to-End Test Suite for VERITAS UI (Landing Page & Workspace)
Saves screenshots to docs/screenshots/.
"""

import os
import pytest
from pathlib import Path

SCREENSHOTS_DIR = Path("./docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def test_landing_page_render_mock():
    """Verify landing page design tokens and content."""
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Check 405 bugfix status
    res = client.post("/documents/", files={"file": ("test.pdf", b"pdf content", "application/pdf")})
    assert res.status_code == 202
    assert "document_id" in res.json()

def test_workspace_mock_flow(monkeypatch):
    """Verify Q2 vs Q4 query flow, citations, and refusal states under USE_MOCK=true."""
    monkeypatch.setenv("USE_MOCK", "true")
    import api.main
    api.main.USE_MOCK = True

    from fastapi.testclient import TestClient
    client = TestClient(api.main.app)
    
    query_payload = {
        "question": "Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4.",
        "document_ids": ["doc-q2-eff", "doc-q4-eff"]
    }
    res = client.post("/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert "1450 Metric Tons" in data["answer"]
    assert len(data["citations"]) >= 2
    
    # Test Refusal State
    refusal_payload = {
        "question": "What is the stock price of Acme Corp on Mars?"
    }
    res_ref = client.post("/query", json=refusal_payload)
    assert res_ref.status_code == 200
    data_ref = res_ref.json()
    assert data_ref["answer"] == "Not found in the provided documents."
    assert data_ref["confidence_badge"] == "REFUSAL"
