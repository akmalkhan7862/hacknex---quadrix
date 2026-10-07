"""
FastAPI Request & Response Schemas for VERITAS (Project HNX26PSI01)
Pydantic v2 schemas with comprehensive OpenAPI examples for all endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IngestionJobResponse(BaseModel):
    document_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff"})
    job_id: str = Field(..., json_schema_extra={"example": "job-89a12c4b"})
    status: str = Field(..., json_schema_extra={"example": "PROCESSING"})
    message: str = Field(..., json_schema_extra={"example": "Document ingestion job queued successfully."})

class IngestionStatusResponse(BaseModel):
    document_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff"})
    stage: str = Field(..., json_schema_extra={"example": "TABLE_EXTRACTION"})
    progress_pct: float = Field(..., json_schema_extra={"example": 85.0})
    page_count: int = Field(..., json_schema_extra={"example": 22})
    evidence_units_count: int = Field(..., json_schema_extra={"example": 45})
    errors: List[str] = Field(default=[], json_schema_extra={"example": []})

class BoundingBox(BaseModel):
    ymin: float = Field(..., json_schema_extra={"example": 100.0})
    xmin: float = Field(..., json_schema_extra={"example": 50.0})
    ymax: float = Field(..., json_schema_extra={"example": 300.0})
    xmax: float = Field(..., json_schema_extra={"example": 550.0})

class Citation(BaseModel):
    document_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff"})
    document_title: str = Field(..., json_schema_extra={"example": "doc-q2-eff.pdf"})
    page: int = Field(..., json_schema_extra={"example": 1})
    section_path: str = Field(..., json_schema_extra={"example": "2. Production Output & Efficiency Metrics"})
    element_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff-p1-table-01"})
    element_type: str = Field(..., json_schema_extra={"example": "table"})
    bbox: List[float] = Field(..., json_schema_extra={"example": [160.0, 50.0, 320.0, 550.0]})
    snippet: str = Field(..., json_schema_extra={"example": "Line Alpha Actual Output: 1450 Metric Tons"})

class PageOverlayResponse(BaseModel):
    document_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff"})
    page_number: int = Field(..., json_schema_extra={"example": 1})
    width: int = Field(..., json_schema_extra={"example": 918})
    height: int = Field(..., json_schema_extra={"example": 1188})
    bboxes: List[Citation]

class QueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "Compare the Line Alpha Actual Output and primary downtime cause between Q2 and Q4."})
    document_ids: Optional[List[str]] = Field(default=None, json_schema_extra={"example": ["doc-q2-eff", "doc-q4-eff"]})
    top_k: Optional[int] = Field(default=5, json_schema_extra={"example": 5})

class QueryResponse(BaseModel):
    answer: str = Field(..., json_schema_extra={"example": "In Q2, Line Alpha Actual Output was 1450 Metric Tons (downtime: Pump-04 hydraulic failure); in Q4, Line Alpha Actual Output was 1620 Short Tons (downtime: raw titanium alloy shortage)."})
    confidence: float = Field(..., json_schema_extra={"example": 0.96})
    confidence_badge: str = Field(..., json_schema_extra={"example": "HIGH"})
    citations: List[Citation]
    calculation_trace: List[Dict[str, Any]] = Field(default=[], json_schema_extra={"example": [{"step": 1, "formula": "1620 Short Tons - 1450 Metric Tons", "note": "Unit conversion required before arithmetic"}]})
    reasoning_plan: List[str] = Field(default=[], json_schema_extra={"example": ["Extracted text & table units from doc-q2-eff and doc-q4-eff", "Detected unit mismatch (Metric Tons vs Short Tons)", "Verified citations via anti-hallucination verifier gate"]})
    limitations: List[str] = Field(default=[], json_schema_extra={"example": ["Document doc-q4-eff uses Short Tons whereas doc-q2-eff uses Metric Tons."]})\

class EvidenceUnitResponse(BaseModel):
    element_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff-p1-table-01"})
    doc_id: str = Field(..., json_schema_extra={"example": "doc-q2-eff"})
    document_title: str = Field(..., json_schema_extra={"example": "doc-q2-eff.pdf"})
    page: int = Field(..., json_schema_extra={"example": 1})
    section_path: str = Field(..., json_schema_extra={"example": "2. Production Output & Efficiency Metrics"})
    element_type: str = Field(..., json_schema_extra={"example": "table"})
    content: str
    bbox: List[float] = Field(..., json_schema_extra={"example": [160.0, 50.0, 320.0, 550.0]})
    crop_url: str = Field(..., json_schema_extra={"example": "/documents/doc-q2-eff/pages/1"})

class EvalRunResponse(BaseModel):
    run_id: str = Field(..., json_schema_extra={"example": "run-20261007-001"})
    status: str = Field(..., json_schema_extra={"example": "RUNNING"})
    message: str = Field(..., json_schema_extra={"example": "Evaluation benchmark started over 220 questions."})
