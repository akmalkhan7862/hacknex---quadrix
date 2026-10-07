"""
Shared Schema Contracts for Multimodal Document Intelligence (MDI)
DO NOT CHANGE THESE CONTRACTS.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IndexedItem(BaseModel):
    element_id: str
    doc_id: str
    document_title: str
    page: int
    page_label: str
    section_path: str
    element_type: str = Field(description="'text', 'table', 'chart', or 'image'")
    content: str
    vector: Optional[List[float]] = None
    bbox: Optional[List[float]] = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized 0..1000")
    metadata: Optional[Dict[str, Any]] = None

class GroundTruthAnnotation(BaseModel):
    doc_id: str
    document_title: str
    page: int
    section_path: str
    element_id: str
    element_type: str
    value: Any
    unit: Optional[str] = None
    bbox: Optional[List[float]] = None

class QuestionGroundTruth(BaseModel):
    question_id: str
    question: str
    required_modality: str  # text, table, chart, image, mixed, cross-document, math, unanswerable
    answer: str
    gold_doc_ids: List[str]
    gold_pages: List[int]
    gold_sections: List[str]
    gold_element_ids: List[str]
    is_answerable: bool = True
