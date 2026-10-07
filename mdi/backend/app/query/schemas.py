"""
Shared Contract & Schema Definitions for Adaptive Multimodal Retrieval in MDI.

Integrates with existing EvidenceUnitSchema while providing structured definitions for:
- RetrievalStrategy
- Plan
- EvidenceUnit
- Provenance
- EvidencePacket
- Citation
- AnswerJSON
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.app.ingest.provenance import EvidenceUnitSchema


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies chosen by the Adaptive Router."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"
    CROSS_DOCUMENT = "cross_document"


class Plan(BaseModel):
    """
    Structured query plan produced by the Query Analyzer.
    """
    question: str
    intent: str = "fact_lookup"  # comparison, aggregation, fact_lookup, visual_analysis, cross_doc_synthesis, calculation
    entities: List[str] = Field(default_factory=list)
    periods: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=lambda: ["text"])  # text, table, chart, image
    needs_calculation: bool = False
    cross_document: bool = False
    query_type: str = "factual"  # factual, analytical, comparative, visual
    filters: Dict[str, Any] = Field(default_factory=dict)
    sub_questions: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Explicit source tracking provenance model."""
    doc_id: str
    document_title: str
    page: int
    page_label: str
    section_path: str
    element_id: str


class EvidenceUnit(EvidenceUnitSchema):
    """
    Unified Evidence Unit adhering strictly to the team's EvidenceUnitSchema contract,
    with backwards-compatible extensions for multimodal retrieval.
    """
    element_type: str = Field(default="text", description="'text', 'table', 'chart', or 'image'")
    visual_ref: Optional[str] = Field(default=None, description="Path or URI to image/crop")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = Field(default=None, description="Retrieval / reranker confidence score")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceUnit":
        """Factory method to construct an EvidenceUnit from raw dict or database row."""
        return cls(
            doc_id=data.get("doc_id", "unknown_doc"),
            document_title=data.get("document_title", "Unknown Document"),
            page=data.get("page", 1),
            page_label=data.get("page_label", f"Page {data.get('page', 1)}"),
            section_path=data.get("section_path", "Unknown Section"),
            element_type=data.get("element_type", "text"),
            element_id=data.get("element_id", "unknown_id"),
            content=data.get("content", ""),
            table_json=data.get("table_json"),
            table_markdown=data.get("table_markdown"),
            visual_ref=data.get("visual_ref") or data.get("image_path") or data.get("crop_path"),
            metadata=data.get("metadata", {}),
            score=data.get("score") or data.get("vector_score") or data.get("bm25_score")
        )

    def to_provenance(self) -> Provenance:
        """Extract explicit provenance metadata."""
        return Provenance(
            doc_id=self.doc_id,
            document_title=self.document_title,
            page=self.page,
            page_label=self.page_label,
            section_path=self.section_path,
            element_id=self.element_id
        )


class EvidencePacket(BaseModel):
    """
    Standard output contract produced by Multimodal Retrieval for the Reasoning Engine.
    """
    question: str
    plan: Optional[Plan] = None
    strategy: Optional[RetrievalStrategy] = None
    evidence_units: List[EvidenceUnit] = Field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_unit_by_id(self, element_id: str) -> Optional[EvidenceUnit]:
        """Lookup an evidence unit by its element ID."""
        for u in self.evidence_units:
            if u.element_id == element_id:
                return u
        return None

    def filter_by_modality(self, modality: str) -> List[EvidenceUnit]:
        """Return evidence units matching a specific modality (text, table, chart, image)."""
        return [u for u in self.evidence_units if u.element_type == modality]


class Citation(BaseModel):
    """Formatted citation for evidence presentation."""
    document: str
    page: int
    section: str
    content_type: str
    element_id: str
    snippet: str


class Claim(BaseModel):
    """Factual claim mapped to supporting evidence IDs."""
    text: str
    evidence_ids: List[str] = Field(default_factory=list)


class AnswerJSON(BaseModel):
    """Target output schema expected by downstream engine and API consumers."""
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    sources: List[Citation] = Field(default_factory=list)
    calculations: List[Dict[str, Any]] = Field(default_factory=list)
