"""
Provenance Engine & Evidence Schema Definition
Ensures every evidence unit maintains immutable provenance fields.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field

class EvidenceUnitSchema(BaseModel):
    doc_id: str
    document_title: str
    page: int
    page_label: str
    section_path: str
    element_type: str = Field(description="'text' or 'table'")
    element_id: str
    content: str
    table_json: str | None = None
    table_markdown: str | None = None

def create_provenance_record(
    doc_id: str,
    document_title: str,
    page: int,
    section_path: str,
    element_type: str,
    element_id: str,
    content: str,
    table_json: str | None = None,
    table_markdown: str | None = None
) -> Dict[str, Any]:
    """
    Constructs a standardized evidence dictionary guaranteeing exact provenance structure.
    """
    page_label = f"Page {page}"
    sec_path = section_path if section_path and section_path.strip() else "Unknown Section"
    
    evidence = EvidenceUnitSchema(
        doc_id=doc_id,
        document_title=document_title,
        page=page,
        page_label=page_label,
        section_path=sec_path,
        element_type=element_type,
        element_id=element_id,
        content=content.strip(),
        table_json=table_json,
        table_markdown=table_markdown
    )
    
    return evidence.model_dump()
