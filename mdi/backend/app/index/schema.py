"""
Indexing Schema Definitions for MDI
"""

from typing import Dict, Any, List
from pydantic import BaseModel

class IndexedItem(BaseModel):
    element_id: str
    doc_id: str
    document_title: str
    page: int
    page_label: str
    section_path: str
    element_type: str
    content: str
    vector: List[float] | None = None
