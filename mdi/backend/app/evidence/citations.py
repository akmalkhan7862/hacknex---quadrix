"""
Citations Module for MDI
Formats evidence sources for display in API response and Frontend.
"""

from typing import Dict, Any, List

def format_citations(evidence_items: List[Dict[str, Any]], referenced_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Filter retrieved evidence items to those referenced by claims, and format sources.
    """
    sources = []
    seen_ids = set()
    
    for item in evidence_items:
        elem_id = item.get("element_id")
        if not elem_id or elem_id in seen_ids:
            continue
            
        if not referenced_ids or elem_id in referenced_ids:
            seen_ids.add(elem_id)
            sources.append({
                "document": item.get("document_title", "Unknown Document"),
                "page": item.get("page", 0),
                "section": item.get("section_path", "Unknown Section"),
                "content_type": item.get("element_type", "text"),
                "element_id": elem_id,
                "snippet": item.get("content", "")[:200]
            })
            
    return sources
