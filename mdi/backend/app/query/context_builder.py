"""
Context Builder Module for MDI
Formats top reranked evidence objects into structured LLM context blocks.
Strictly sends ONLY retrieved evidence to the LLM.
"""

from typing import List, Dict, Any

def build_context(reranked_evidence: List[Dict[str, Any]]) -> str:
    """
    Formats evidence items into exact structured context string.
    """
    if not reranked_evidence:
        return "NO_RELEVANT_EVIDENCE_FOUND"
    
    blocks = []
    for item in reranked_evidence:
        block = (
            f"DOCUMENT: {item.get('document_title', 'Unknown')}\n"
            f"PAGE: {item.get('page', 0)}\n"
            f"SECTION: {item.get('section_path', 'Unknown Section')}\n"
            f"TYPE: {item.get('element_type', 'text')}\n"
            f"EVIDENCE_ID: {item.get('element_id', '')}\n\n"
            f"{item.get('content', '')}"
        )
        blocks.append(block)
        
    return "\n\n---\n\n".join(blocks)
