"""
Layout & Text Extraction Module for MDI
Uses PyMuPDF (fitz) to extract text page-by-page and chunk into blocks with provenance.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from backend.app.ingest.sections import SectionTracker

def extract_text_and_blocks_from_pdf(
    pdf_path: str, 
    doc_id: str, 
    doc_title: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract text page by page from PDF using PyMuPDF.
    
    Returns:
        (pages_summary, text_units)
    """
    doc = fitz.open(pdf_path)
    pages_summary = []
    text_units = []
    
    section_tracker = SectionTracker()
    
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1  # 1-based page numbering
        raw_text = page.get_text("text")
        
        # Save page summary information
        pages_summary.append({
            "page": page_num,
            "text_content": raw_text
        })
        
        # Extract blocks with coordinates/formatting
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        
        text_idx = 1
        for b in blocks:
            # Check if block is text (type 0)
            if len(b) >= 7 and b[6] == 0:
                block_text = b[4].strip()
                if not block_text:
                    continue
                
                # Update section tracker
                lines = block_text.split("\n")
                first_line = lines[0]
                section_path = section_tracker.process_text_line(first_line)
                
                element_id = f"{doc_id}-p{page_num}-text-{text_idx:02d}"
                text_idx += 1
                
                text_units.append({
                    "doc_id": doc_id,
                    "document_title": doc_title,
                    "page": page_num,
                    "page_label": f"Page {page_num}",
                    "section_path": section_path,
                    "element_type": "text",
                    "element_id": element_id,
                    "content": block_text,
                    "table_json": None,
                    "table_markdown": None
                })
                
    doc.close()
    return pages_summary, text_units
