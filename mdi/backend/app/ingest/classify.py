"""
Page Classification Module for MDI
Classifies each document page into digital/text-based vs scanned/image-based.
Computes quality score without inventing page numbers.
"""

from typing import Dict, Any

def classify_page(text: str, page_num: int) -> Dict[str, Any]:
    """
    Classify a page based on extracted text length and character density.
    
    Args:
        text: Extracted raw text content of the page
        page_num: 1-based page number
        
    Returns:
        Dict with keys: page, page_type ('digital' or 'scanned'), quality_score (0.0 - 1.0)
    """
    clean_text = text.strip() if text else ""
    
    # If text is extremely short or empty, mark as scanned/image-based
    if len(clean_text) < 20:
        return {
            "page": page_num,
            "page_type": "scanned",
            "quality_score": 0.2
        }
    
    # Calculate basic textual quality score based on readable ASCII / printable characters
    printable_chars = sum(1 for c in clean_text if c.isprintable())
    density = printable_chars / len(clean_text) if clean_text else 0
    
    return {
        "page": page_num,
        "page_type": "digital",
        "quality_score": round(max(0.5, density), 2)
    }
