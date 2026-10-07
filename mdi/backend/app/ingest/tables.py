"""
Table Extraction Module for MDI
Uses pdfplumber to extract structured tables page-by-page.
Converts extracted tables to structured JSON and Markdown format.
"""

import json
from typing import List, Dict, Any, Optional
import pdfplumber

def table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """
    Convert a 2D list table into a markdown table string.
    """
    if not table or not any(table):
        return ""
    
    # Filter out completely empty rows
    clean_rows = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cleaned_row):
            clean_rows.append(cleaned_row)
            
    if not clean_rows:
        return ""
    
    # Determine column width based on max columns in any row
    num_cols = max(len(r) for r in clean_rows)
    
    # Pad rows to have uniform column length
    padded_rows = [r + [""] * (num_cols - len(r)) for r in clean_rows]
    
    headers = padded_rows[0]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * num_cols) + " |"
    
    body_lines = []
    for row in padded_rows[1:]:
        body_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join([header_line, separator_line] + body_lines)

def extract_tables_from_pdf_page(pdf_path: str, page_num: int, section_path: str, doc_id: str, doc_title: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a single page of a PDF using pdfplumber.
    page_num is 1-indexed.
    """
    tables_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return []
            
            page = pdf.pages[page_num - 1]
            extracted_tables = page.extract_tables()
            
            for idx, table in enumerate(extracted_tables):
                if not table:
                    continue
                
                md_table = table_to_markdown(table)
                if not md_table.strip():
                    continue
                
                table_id = f"{doc_id}-p{page_num}-table-{idx+1:02d}"
                
                tables_data.append({
                    "doc_id": doc_id,
                    "document_title": doc_title,
                    "page": page_num,
                    "page_label": f"Page {page_num}",
                    "section_path": section_path,
                    "element_type": "table",
                    "element_id": table_id,
                    "content": md_table,
                    "table_json": json.dumps(table),
                    "table_markdown": md_table
                })
    except Exception as e:
        print(f"[tables.py] Warning extracting tables from {pdf_path} page {page_num}: {e}")
        
    return tables_data
