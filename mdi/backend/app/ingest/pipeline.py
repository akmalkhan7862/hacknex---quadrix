"""
Ingestion Pipeline Coordinator for MDI
Orchestrates PDF page classification, text extraction, table extraction, section detection,
provenance assignment, SQLite database persisting, and vector + BM25 indexing.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

from backend.app.db import get_db
from backend.app.ingest.classify import classify_page
from backend.app.ingest.layout import extract_text_and_blocks_from_pdf
from backend.app.ingest.tables import extract_tables_from_pdf_page
from backend.app.ingest.provenance import create_provenance_record
from backend.app.index.embeddings import generate_embeddings
from backend.app.index.vector_store import global_vector_store
from backend.app.index.bm25 import global_bm25_index

def run_ingestion_pipeline(doc_id: str, file_path: str, document_title: str) -> Dict[str, Any]:
    """
    Execute full ingestion pipeline for an uploaded PDF file.
    """
    db = get_db()
    cursor = db.cursor()
    
    try:
        # 1. Update status to PROCESSING
        cursor.execute("UPDATE documents SET status = 'PROCESSING' WHERE doc_id = ?", (doc_id,))
        db.commit()
        
        # 2. Extract text & layout blocks using PyMuPDF
        pages_summary, text_units = extract_text_and_blocks_from_pdf(file_path, doc_id, document_title)
        page_count = len(pages_summary)
        
        # 3. Classify each page and save to pages table
        for page_data in pages_summary:
            p_num = page_data["page"]
            p_text = page_data["text_content"]
            classification = classify_page(p_text, p_num)
            
            cursor.execute(
                """
                INSERT INTO pages (doc_id, page, page_type, quality_score, text_content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, p_num, classification["page_type"], classification["quality_score"], p_text)
            )
            
        # 4. Extract tables page-by-page using pdfplumber
        table_units = []
        for page_data in pages_summary:
            p_num = page_data["page"]
            # Find section path for page if text units exist on that page
            page_text_units = [u for u in text_units if u["page"] == p_num]
            sec_path = page_text_units[0]["section_path"] if page_text_units else "Unknown Section"
            
            extracted_tables = extract_tables_from_pdf_page(file_path, p_num, sec_path, doc_id, document_title)
            table_units.extend(extracted_tables)
            
        # 5. Combine text & table units and build standardized provenance records
        all_evidence_units: List[Dict[str, Any]] = []
        
        for unit in text_units:
            rec = create_provenance_record(
                doc_id=unit["doc_id"],
                document_title=unit["document_title"],
                page=unit["page"],
                section_path=unit["section_path"],
                element_type=unit["element_type"],
                element_id=unit["element_id"],
                content=unit["content"]
            )
            all_evidence_units.append(rec)
            
        for t_unit in table_units:
            rec = create_provenance_record(
                doc_id=t_unit["doc_id"],
                document_title=t_unit["document_title"],
                page=t_unit["page"],
                section_path=t_unit["section_path"],
                element_type=t_unit["element_type"],
                element_id=t_unit["element_id"],
                content=t_unit["content"],
                table_json=t_unit.get("table_json"),
                table_markdown=t_unit.get("table_markdown")
            )
            all_evidence_units.append(rec)
            
        # 6. Save evidence units into SQLite database
        for ev in all_evidence_units:
            cursor.execute(
                """
                INSERT OR REPLACE INTO evidence_units 
                (element_id, doc_id, document_title, page, page_label, section_path, element_type, content, table_json, table_markdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev["element_id"], ev["doc_id"], ev["document_title"], ev["page"], ev["page_label"],
                    ev["section_path"], ev["element_type"], ev["content"], ev.get("table_json"), ev.get("table_markdown")
                )
            )
            
        # 7. Generate dense embeddings & index into VectorStore & BM25Index
        if all_evidence_units:
            contents = [ev["content"] for ev in all_evidence_units]
            vectors = generate_embeddings(contents)
            
            global_vector_store.add_items(vectors, all_evidence_units)
            global_bm25_index.add_items(all_evidence_units)
            
        # 8. Update document status to PROCESSED
        cursor.execute(
            "UPDATE documents SET page_count = ?, status = 'PROCESSED' WHERE doc_id = ?",
            (page_count, doc_id)
        )
        db.commit()
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "document_title": document_title,
            "page_count": page_count,
            "evidence_units_count": len(all_evidence_units)
        }
        
    except Exception as e:
        db.rollback()
        cursor.execute("UPDATE documents SET status = 'FAILED', error_message = ? WHERE doc_id = ?", (str(e), doc_id))
        db.commit()
        raise e
