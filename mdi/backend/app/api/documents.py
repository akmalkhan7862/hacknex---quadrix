"""
Document Upload & Retrieval API Endpoints
"""

import uuid
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from backend.app.db import get_db, RAW_DIR
from backend.app.ingest.pipeline import run_ingestion_pipeline

router = APIRouter()

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Accepts one or multiple PDF documents, saves them to data/raw/,
    creates database records, and triggers the ingestion pipeline.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    results = []
    db = get_db()
    cursor = db.cursor()
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "status": "FAILED",
                "error": "Only PDF files are supported."
            })
            continue
            
        doc_id = uuid.uuid4().hex[:12]
        saved_filename = f"{doc_id}_{file.filename}"
        saved_path = RAW_DIR / saved_filename
        
        # Save uploaded file
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(saved_path)
        
        # Save initial document record in SQLite
        cursor.execute(
            """
            INSERT INTO documents (doc_id, document_title, file_path, file_size, page_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, file.filename, str(saved_path), file_size, 0, "UPLOADED")
        )
        db.commit()
        
        # Run ingestion pipeline
        try:
            pipeline_res = run_ingestion_pipeline(doc_id, str(saved_path), file.filename)
            results.append({
                "doc_id": doc_id,
                "document_title": file.filename,
                "status": "PROCESSED",
                "page_count": pipeline_res["page_count"],
                "evidence_units": pipeline_res["evidence_units_count"]
            })
        except Exception as e:
            results.append({
                "doc_id": doc_id,
                "document_title": file.filename,
                "status": "FAILED",
                "error": str(e)
            })
            
    return {"uploaded_documents": results}

@router.get("")
def list_documents():
    """Returns list of all uploaded documents and their processing status."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT doc_id, document_title, file_size, page_count, status, created_at, error_message FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    docs = [dict(row) for row in rows]
    return {"documents": docs}

@router.get("/{document_id}")
def get_document_details(document_id: str):
    """Returns details, page classifications, and evidence unit count for a specific document."""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (document_id,))
    doc_row = cursor.fetchone()
    if not doc_row:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    doc_dict = dict(doc_row)
    
    cursor.execute("SELECT page, page_type, quality_score FROM pages WHERE doc_id = ? ORDER BY page ASC", (document_id,))
    pages_rows = cursor.fetchall()
    pages_list = [dict(p) for p in pages_rows]
    
    cursor.execute("SELECT COUNT(*) as ev_count FROM evidence_units WHERE doc_id = ?", (document_id,))
    ev_count = cursor.fetchone()["ev_count"]
    
    doc_dict["pages"] = pages_list
    doc_dict["evidence_units_count"] = ev_count
    
    return doc_dict
