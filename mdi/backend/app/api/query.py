"""
Query & Evidence API Endpoints
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.db import get_db
from backend.app.query.analyzer import analyze_query
from backend.app.query.router import global_router
from backend.app.query.retriever import hybrid_retrieve
from backend.app.query.reranker import rerank_evidence
from backend.app.query.context_builder import build_context
from backend.app.reasoning.llm import generate_llm_answer
from backend.app.evidence.engine import global_evidence_engine

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

@router.post("/query")
def process_query(request: QueryRequest):
    """
    Executes complete Query RAG & Evidence validation pipeline.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    # 1. Query Analyzer
    query_analysis = analyze_query(question)
    
    # 2. Query Router
    active_engines = global_router.route_query(query_analysis)
    
    # 3. Hybrid Retrieval (Vector + BM25)
    candidates = hybrid_retrieve(question, top_k=request.top_k * 2)
    
    # 4. Reranker
    reranked = rerank_evidence(candidates, top_n=request.top_k)
    
    # 5. Context Builder
    context_str = build_context(reranked)
    
    # 6. LLM Answer Generation
    llm_output = generate_llm_answer(question, context_str, reranked)
    
    # 7. Evidence Engine & Citation Validation
    final_response = global_evidence_engine.process(llm_output, reranked)
    
    return final_response

@router.get("/evidence/{element_id}")
def get_evidence_details(element_id: str):
    """
    Retrieves full details of a specific evidence unit by element_id.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM evidence_units WHERE element_id = ?", (element_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Evidence unit not found.")
        
    return dict(row)
