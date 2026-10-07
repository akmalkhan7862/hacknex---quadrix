"""
Hybrid Retriever Module for MDI
Executes Vector search and BM25 search, merges candidate pools,
deduplicates by element_id, and preserves full provenance metadata.
"""

from typing import List, Dict, Any
from backend.app.index.embeddings import generate_embedding
from backend.app.index.vector_store import global_vector_store
from backend.app.index.bm25 import global_bm25_index

def hybrid_retrieve(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Execute Vector and BM25 search, then combine and deduplicate results.
    """
    # 1. Vector RAG Retrieval
    q_vec = generate_embedding(question)
    vector_results = global_vector_store.search(q_vec, top_k=top_k)
    
    # 2. BM25 Retrieval
    bm25_results = global_bm25_index.search(question, top_k=top_k)
    
    # 3. Merge & Deduplicate by element_id
    candidates_map: Dict[str, Dict[str, Any]] = {}
    
    for item in vector_results:
        elem_id = item["element_id"]
        candidates_map[elem_id] = dict(item)
        candidates_map[elem_id]["vector_score"] = item.get("vector_score", 0.0)
        candidates_map[elem_id]["bm25_score"] = 0.0
        
    for item in bm25_results:
        elem_id = item["element_id"]
        if elem_id in candidates_map:
            candidates_map[elem_id]["bm25_score"] = item.get("bm25_score", 0.0)
        else:
            candidates_map[elem_id] = dict(item)
            candidates_map[elem_id]["vector_score"] = 0.0
            candidates_map[elem_id]["bm25_score"] = item.get("bm25_score", 0.0)
            
    return list(candidates_map.values())
