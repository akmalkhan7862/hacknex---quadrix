"""
Reranker Module for MDI
Reranks candidate evidence items using cross-encoder or deterministic score fusion.
"""

from typing import List, Dict, Any

def rerank_evidence(candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank evidence candidates based on normalized score fusion.
    """
    if not candidates:
        return []
    
    # Extract raw scores
    v_scores = [c.get("vector_score", 0.0) for c in candidates]
    b_scores = [c.get("bm25_score", 0.0) for c in candidates]
    
    max_v = max(v_scores) if v_scores and max(v_scores) > 0 else 1.0
    min_v = min(v_scores) if v_scores else 0.0
    
    max_b = max(b_scores) if b_scores and max(b_scores) > 0 else 1.0
    min_b = min(b_scores) if b_scores else 0.0
    
    reranked = []
    for c in candidates:
        v_raw = c.get("vector_score", 0.0)
        b_raw = c.get("bm25_score", 0.0)
        
        # Normalize 0.0 - 1.0
        norm_v = (v_raw - min_v) / (max_v - min_v) if max_v > min_v else v_raw
        norm_b = (b_raw - min_b) / (max_b - min_b) if max_b > min_b else b_raw
        
        # Weighted combination: 60% vector + 40% BM25
        final_score = round(0.6 * norm_v + 0.4 * norm_b, 4)
        
        item = dict(c)
        item["score"] = final_score
        reranked.append(item)
        
    # Sort descending by score
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_n]
