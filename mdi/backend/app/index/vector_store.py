"""
Vector Store Module for MDI
Stores dense embeddings and complete provenance metadata.
Supports vector similarity search using numpy/cosine similarity with FAISS compatibility.
"""

from typing import List, Dict, Any
import numpy as np

class VectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def clear(self):
        self.vectors = []
        self.metadata = []

    def add_item(self, vector: List[float], provenance: Dict[str, Any]):
        vec_arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec_arr)
        if norm > 0:
            vec_arr = vec_arr / norm
        self.vectors.append(vec_arr)
        self.metadata.append(provenance)

    def add_items(self, vectors: List[List[float]], provenances: List[Dict[str, Any]]):
        for vec, prov in zip(vectors, provenances):
            self.add_item(vec, prov)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        
        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm
            
        matrix = np.vstack(self.vectors)  # (N, D)
        scores = np.dot(matrix, q_arr)    # Cosine similarity (N,)
        
        # Sort indices descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            item = dict(self.metadata[idx])
            item["vector_score"] = round(score, 4)
            results.append(item)
            
        return results

# Singleton vector store instance
global_vector_store = VectorStore()
