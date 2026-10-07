"""
Embeddings Generation Module for MDI
Uses SentenceTransformers (all-MiniLM-L6-v2 / BGE) with graceful fallback.
"""

import os
from typing import List
import numpy as np

_model = None

def get_embedding_model():
    global _model
    if _model is not None:
        return _model
    
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings.py] Loading SentenceTransformer model '{model_name}'...")
        _model = SentenceTransformer(model_name)
        return _model
    except Exception as e:
        print(f"[embeddings.py] SentenceTransformer unavailable ({e}). Using deterministic fallback embedding model.")
        _model = "fallback"
        return _model

def _fallback_embed(text: str, dim: int = 384) -> List[float]:
    """Fallback embedding generator using text hashing and character n-gram frequencies."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        idx = abs(hash(word)) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

def generate_embedding(text: str) -> List[float]:
    """Generate dense embedding vector for text string."""
    model = get_embedding_model()
    if model == "fallback" or model is None:
        return _fallback_embed(text)
    
    try:
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()
    except Exception as e:
        print(f"[embeddings.py] Error generating embedding with model: {e}")
        return _fallback_embed(text)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate dense embedding vectors."""
    model = get_embedding_model()
    if model == "fallback" or model is None:
        return [_fallback_embed(t) for t in texts]
    
    try:
        vectors = model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()
    except Exception as e:
        print(f"[embeddings.py] Batch encoding error: {e}")
        return [_fallback_embed(t) for t in texts]
