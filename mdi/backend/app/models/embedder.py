"""
Embedding Abstraction Module for MDI.

Defines:
  - EmbeddingProvider (ABC): Replaceable interface for text and visual embeddings
  - SentenceTransformerEmbedder: Neural sentence embeddings using SentenceTransformers
  - DeterministicFallbackEmbedder: Hash-based L2-normalized vectors (offline, zero-dependency)
  - MockEmbedder: Controllable test embedder
  - get_embedder(): Factory function returning the configured embedder
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract interface for dense embedding providers.
    Supports text embeddings and defines a replaceable interface for future visual embeddings.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a dense vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in a batch."""
        pass

    def embed_image(self, image_path: str) -> List[float]:
        """
        Hook for future visual embeddings.
        Not implemented per project architecture constraints (start with text embeddings).
        """
        raise NotImplementedError(
            "Visual embeddings are not required by current architecture. "
            "Vision analysis is handled by the VLM layer."
        )

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of produced vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        pass


class DeterministicFallbackEmbedder(EmbeddingProvider):
    """
    Deterministic hash-based embedding provider.
    Works completely offline with zero dependencies or model downloads.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = abs(hash(word)) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"deterministic-fallback-{self._dim}"


class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    SentenceTransformers embedding provider (e.g. all-MiniLM-L6-v2, BGE).
    Gracefully falls back to DeterministicFallbackEmbedder if sentence_transformers is unavailable.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model = None
        self._fallback = DeterministicFallbackEmbedder(dim=384)
        self._dim = 384

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            # Detect dimension
            probe = self._model.encode("probe", convert_to_numpy=True)
            self._dim = len(probe)
            return self._model
        except Exception as e:
            print(f"[embedder.py] SentenceTransformer unavailable ({e}). Using fallback embedder.")
            self._model = "fallback"
            return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self._load_model()
        if model == "fallback" or model is None:
            return self._fallback.embed_text(text)
        try:
            vec = model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as e:
            print(f"[embedder.py] Error in encode: {e}")
            return self._fallback.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load_model()
        if model == "fallback" or model is None:
            return self._fallback.embed_batch(texts)
        try:
            vecs = model.encode(texts, convert_to_numpy=True)
            return vecs.tolist()
        except Exception as e:
            print(f"[embedder.py] Error in batch encode: {e}")
            return self._fallback.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


class MockEmbedder(EmbeddingProvider):
    """
    Deterministic mock embedder for tests, supporting fixed or custom term-to-vector mappings.
    """

    def __init__(self, dim: int = 4, mapping: Optional[Dict[str, List[float]]] = None):
        self._dim = dim
        self._mapping = mapping or {}

    def embed_text(self, text: str) -> List[float]:
        t_lower = text.lower()
        for k, v in self._mapping.items():
            if k.lower() in t_lower:
                return list(v)
        # Default unit vector
        v = [0.0] * self._dim
        v[abs(hash(text)) % self._dim] = 1.0
        return v

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"mock-embedder-{self._dim}"


def get_embedder(model_name: Optional[str] = None) -> EmbeddingProvider:
    """Factory creating the primary configured embedding provider."""
    return SentenceTransformerEmbedder(model_name=model_name)


# Global singleton instance
global_embedder = get_embedder()
