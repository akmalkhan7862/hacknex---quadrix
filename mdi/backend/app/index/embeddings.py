"""
Embeddings Generation Module for MDI.

Provides two interfaces:

1. Module-level functions (existing, kept intact for all callers):
     generate_embedding(text)       -> List[float]
     generate_embeddings(texts)     -> List[List[float]]
     get_embedding_model()          -> model | "fallback"

2. EmbeddingProvider class (new, Phase D):
     A replaceable abstraction so any embedding backend can be swapped
     without touching ingestion or retrieval code.
     embed_text(text)               -> List[float]
     embed_batch(texts)             -> List[List[float]]
     dimension                      -> int   (property)

Uses SentenceTransformers (all-MiniLM-L6-v2 / BGE) with a
deterministic hash-based fallback that requires no model download.
"""

import os
from typing import List
import numpy as np

# ---------------------------------------------------------------------------
# Internal model singleton
# ---------------------------------------------------------------------------

_model = None


def get_embedding_model():
    """Return loaded SentenceTransformer model, or the string 'fallback'."""
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
        print(
            f"[embeddings.py] SentenceTransformer unavailable ({e}). "
            "Using deterministic fallback embedding model."
        )
        _model = "fallback"
        return _model


def _fallback_embed(text: str, dim: int = 384) -> List[float]:
    """Fallback embedding: hash-based word-frequency vector, L2-normalised."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        idx = abs(hash(word)) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Module-level functions (unchanged — used by ingestion pipeline, retriever,
# and existing tests)
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> List[float]:
    """Generate a dense embedding vector for a single text string."""
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
    """Batch-generate dense embedding vectors."""
    model = get_embedding_model()
    if model == "fallback" or model is None:
        return [_fallback_embed(t) for t in texts]
    try:
        vectors = model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()
    except Exception as e:
        print(f"[embeddings.py] Batch encoding error: {e}")
        return [_fallback_embed(t) for t in texts]


# ---------------------------------------------------------------------------
# EmbeddingProvider — Phase D abstraction
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Replaceable embedding abstraction used by the Vector Retriever and
    the fetch_evidence() pipeline.

    Wraps the existing generate_embedding / generate_embeddings functions
    so the retrieval layer is decoupled from the specific embedding backend.

    A different backend (e.g. OpenAI, Cohere, a local Ollama model) can be
    injected by subclassing EmbeddingProvider and overriding embed_text()
    and embed_batch().

    Attributes
    ----------
    model_name : str
        Human-readable identifier of the active embedding backend.
    """

    # Expected vector dimension for the default SentenceTransformer model.
    # Fallback also produces 384-dimensional vectors.
    _DEFAULT_DIM = 384

    def __init__(self):
        # Trigger model load on construction so the first search is fast.
        self._raw_model = get_embedding_model()
        self._dim: int = self._detect_dimension()

    # ── Core interface ────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return a float vector."""
        return generate_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings in one call."""
        if not texts:
            return []
        return generate_embeddings(texts)

    # ── Optional visual embedding hook (stub — Phase L may implement) ─────

    def embed_image(self, image_path: str) -> List[float]:
        """
        Hook for visual embeddings.  Not implemented in Phase D.
        Raises NotImplementedError until a VLM embedding backend is wired in.
        """
        raise NotImplementedError(
            "Visual embeddings require a VLM backend. "
            "Implement in Phase L or inject a custom EmbeddingProvider."
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors this provider produces."""
        return self._dim

    @property
    def model_name(self) -> str:
        """Human-readable backend identifier."""
        if self._raw_model == "fallback" or self._raw_model is None:
            return "fallback-hash-embed-384"
        try:
            # SentenceTransformer stores its name on the object
            return getattr(self._raw_model, "_model_card_vars", {}).get(
                "model_id", os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
        except Exception:
            return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @property
    def is_fallback(self) -> bool:
        """True when the deterministic hash-based fallback is active."""
        return self._raw_model == "fallback" or self._raw_model is None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _detect_dimension(self) -> int:
        """Probe the actual vector dimension by embedding an empty string."""
        try:
            probe = self.embed_text("probe")
            return len(probe) if probe else self._DEFAULT_DIM
        except Exception:
            return self._DEFAULT_DIM


# ---------------------------------------------------------------------------
# Module-level singleton provider (used by VectorRetriever by default)
# ---------------------------------------------------------------------------

_default_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the module-level EmbeddingProvider singleton.

    Creates it on first call so it shares the already-loaded model with
    the module-level generate_embedding / generate_embeddings functions.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = EmbeddingProvider()
    return _default_provider
