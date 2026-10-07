"""
Reranker Module for MDI (Alias / Re-export to backend.app.query.rerank).
Maintains 100% backward compatibility for all existing imports.
"""

from backend.app.query.rerank import (
    Reranker,
    DeterministicScoreFusionReranker,
    MockReranker,
    CrossEncoderReranker,
    NoOpReranker,
    rerank_evidence,
    global_reranker,
)

__all__ = [
    "Reranker",
    "DeterministicScoreFusionReranker",
    "MockReranker",
    "CrossEncoderReranker",
    "NoOpReranker",
    "rerank_evidence",
    "global_reranker",
]
