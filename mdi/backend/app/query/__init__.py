"""
Query package for MDI Adaptive Router and Multimodal Retrieval.
Exposes the primary public entry point: fetch_evidence()
"""

from backend.app.query.pipeline import fetch_evidence
from backend.app.query.schemas import (
    Plan,
    RetrievalStrategy,
    EvidenceUnit,
    EvidencePacket,
    Provenance,
    Citation,
    Claim,
    AnswerJSON,
)
from backend.app.query.analyzer import analyze_query, build_plan
from backend.app.query.router import route_plan, global_router
from backend.app.query.retriever import (
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    global_vector_retriever,
    global_bm25_retriever,
    global_hybrid_retriever,
    hybrid_retrieve,
)
from backend.app.query.multimodal import MultimodalRetriever, global_multimodal_retriever
from backend.app.query.rerank import (
    Reranker,
    DeterministicScoreFusionReranker,
    MockReranker,
    CrossEncoderReranker,
    NoOpReranker,
    rerank_evidence,
    global_reranker,
)
from backend.app.query.graph import GraphRetriever, global_graph_retriever

__all__ = [
    "fetch_evidence",
    "Plan",
    "RetrievalStrategy",
    "EvidenceUnit",
    "EvidencePacket",
    "Provenance",
    "Citation",
    "Claim",
    "AnswerJSON",
    "analyze_query",
    "build_plan",
    "route_plan",
    "global_router",
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "global_vector_retriever",
    "global_bm25_retriever",
    "global_hybrid_retriever",
    "hybrid_retrieve",
    "MultimodalRetriever",
    "global_multimodal_retriever",
    "Reranker",
    "DeterministicScoreFusionReranker",
    "MockReranker",
    "CrossEncoderReranker",
    "NoOpReranker",
    "rerank_evidence",
    "global_reranker",
    "GraphRetriever",
    "global_graph_retriever",
]
