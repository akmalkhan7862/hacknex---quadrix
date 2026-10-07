"""
End-to-End Retrieval Pipeline Module for MDI.

Coordinates:
  Query Plan (or sub_question)
         │
         ▼
  Adaptive Router
         │
  ┌──────┼──────────────┐
  ▼      ▼              ▼
Vector Hybrid       Multimodal / Graph
  │      │              │
  └──────┼──────────────┘
         ▼
      Reranker
         │
         ▼
   EvidencePacket

Exposes the primary public module entry point:
  fetch_evidence(plan: Plan, sub_question: Optional[str] = None) -> EvidencePacket
"""

from typing import Optional, List, Dict, Any, Set

from backend.app.query.schemas import Plan, EvidencePacket, EvidenceUnit, RetrievalStrategy
from backend.app.query.router import route_plan, global_router
from backend.app.query.retriever import (
    VectorRetriever,
    HybridRetriever,
    global_vector_retriever,
    global_hybrid_retriever,
)
from backend.app.query.multimodal import MultimodalRetriever, global_multimodal_retriever
from backend.app.query.graph import GraphRetriever, global_graph_retriever
from backend.app.query.rerank import Reranker, DeterministicScoreFusionReranker, global_reranker


def fetch_evidence(
    plan: Plan,
    sub_question: Optional[str] = None,
    top_k: int = 10,
    vector_retriever: Optional[VectorRetriever] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    multimodal_retriever: Optional[MultimodalRetriever] = None,
    graph_retriever: Optional[GraphRetriever] = None,
    reranker: Optional[Reranker] = None,
) -> EvidencePacket:
    """
    Public entry point for the Adaptive Router + Multimodal Retrieval module.

    Parameters
    ----------
    plan : Plan
        Structured query plan produced by Query Analyzer (build_plan).
    sub_question : Optional[str]
        Optional targeted sub-question for multi-step reasoning decomposition.
    top_k : int
        Maximum number of evidence units to return in the EvidencePacket.
    vector_retriever, hybrid_retriever, multimodal_retriever, graph_retriever, reranker :
        Optional injectable dependencies for custom or test configurations.

    Returns
    -------
    EvidencePacket
        Structured evidence packet containing verified, provenance-backed EvidenceUnits.
    """
    # 1. Determine effective query text
    query_text = (sub_question.strip() if sub_question and sub_question.strip()
                  else (plan.question.strip() if plan.question else ""))

    if not query_text:
        return EvidencePacket(
            question="",
            plan=plan,
            strategy=RetrievalStrategy.VECTOR,
            evidence_units=[],
            retrieval_metadata={"status": "empty_query", "total_units": 0}
        )

    # 2. Resolve component dependencies
    v_retriever = vector_retriever or global_vector_retriever
    h_retriever = hybrid_retriever or global_hybrid_retriever
    m_retriever = multimodal_retriever or global_multimodal_retriever
    g_retriever = graph_retriever or global_graph_retriever
    r_reranker = reranker or global_reranker

    # 3. Determine retrieval strategy via Adaptive Router
    strategy = route_plan(plan)
    filters = dict(plan.filters) if plan.filters else {}

    raw_units: List[EvidenceUnit] = []
    seen_ids: Set[str] = set()

    # 4. Dispatch based on strategy
    if strategy == RetrievalStrategy.VECTOR:
        raw_units = v_retriever.retrieve(query_text, top_k=top_k, filters=filters)

    elif strategy == RetrievalStrategy.HYBRID:
        raw_units = h_retriever.retrieve(query_text, top_k=top_k, filters=filters)

    elif strategy == RetrievalStrategy.MULTIMODAL:
        effective_plan = plan.model_copy()
        effective_plan.question = query_text
        packet = m_retriever.retrieve(effective_plan, top_k=top_k)
        return packet  # MultimodalRetriever handles complete packet assembly internally

    elif strategy == RetrievalStrategy.CROSS_DOCUMENT:
        # Cross-document questions combine hybrid document search + graph traversal
        hybrid_candidates = h_retriever.retrieve(query_text, top_k=top_k, filters=filters)
        graph_candidates = g_retriever.retrieve(query_text, top_k=top_k, filters=filters)

        for u in graph_candidates + hybrid_candidates:
            if u.element_id not in seen_ids:
                seen_ids.add(u.element_id)
                raw_units.append(u)

    elif strategy == RetrievalStrategy.GRAPH:
        raw_units = g_retriever.retrieve(query_text, top_k=top_k, filters=filters)
        if not raw_units:
            # Fallback to hybrid if graph is unconfigured
            raw_units = h_retriever.retrieve(query_text, top_k=top_k, filters=filters)

    else:
        # Default safety fallback
        raw_units = h_retriever.retrieve(query_text, top_k=top_k, filters=filters)

    # 5. Optional Reranking layer
    final_units = r_reranker.rerank(query_text, raw_units, top_n=top_k)

    # 6. Deduplicate & preserve order
    deduped_units: List[EvidenceUnit] = []
    seen: Set[str] = set()
    for unit in final_units:
        if unit.element_id not in seen:
            seen.add(unit.element_id)
            deduped_units.append(unit)

    # 7. Construct and return final EvidencePacket
    return EvidencePacket(
        question=query_text,
        plan=plan,
        strategy=strategy,
        evidence_units=deduped_units,
        retrieval_metadata={
            "strategy": strategy.value,
            "sub_question_used": sub_question is not None,
            "total_units": len(deduped_units),
            "text_units": sum(1 for u in deduped_units if u.element_type == "text"),
            "table_units": sum(1 for u in deduped_units if u.element_type == "table"),
            "visual_units": sum(1 for u in deduped_units if u.element_type in ("chart", "image")),
        }
    )
