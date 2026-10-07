"""
Multimodal Retrieval Module for MDI.

Orchestrates retrieval across heterogeneous modalities:
  - Text retrieval (content_type='text')
  - Table retrieval (content_type='table')
  - Visual retrieval (content_type in ('chart', 'image') or items with visual_ref)

Produces structured EvidencePacket containing typed EvidenceUnits with full provenance.
"""

from typing import List, Dict, Any, Optional, Set

from backend.app.query.schemas import Plan, EvidenceUnit, EvidencePacket, RetrievalStrategy
from backend.app.query.router import route_plan
from backend.app.query.retriever import HybridRetriever, global_hybrid_retriever
from backend.app.query.rerank import Reranker, DeterministicScoreFusionReranker, global_reranker


class MultimodalRetriever:
    """
    Multimodal Retriever that inspects query plan modalities and element content_types,
    routing queries through specialized modality retrieval paths and assembling EvidencePackets.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.hybrid_retriever: HybridRetriever = (
            hybrid_retriever if hybrid_retriever is not None
            else global_hybrid_retriever
        )
        self.reranker: Reranker = (
            reranker if reranker is not None
            else global_reranker
        )

    # ── Specialized modality retrieval paths ──────────────────────────────

    def retrieve_text(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """Retrieve purely textual evidence units."""
        text_filters = dict(filters) if filters else {}
        text_filters["content_type"] = "text"
        return self.hybrid_retriever.retrieve(question, top_k=top_k, filters=text_filters)

    def retrieve_tables(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """Retrieve structured table evidence units."""
        table_filters = dict(filters) if filters else {}
        table_filters["content_type"] = "table"
        return self.hybrid_retriever.retrieve(question, top_k=top_k, filters=table_filters)

    def retrieve_visuals(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """
        Retrieve visual evidence units (charts and images).
        Searches for content_type in ('chart', 'image') or units with visual_ref.
        """
        base_filters = dict(filters) if filters else {}

        # Search charts
        chart_filters = dict(base_filters)
        chart_filters["content_type"] = "chart"
        chart_units = self.hybrid_retriever.retrieve(question, top_k=top_k, filters=chart_filters)

        # Search images
        image_filters = dict(base_filters)
        image_filters["content_type"] = "image"
        image_units = self.hybrid_retriever.retrieve(question, top_k=top_k, filters=image_filters)

        # Combine and deduplicate
        seen: Set[str] = set()
        visual_units: List[EvidenceUnit] = []
        for u in chart_units + image_units:
            if u.element_id not in seen:
                seen.add(u.element_id)
                visual_units.append(u)

        return visual_units[:top_k]

    # ── Core multimodal dispatch ──────────────────────────────────────────

    def retrieve(
        self,
        plan: Plan,
        top_k: int = 10,
    ) -> EvidencePacket:
        """
        Execute multimodal retrieval according to the provided Plan.
        Inspects plan.modalities, queries the corresponding retrieval paths,
        merges candidates, applies reranking, and returns a verified EvidencePacket.
        """
        question = plan.question
        if not question or not question.strip():
            return EvidencePacket(
                question=question,
                plan=plan,
                strategy=RetrievalStrategy.VECTOR,
                evidence_units=[],
                retrieval_metadata={"status": "empty_question"}
            )

        strategy = route_plan(plan)
        requested_modalities = set(plan.modalities or ["text"])
        user_filters = dict(plan.filters) if plan.filters else {}

        collected_units: List[EvidenceUnit] = []
        seen_ids: Set[str] = set()
        searched_paths: List[str] = []

        # 1. Visual Path (if requested)
        if "chart" in requested_modalities or "image" in requested_modalities or strategy == RetrievalStrategy.MULTIMODAL:
            searched_paths.append("visual")
            visuals = self.retrieve_visuals(question, top_k=top_k, filters=user_filters)
            for v in visuals:
                if v.element_id not in seen_ids:
                    seen_ids.add(v.element_id)
                    collected_units.append(v)

        # 2. Table Path (if requested)
        if "table" in requested_modalities:
            searched_paths.append("table")
            tables = self.retrieve_tables(question, top_k=top_k, filters=user_filters)
            for t in tables:
                if t.element_id not in seen_ids:
                    seen_ids.add(t.element_id)
                    collected_units.append(t)

        # 3. Text Path (always executed or as fallback if specific modalities returned few items)
        if "text" in requested_modalities or len(collected_units) < top_k:
            searched_paths.append("text")
            # If user explicitly filtered by content_type in plan.filters, respect it; otherwise search text
            text_units = self.retrieve_text(question, top_k=top_k, filters=user_filters)
            for tx in text_units:
                if tx.element_id not in seen_ids:
                    seen_ids.add(tx.element_id)
                    collected_units.append(tx)

        # 4. Fallback broad search if no modal-filtered units matched
        if not collected_units:
            searched_paths.append("broad_hybrid_fallback")
            broad_units = self.hybrid_retriever.retrieve(question, top_k=top_k, filters=user_filters)
            for bu in broad_units:
                if bu.element_id not in seen_ids:
                    seen_ids.add(bu.element_id)
                    collected_units.append(bu)

        # 5. Optional Reranking layer
        final_units = self.reranker.rerank(question, collected_units, top_n=top_k)

        # 6. Build EvidencePacket
        text_count = sum(1 for u in final_units if u.element_type == "text")
        table_count = sum(1 for u in final_units if u.element_type == "table")
        visual_count = sum(1 for u in final_units if u.element_type in ("chart", "image"))

        return EvidencePacket(
            question=question,
            plan=plan,
            strategy=strategy,
            evidence_units=final_units,
            retrieval_metadata={
                "strategy": strategy.value,
                "requested_modalities": list(requested_modalities),
                "searched_paths": searched_paths,
                "total_units": len(final_units),
                "text_units": text_count,
                "table_units": table_count,
                "visual_units": visual_count,
            }
        )


# Singleton instance
global_multimodal_retriever = MultimodalRetriever()
