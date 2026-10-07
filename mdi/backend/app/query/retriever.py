"""
Retriever Module for MDI.

Provides:

1. hybrid_retrieve(question, top_k) -> List[Dict]
        Legacy function — kept intact for all existing callers
        (api/query.py, test_all.py, etc.).

2. VectorRetriever (Phase D):
        Typed class-based interface for dense semantic retrieval.
        Supports metadata filtering and dependency injection.

3. BM25Retriever (Phase E):
        Typed class-based interface for keyword retrieval over BM25Index.
        Supports metadata filtering and lightweight ID reference retrieval.

4. HybridRetriever & reciprocal_rank_fusion (Phase F):
        Combines Vector search + BM25 search with Reciprocal Rank Fusion (RRF)
        and metadata filtering (document_id, period, content_type, section, entity).
        Provides vector-only, BM25-only, and hybrid modes.

Every result returned preserves:
    element_id, doc_id, document_title, page, page_label,
    section_path, element_type, content, score
"""

from typing import List, Dict, Any, Optional, Tuple

from backend.app.index.embeddings import generate_embedding, get_embedding_provider, EmbeddingProvider
from backend.app.index.vector_store import VectorStore, global_vector_store
from backend.app.index.bm25 import BM25Index, global_bm25_index
from backend.app.query.schemas import EvidenceUnit


# ---------------------------------------------------------------------------
# Legacy function (unchanged — used by api/query.py and existing tests)
# ---------------------------------------------------------------------------

def hybrid_retrieve(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Execute Vector and BM25 search, then combine and deduplicate results.
    Returns raw dicts so existing callers (api/query.py) are unaffected.
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


# ---------------------------------------------------------------------------
# Metadata filter helpers
# ---------------------------------------------------------------------------

def _apply_metadata_filters(
    items: List[Dict[str, Any]],
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Filter a list of provenance dicts by metadata constraints.

    Supported filter keys
    ---------------------
    document_id   : str  — match item["doc_id"]
    content_type  : str  — match item["element_type"]  (text | table | chart | image)
    section       : str  — substring match on item["section_path"]
    period        : str  — substring match on item["content"]  (e.g. "Q4")
    entity        : str  — substring match on item["content"], section_path, or document_title
    page          : int  — exact match on item["page"]
    """
    if not filters:
        return items

    out = []
    for item in items:
        if "document_id" in filters:
            if item.get("doc_id", "") != filters["document_id"]:
                continue
        if "content_type" in filters:
            if item.get("element_type", "") != filters["content_type"]:
                continue
        if "section" in filters:
            if filters["section"].lower() not in item.get("section_path", "").lower():
                continue
        if "period" in filters:
            if filters["period"].upper() not in item.get("content", "").upper():
                continue
        if "entity" in filters:
            e_str = str(filters["entity"]).lower()
            content_lower = item.get("content", "").lower()
            section_lower = item.get("section_path", "").lower()
            doc_lower = item.get("document_title", "").lower()
            if e_str not in content_lower and e_str not in section_lower and e_str not in doc_lower:
                continue
        if "page" in filters:
            if item.get("page") != filters["page"]:
                continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).
    score = sum(1 / (rrf_k + rank))
    Preserves vector_score, bm25_score, and all source metadata without data loss.
    """
    fused_scores: Dict[str, float] = {}
    item_map: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            elem_id = item.get("element_id")
            if not elem_id:
                continue
            if elem_id not in item_map:
                item_map[elem_id] = dict(item)
            else:
                existing = item_map[elem_id]
                if "vector_score" in item:
                    existing["vector_score"] = max(existing.get("vector_score", 0.0), item["vector_score"])
                if "bm25_score" in item:
                    existing["bm25_score"] = max(existing.get("bm25_score", 0.0), item["bm25_score"])

            rrf_score = 1.0 / (rrf_k + rank)
            fused_scores[elem_id] = fused_scores.get(elem_id, 0.0) + rrf_score

    sorted_ids = sorted(fused_scores.keys(), key=lambda eid: fused_scores[eid], reverse=True)

    results = []
    for eid in sorted_ids[:top_n]:
        item = item_map[eid]
        fused = round(fused_scores[eid], 6)
        item["score"] = fused
        item["rrf_score"] = fused
        results.append(item)

    return results


# ---------------------------------------------------------------------------
# VectorRetriever — Phase D typed class
# ---------------------------------------------------------------------------

class VectorRetriever:
    """
    Semantic vector retrieval over the VectorStore.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self._embedder: EmbeddingProvider = (
            embedding_provider if embedding_provider is not None
            else get_embedding_provider()
        )
        self._store: VectorStore = (
            vector_store if vector_store is not None
            else global_vector_store
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        if not question or not question.strip():
            return []
        q_vec = self._embedder.embed_text(question)
        raw_hits: List[Dict[str, Any]] = self._store.search(q_vec, top_k=top_k)
        if filters:
            raw_hits = _apply_metadata_filters(raw_hits, filters)
        return [self._to_evidence_unit(hit) for hit in raw_hits]

    def retrieve_raw(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            return []
        q_vec = self._embedder.embed_text(question)
        raw_hits = self._store.search(q_vec, top_k=top_k)
        if filters:
            raw_hits = _apply_metadata_filters(raw_hits, filters)
        return raw_hits

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    @staticmethod
    def _to_evidence_unit(raw: Dict[str, Any]) -> EvidenceUnit:
        known_keys = {
            "doc_id", "document_title", "page", "page_label", "section_path",
            "element_type", "element_id", "content", "table_json",
            "table_markdown", "vector_score", "bm25_score", "score", "rrf_score",
        }
        extra_meta = {k: v for k, v in raw.items() if k not in known_keys}

        return EvidenceUnit(
            doc_id=raw.get("doc_id", "unknown_doc"),
            document_title=raw.get("document_title", "Unknown Document"),
            page=raw.get("page", 1),
            page_label=raw.get("page_label", f"Page {raw.get('page', 1)}"),
            section_path=raw.get("section_path", "Unknown Section"),
            element_type=raw.get("element_type", "text"),
            element_id=raw.get("element_id", "unknown_id"),
            content=raw.get("content", ""),
            table_json=raw.get("table_json"),
            table_markdown=raw.get("table_markdown"),
            score=raw.get("vector_score") or raw.get("score"),
            metadata={
                "vector_score": raw.get("vector_score", 0.0),
                "bm25_score": raw.get("bm25_score", 0.0),
                **extra_meta,
            },
        )


# ---------------------------------------------------------------------------
# BM25Retriever — Phase E typed class
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Keyword BM25 retrieval over BM25Index.
    """

    def __init__(self, index: Optional[BM25Index] = None):
        self._index: BM25Index = index if index is not None else global_bm25_index

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        if not question or not question.strip():
            return []
        raw_hits = self._index.search(query=question, top_k=top_k, filters=filters)
        return [self._to_evidence_unit(hit) for hit in raw_hits]

    def retrieve_ids(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if not question or not question.strip():
            return []
        return self._index.search_ids(query=question, top_k=top_k, filters=filters)

    def retrieve_raw(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            return []
        return self._index.search(query=question, top_k=top_k, filters=filters)

    @property
    def index(self) -> BM25Index:
        return self._index

    @staticmethod
    def _to_evidence_unit(raw: Dict[str, Any]) -> EvidenceUnit:
        known_keys = {
            "doc_id", "document_title", "page", "page_label", "section_path",
            "element_type", "element_id", "content", "table_json",
            "table_markdown", "vector_score", "bm25_score", "score", "rrf_score",
        }
        extra_meta = {k: v for k, v in raw.items() if k not in known_keys}

        return EvidenceUnit(
            doc_id=raw.get("doc_id", "unknown_doc"),
            document_title=raw.get("document_title", "Unknown Document"),
            page=raw.get("page", 1),
            page_label=raw.get("page_label", f"Page {raw.get('page', 1)}"),
            section_path=raw.get("section_path", "Unknown Section"),
            element_type=raw.get("element_type", "text"),
            element_id=raw.get("element_id", "unknown_id"),
            content=raw.get("content", ""),
            table_json=raw.get("table_json"),
            table_markdown=raw.get("table_markdown"),
            score=raw.get("bm25_score") or raw.get("score"),
            metadata={
                "vector_score": raw.get("vector_score", 0.0),
                "bm25_score": raw.get("bm25_score", 0.0),
                **extra_meta,
            },
        )


# ---------------------------------------------------------------------------
# HybridRetriever — Phase F typed class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid Retriever combining Vector search and BM25 search with
    Reciprocal Rank Fusion (RRF) and metadata filtering.
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
    ):
        self._vector_retriever: VectorRetriever = (
            vector_retriever if vector_retriever is not None
            else global_vector_retriever
        )
        self._bm25_retriever: BM25Retriever = (
            bm25_retriever if bm25_retriever is not None
            else global_bm25_retriever
        )
        self.rrf_k: int = rrf_k

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """
        Execute Hybrid retrieval: Vector top-k + BM25 top-k, metadata filtering, and RRF fusion.
        """
        if not question or not question.strip():
            return []

        candidates = self.retrieve_raw(question=question, top_k=top_k, filters=filters)
        return [self._to_evidence_unit(c) for c in candidates]

    def retrieve_vector_only(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """Execute vector-only retrieval."""
        return self._vector_retriever.retrieve(question=question, top_k=top_k, filters=filters)

    def retrieve_bm25_only(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """Execute BM25-only retrieval."""
        return self._bm25_retriever.retrieve(question=question, top_k=top_k, filters=filters)

    def retrieve_raw(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return fused raw candidate dicts with RRF scores.
        """
        fetch_k = max(top_k * 2, 10)
        vec_candidates = self._vector_retriever.retrieve_raw(question=question, top_k=fetch_k, filters=filters)
        bm25_candidates = self._bm25_retriever.retrieve_raw(question=question, top_k=fetch_k, filters=filters)

        fused = reciprocal_rank_fusion(
            ranked_lists=[vec_candidates, bm25_candidates],
            rrf_k=self.rrf_k,
            top_n=top_k,
        )
        return fused

    @property
    def vector_retriever(self) -> VectorRetriever:
        return self._vector_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        return self._bm25_retriever

    @staticmethod
    def _to_evidence_unit(raw: Dict[str, Any]) -> EvidenceUnit:
        known_keys = {
            "doc_id", "document_title", "page", "page_label", "section_path",
            "element_type", "element_id", "content", "table_json",
            "table_markdown", "vector_score", "bm25_score", "score", "rrf_score",
        }
        extra_meta = {k: v for k, v in raw.items() if k not in known_keys}

        return EvidenceUnit(
            doc_id=raw.get("doc_id", "unknown_doc"),
            document_title=raw.get("document_title", "Unknown Document"),
            page=raw.get("page", 1),
            page_label=raw.get("page_label", f"Page {raw.get('page', 1)}"),
            section_path=raw.get("section_path", "Unknown Section"),
            element_type=raw.get("element_type", "text"),
            element_id=raw.get("element_id", "unknown_id"),
            content=raw.get("content", ""),
            table_json=raw.get("table_json"),
            table_markdown=raw.get("table_markdown"),
            score=raw.get("score") or raw.get("rrf_score"),
            metadata={
                "vector_score": raw.get("vector_score", 0.0),
                "bm25_score": raw.get("bm25_score", 0.0),
                "rrf_score": raw.get("rrf_score", 0.0),
                **extra_meta,
            },
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

global_vector_retriever = VectorRetriever()
global_bm25_retriever = BM25Retriever()
global_hybrid_retriever = HybridRetriever()
