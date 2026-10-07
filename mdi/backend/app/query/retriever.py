"""
Retriever Module for MDI.

Provides two interfaces:

1. hybrid_retrieve(question, top_k) -> List[Dict]
        Legacy function — kept intact for all existing callers
        (api/query.py, test_all.py, etc.).

2. VectorRetriever (Phase D):
        Typed class-based interface that:
          - accepts an EmbeddingProvider and a VectorStore
          - returns List[EvidenceUnit] with full provenance guaranteed
          - supports metadata filtering (document_id, page, section,
            content_type / element_type, period keywords)
          - is injected by default from module-level singletons so
            existing callers need zero changes

Every result returned by VectorRetriever preserves:
    element_id, doc_id, document_title, page, page_label,
    section_path, element_type, content, vector_score
"""

from typing import List, Dict, Any, Optional

from backend.app.index.embeddings import generate_embedding, get_embedding_provider, EmbeddingProvider
from backend.app.index.vector_store import VectorStore, global_vector_store
from backend.app.index.bm25 import global_bm25_index
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
        if "page" in filters:
            if item.get("page") != filters["page"]:
                continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# VectorRetriever — Phase D typed class
# ---------------------------------------------------------------------------

class VectorRetriever:
    """
    Semantic vector retrieval over the VectorStore.

    Embeds the query via EmbeddingProvider, searches the VectorStore for
    the top-k nearest neighbours, applies optional metadata filters, and
    returns typed EvidenceUnit objects with full provenance preserved.

    Parameters
    ----------
    embedding_provider : EmbeddingProvider | None
        Provider used to embed queries.  Defaults to the module singleton.
    vector_store : VectorStore | None
        Store to search.  Defaults to global_vector_store singleton.
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

    # ── Core retrieval ────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceUnit]:
        """
        Embed the question, search the vector store, apply filters, and
        return typed EvidenceUnit objects.

        Provenance fields guaranteed on every returned unit:
            element_id, doc_id, document_title, page, page_label,
            section_path, element_type, content

        Parameters
        ----------
        question : str
            Natural-language query to embed and search.
        top_k : int
            Maximum number of nearest-neighbour candidates to retrieve.
        filters : dict | None
            Optional metadata constraints (see _apply_metadata_filters).

        Returns
        -------
        List[EvidenceUnit]
            Retrieved units sorted by descending vector similarity.
        """
        if not question or not question.strip():
            return []

        # Embed query
        q_vec = self._embedder.embed_text(question)

        # Search vector store — returns raw dicts with vector_score
        raw_hits: List[Dict[str, Any]] = self._store.search(q_vec, top_k=top_k)

        # Apply optional metadata filters
        if filters:
            raw_hits = _apply_metadata_filters(raw_hits, filters)

        # Convert to typed EvidenceUnit, preserving all provenance
        return [self._to_evidence_unit(hit) for hit in raw_hits]

    def retrieve_raw(
        self,
        question: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Same as retrieve() but returns raw dicts (for hybrid fusion use).
        """
        if not question or not question.strip():
            return []
        q_vec = self._embedder.embed_text(question)
        raw_hits = self._store.search(q_vec, top_k=top_k)
        if filters:
            raw_hits = _apply_metadata_filters(raw_hits, filters)
        return raw_hits

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_evidence_unit(raw: Dict[str, Any]) -> EvidenceUnit:
        """
        Convert a raw vector-store result dict into a typed EvidenceUnit.
        Provenance fields are mapped explicitly; extra keys go into metadata.
        """
        known_keys = {
            "doc_id", "document_title", "page", "page_label", "section_path",
            "element_type", "element_id", "content", "table_json",
            "table_markdown", "vector_score", "bm25_score", "score",
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
# Module-level singleton (used by future HybridRetriever in Phase F)
# ---------------------------------------------------------------------------

global_vector_retriever = VectorRetriever()
