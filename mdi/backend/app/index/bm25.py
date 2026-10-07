"""
BM25 Keyword Index Module for MDI
Creates BM25 index over text and table content while preserving full provenance and IDs.
Supports reference-based search (IDs + scores) and metadata filtering.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple


def _apply_bm25_filters(
    items: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Filter items by metadata constraints."""
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


class BM25Index:
    """
    In-memory BM25 index over EvidenceUnit content.
    Preserves element_id references and provides lightweight reference-based retrieval.
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.doc_map: Dict[str, Dict[str, Any]] = {}
        self.bm25_model = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer (converts to lowercase tokens)."""
        return re.findall(r"\w+", text.lower())

    def clear(self):
        self.documents = []
        self.corpus_tokens = []
        self.doc_map = {}
        self.bm25_model = None

    def build_index(self, provenances: List[Dict[str, Any]]):
        self.documents = provenances
        self.doc_map = {doc["element_id"]: doc for doc in provenances if "element_id" in doc}
        self.corpus_tokens = [self._tokenize(doc.get("content", "")) for doc in provenances]
        
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_model = BM25Okapi(self.corpus_tokens)
        except Exception as e:
            print(f"[bm25.py] rank_bm25 unavailable ({e}). Using native BM25 implementation.")
            self.bm25_model = None

    def add_items(self, provenances: List[Dict[str, Any]]):
        all_docs = self.documents + provenances
        self.build_index(all_docs)

    def get_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Resolve an element_id reference back to the original document record in O(1)."""
        return self.doc_map.get(element_id)

    def search_ids(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Return references as (element_id, score) pairs without cloning full documents.
        """
        results = self.search(query=query, top_k=top_k, filters=filters)
        return [(item["element_id"], item.get("bm25_score", 0.0)) for item in results]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents by query tokens, score with BM25, apply optional filters,
        and return top_k ranked documents.
        """
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        if self.bm25_model:
            doc_scores = self.bm25_model.get_scores(query_tokens)
        else:
            # Fallback simple TF-IDF / keyword overlap scoring
            doc_scores = []
            N = len(self.corpus_tokens)
            for doc_toks in self.corpus_tokens:
                score = 0.0
                doc_len = len(doc_toks) or 1
                for qt in query_tokens:
                    tf = doc_toks.count(qt)
                    if tf > 0:
                        df = sum(1 for dt in self.corpus_tokens if qt in dt)
                        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                        score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / 50.0)))
                doc_scores.append(score)
        
        # Rank all scores
        indexed_scores = list(enumerate(doc_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores:
            if score <= 0.0 and len(results) >= 1:
                # Don't include 0-score items if we already have matching candidates
                continue
            item = dict(self.documents[idx])
            item["bm25_score"] = round(float(score), 4)
            results.append(item)
            
        if filters:
            results = _apply_bm25_filters(results, filters)

        return results[:top_k]


# Singleton BM25 index instance (maintained for backward compatibility)
global_bm25_index = BM25Index()
