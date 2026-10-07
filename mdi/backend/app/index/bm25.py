"""
BM25 Keyword Index Module for MDI
Creates BM25 index over text and table content while preserving full provenance.
"""

import re
import math
from typing import List, Dict, Any

class BM25Index:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25_model = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer."""
        return re.findall(r"\w+", text.lower())

    def clear(self):
        self.documents = []
        self.corpus_tokens = []
        self.bm25_model = None

    def build_index(self, provenances: List[Dict[str, Any]]):
        self.documents = provenances
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

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        
        # Get top indices
        indexed_scores = list(enumerate(doc_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0.0 and len(results) >= 1:
                # Don't include 0-score items if we already have candidates
                continue
            item = dict(self.documents[idx])
            item["bm25_score"] = round(float(score), 4)
            results.append(item)
            
        return results

# Singleton BM25 index instance
global_bm25_index = BM25Index()
