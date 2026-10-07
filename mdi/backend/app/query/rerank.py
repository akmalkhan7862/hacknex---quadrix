"""
Reranker Module for MDI.

Provides modular reranking as an OPTIONAL pipeline layer:
Candidate EvidenceUnits -> Reranker -> Top relevant evidence

Abstractions:
  - Reranker: Abstract base class
  - DeterministicScoreFusionReranker: Fuses normalized vector + BM25/RRF scores
  - MockReranker: Deterministic query-overlap scoring (offline / zero-dependency)
  - CrossEncoderReranker: Optional cross-encoder model with fallback
  - NoOpReranker: Transparent pass-through (reranking disabled)

Also provides backward-compatible function:
  rerank_evidence(candidates, top_n) -> List[Dict]
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

from backend.app.query.schemas import EvidenceUnit


class Reranker(ABC):
    """Abstract base class for evidence rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Union[EvidenceUnit, Dict[str, Any]]],
        top_n: int = 5
    ) -> List[EvidenceUnit]:
        """
        Rerank a candidate evidence pool and return the top_n most relevant EvidenceUnits.
        """
        pass

    @staticmethod
    def _ensure_evidence_unit(item: Union[EvidenceUnit, Dict[str, Any]]) -> EvidenceUnit:
        """Convert dict or EvidenceUnit to EvidenceUnit with complete provenance."""
        if isinstance(item, EvidenceUnit):
            return item
        return EvidenceUnit.from_dict(item)


class DeterministicScoreFusionReranker(Reranker):
    """
    Reranks candidates using normalized score fusion (60% vector + 40% BM25/RRF).
    Guarantees reproducible, deterministic ordering without requiring neural models.
    """

    def __init__(self, vector_weight: float = 0.6, bm25_weight: float = 0.4):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def rerank(
        self,
        query: str,
        candidates: List[Union[EvidenceUnit, Dict[str, Any]]],
        top_n: int = 5
    ) -> List[EvidenceUnit]:
        if not candidates:
            return []

        units = [self._ensure_evidence_unit(c) for c in candidates]

        v_scores = [float(u.metadata.get("vector_score") or (u.score if u.score is not None else 0.0)) for u in units]
        b_scores = [float(u.metadata.get("bm25_score") or 0.0) for u in units]

        max_v = max(v_scores) if v_scores and max(v_scores) > 0 else 1.0
        min_v = min(v_scores) if v_scores else 0.0

        max_b = max(b_scores) if b_scores and max(b_scores) > 0 else 1.0
        min_b = min(b_scores) if b_scores else 0.0

        scored_units = []
        for u in units:
            v_raw = float(u.metadata.get("vector_score") or (u.score if u.score is not None else 0.0))
            b_raw = float(u.metadata.get("bm25_score") or 0.0)

            norm_v = (v_raw - min_v) / (max_v - min_v) if max_v > min_v else v_raw
            norm_b = (b_raw - min_b) / (max_b - min_b) if max_b > min_b else b_raw

            final_score = round(self.vector_weight * norm_v + self.bm25_weight * norm_b, 4)
            unit_copy = u.model_copy(deep=True)
            unit_copy.score = final_score
            unit_copy.metadata["rerank_score"] = final_score
            scored_units.append(unit_copy)

        scored_units.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_units[:top_n]


class MockReranker(Reranker):
    """
    Deterministic mock reranker using exact query token frequency and overlap.
    Works completely offline without any AI model dependencies.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Union[EvidenceUnit, Dict[str, Any]]],
        top_n: int = 5
    ) -> List[EvidenceUnit]:
        if not candidates:
            return []

        units = [self._ensure_evidence_unit(c) for c in candidates]
        q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 2]

        scored_units = []
        for u in units:
            content_lower = u.content.lower()
            overlap_score = sum(content_lower.count(t) for t in q_tokens)
            base_score = float(u.score or 0.0)
            rerank_score = round(base_score + (0.5 * overlap_score), 4)

            unit_copy = u.model_copy(deep=True)
            unit_copy.score = rerank_score
            unit_copy.metadata["mock_rerank_score"] = rerank_score
            scored_units.append(unit_copy)

        scored_units.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_units[:top_n]


class CrossEncoderReranker(Reranker):
    """
    Optional neural cross-encoder reranker.
    Gracefully falls back to DeterministicScoreFusionReranker if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = DeterministicScoreFusionReranker()

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as e:
            print(f"[rerank.py] CrossEncoder unavailable ({e}). Using deterministic fallback.")
            self._model = "fallback"
            return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Union[EvidenceUnit, Dict[str, Any]]],
        top_n: int = 5
    ) -> List[EvidenceUnit]:
        if not candidates:
            return []

        model = self._get_model()
        if model == "fallback" or model is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        units = [self._ensure_evidence_unit(c) for c in candidates]
        pairs = [[query, u.content] for u in units]

        try:
            scores = model.predict(pairs)
            scored_units = []
            for u, s in zip(units, scores):
                unit_copy = u.model_copy(deep=True)
                unit_copy.score = round(float(s), 4)
                unit_copy.metadata["cross_encoder_score"] = round(float(s), 4)
                scored_units.append(unit_copy)

            scored_units.sort(key=lambda x: x.score or 0.0, reverse=True)
            return scored_units[:top_n]
        except Exception as e:
            print(f"[rerank.py] CrossEncoder prediction failed ({e}). Falling back.")
            return self._fallback.rerank(query, candidates, top_n=top_n)


class NoOpReranker(Reranker):
    """Transparent pass-through reranker when reranking is disabled or bypassed."""

    def rerank(
        self,
        query: str,
        candidates: List[Union[EvidenceUnit, Dict[str, Any]]],
        top_n: int = 5
    ) -> List[EvidenceUnit]:
        units = [self._ensure_evidence_unit(c) for c in candidates]
        return units[:top_n]


# ---------------------------------------------------------------------------
# Legacy function (kept for backward compatibility with api/query.py & test_all.py)
# ---------------------------------------------------------------------------

def rerank_evidence(candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank evidence candidates based on normalized score fusion.
    Returns raw dicts so existing callers are unaffected.
    """
    if not candidates:
        return []

    v_scores = [c.get("vector_score", 0.0) for c in candidates]
    b_scores = [c.get("bm25_score", 0.0) for c in candidates]

    max_v = max(v_scores) if v_scores and max(v_scores) > 0 else 1.0
    min_v = min(v_scores) if v_scores else 0.0

    max_b = max(b_scores) if b_scores and max(b_scores) > 0 else 1.0
    min_b = min(b_scores) if b_scores else 0.0

    reranked = []
    for c in candidates:
        v_raw = c.get("vector_score", 0.0)
        b_raw = c.get("bm25_score", 0.0)

        norm_v = (v_raw - min_v) / (max_v - min_v) if max_v > min_v else v_raw
        norm_b = (b_raw - min_b) / (max_b - min_b) if max_b > min_b else b_raw

        final_score = round(0.6 * norm_v + 0.4 * norm_b, 4)

        item = dict(c)
        item["score"] = final_score
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_n]


# Default singleton reranker instance
global_reranker = DeterministicScoreFusionReranker()
