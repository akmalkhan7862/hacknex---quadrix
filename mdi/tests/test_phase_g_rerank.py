"""
Phase G Unit Tests — Reranking Interface & Strategies

Covers:
  - Reranker abstract base class
  - DeterministicScoreFusionReranker (vector + BM25 score normalization & combination)
  - MockReranker (query term overlap scoring without neural models)
  - NoOpReranker (verifies reranking is an optional pass-through layer)
  - CrossEncoderReranker (neural reranker with deterministic fallback)
  - Provenance preservation in reranked EvidenceUnits
  - Backward compatibility: rerank_evidence() function in both rerank.py and reranker.py
  - Edge cases: empty candidates, top_n limits
"""

import pytest
from typing import List, Dict, Any

from backend.app.query.rerank import (
    Reranker,
    DeterministicScoreFusionReranker,
    MockReranker,
    CrossEncoderReranker,
    NoOpReranker,
    rerank_evidence,
    global_reranker,
)
from backend.app.query.reranker import rerank_evidence as legacy_rerank_evidence
from backend.app.query.schemas import EvidenceUnit


def _candidate_units() -> List[EvidenceUnit]:
    return [
        EvidenceUnit(
            doc_id="doc1",
            document_title="Operations Report",
            page=1,
            page_label="Page 1",
            section_path="Summary",
            element_type="text",
            element_id="u1",
            content="Q4 efficiency increased significantly across all manufacturing facilities.",
            score=0.9,
            metadata={"vector_score": 0.9, "bm25_score": 0.2},
        ),
        EvidenceUnit(
            doc_id="doc1",
            document_title="Operations Report",
            page=2,
            page_label="Page 2",
            section_path="Metrics",
            element_type="table",
            element_id="u2",
            content="| Facility | Metric | Value |\n| Plant A | Efficiency | 98% |",
            score=0.4,
            metadata={"vector_score": 0.4, "bm25_score": 0.95},
        ),
        EvidenceUnit(
            doc_id="doc2",
            document_title="Financial Report",
            page=5,
            page_label="Page 5",
            section_path="Expenses",
            element_type="text",
            element_id="u3",
            content="General corporate expenses remained flat throughout the year.",
            score=0.1,
            metadata={"vector_score": 0.1, "bm25_score": 0.05},
        ),
    ]


class TestDeterministicScoreFusionReranker:

    def test_rerank_evidence_units(self):
        reranker = DeterministicScoreFusionReranker(vector_weight=0.6, bm25_weight=0.4)
        candidates = _candidate_units()

        results = reranker.rerank("Q4 efficiency", candidates, top_n=2)
        assert len(results) == 2
        assert all(isinstance(r, EvidenceUnit) for r in results)

        # Check provenance is preserved
        top = results[0]
        assert top.doc_id in ("doc1", "doc2")
        assert top.document_title in ("Operations Report", "Financial Report")
        assert top.score is not None
        assert "rerank_score" in top.metadata

    def test_accepts_raw_dicts(self):
        reranker = DeterministicScoreFusionReranker()
        raw_candidates = [
            {"element_id": "e1", "content": "Text 1", "vector_score": 0.9, "bm25_score": 0.1, "doc_id": "d1", "page": 1},
            {"element_id": "e2", "content": "Text 2", "vector_score": 0.2, "bm25_score": 0.8, "doc_id": "d1", "page": 2},
        ]
        results = reranker.rerank("test", raw_candidates, top_n=2)
        assert len(results) == 2
        assert all(isinstance(r, EvidenceUnit) for r in results)

    def test_top_n_truncation(self):
        reranker = DeterministicScoreFusionReranker()
        candidates = _candidate_units()
        results = reranker.rerank("efficiency", candidates, top_n=1)
        assert len(results) == 1


class TestMockReranker:

    def test_mock_reranker_boosts_token_overlap(self):
        reranker = MockReranker()
        candidates = _candidate_units()

        # Query has terms matching u2 ("Facility Plant Metric")
        results = reranker.rerank("Plant Facility Metric", candidates, top_n=3)
        assert results[0].element_id == "u2"
        assert "mock_rerank_score" in results[0].metadata

    def test_empty_candidates(self):
        reranker = MockReranker()
        assert reranker.rerank("query", [], top_n=5) == []


class TestNoOpReranker:

    def test_noop_passthrough(self):
        reranker = NoOpReranker()
        candidates = _candidate_units()
        results = reranker.rerank("query", candidates, top_n=2)
        assert len(results) == 2
        # Original order preserved
        assert results[0].element_id == candidates[0].element_id
        assert results[1].element_id == candidates[1].element_id


class TestCrossEncoderReranker:

    def test_cross_encoder_fallback(self):
        # Force fallback to test graceful offline behaviour
        reranker = CrossEncoderReranker()
        reranker._model = "fallback"
        candidates = _candidate_units()

        results = reranker.rerank("Q4 efficiency", candidates, top_n=2)
        assert len(results) == 2
        assert all(isinstance(r, EvidenceUnit) for r in results)


class TestRerankerBackwardCompat:

    def test_rerank_evidence_from_rerank(self):
        raw = [
            {"element_id": "e1", "vector_score": 0.8, "bm25_score": 0.2},
            {"element_id": "e2", "vector_score": 0.1, "bm25_score": 0.9},
        ]
        results = rerank_evidence(raw, top_n=2)
        assert len(results) == 2
        assert isinstance(results[0], dict)
        assert "score" in results[0]

    def test_legacy_rerank_evidence_from_reranker(self):
        raw = [
            {"element_id": "e1", "vector_score": 0.8, "bm25_score": 0.2},
        ]
        results = legacy_rerank_evidence(raw, top_n=1)
        assert len(results) == 1
        assert results[0]["element_id"] == "e1"

    def test_global_singleton(self):
        assert global_reranker is not None
        assert isinstance(global_reranker, Reranker)
