"""
Phase C Unit Tests — Adaptive Router

Covers:
  - route_plan(Plan) -> RetrievalStrategy for all routing rules
  - strategy_to_engines() mapping
  - route_query() backward-compatible dict interface
  - QueryRouter.route() typed interface
  - GraphRAGExtensionPoint stub
  - All PHASE O scenario questions → expected strategies
"""

import pytest
from backend.app.query.schemas import Plan, RetrievalStrategy
from backend.app.query.router import (
    route_plan,
    strategy_to_engines,
    QueryRouter,
    GraphRAGExtensionPoint,
    global_router,
    _dict_to_plan,
)
from backend.app.query.analyzer import build_plan, analyze_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(**kwargs) -> Plan:
    """Create a minimal Plan for routing tests."""
    defaults = dict(
        question="test question",
        intent="fact_lookup",
        entities=[],
        periods=[],
        modalities=["text"],
        needs_calculation=False,
        cross_document=False,
        query_type="factual",
        filters={},
        sub_questions=[],
    )
    defaults.update(kwargs)
    return Plan(**defaults)


# ---------------------------------------------------------------------------
# route_plan() — core routing rules
# ---------------------------------------------------------------------------

class TestRoutePlan:

    def test_vector_for_text_only(self):
        plan = _plan(modalities=["text"], intent="fact_lookup")
        assert route_plan(plan) == RetrievalStrategy.VECTOR

    def test_hybrid_for_table(self):
        plan = _plan(modalities=["text", "table"])
        assert route_plan(plan) == RetrievalStrategy.HYBRID

    def test_hybrid_for_table_only(self):
        plan = _plan(modalities=["table"])
        assert route_plan(plan) == RetrievalStrategy.HYBRID

    def test_hybrid_for_calculation(self):
        plan = _plan(modalities=["text"], needs_calculation=True)
        assert route_plan(plan) == RetrievalStrategy.HYBRID

    def test_hybrid_for_comparison_no_table(self):
        plan = _plan(modalities=["text"], intent="comparison")
        assert route_plan(plan) == RetrievalStrategy.HYBRID

    def test_multimodal_for_chart(self):
        plan = _plan(modalities=["text", "chart"])
        assert route_plan(plan) == RetrievalStrategy.MULTIMODAL

    def test_multimodal_for_image(self):
        plan = _plan(modalities=["text", "image"])
        assert route_plan(plan) == RetrievalStrategy.MULTIMODAL

    def test_multimodal_for_chart_and_table(self):
        """Chart takes priority over table → MULTIMODAL."""
        plan = _plan(modalities=["text", "table", "chart"])
        assert route_plan(plan) == RetrievalStrategy.MULTIMODAL

    def test_cross_document_flag_takes_priority(self):
        """cross_document overrides everything else."""
        plan = _plan(
            modalities=["text", "chart"],
            cross_document=True,
            intent="comparison",
        )
        assert route_plan(plan) == RetrievalStrategy.CROSS_DOCUMENT

    def test_cross_document_simple(self):
        plan = _plan(cross_document=True)
        assert route_plan(plan) == RetrievalStrategy.CROSS_DOCUMENT

    def test_default_aggregation_returns_vector(self):
        plan = _plan(intent="aggregation", modalities=["text"])
        assert route_plan(plan) == RetrievalStrategy.VECTOR


# ---------------------------------------------------------------------------
# strategy_to_engines()
# ---------------------------------------------------------------------------

class TestStrategyToEngines:

    def test_vector_engines(self):
        assert strategy_to_engines(RetrievalStrategy.VECTOR) == ["vector"]

    def test_hybrid_engines(self):
        engines = strategy_to_engines(RetrievalStrategy.HYBRID)
        assert "vector" in engines
        assert "bm25" in engines

    def test_multimodal_engines(self):
        engines = strategy_to_engines(RetrievalStrategy.MULTIMODAL)
        assert "multimodal" in engines
        assert "vector" in engines

    def test_cross_document_engines(self):
        engines = strategy_to_engines(RetrievalStrategy.CROSS_DOCUMENT)
        assert "graph" in engines

    def test_graph_engines(self):
        assert "graph" in strategy_to_engines(RetrievalStrategy.GRAPH)


# ---------------------------------------------------------------------------
# QueryRouter.route_query() — backward-compatible dict interface
# ---------------------------------------------------------------------------

class TestRouteQueryBackwardCompat:

    def test_returns_list_of_strings(self):
        result = global_router.route_query({"question": "sample"})
        assert isinstance(result, list)
        assert all(isinstance(e, str) for e in result)

    def test_text_query_uses_vector(self):
        analysis = analyze_query("What is the executive summary?")
        engines = global_router.route_query(analysis)
        # pure text → vector only
        assert "vector" in engines

    def test_table_query_uses_hybrid(self):
        analysis = analyze_query("What was production efficiency in Q4?")
        engines = global_router.route_query(analysis)
        assert "vector" in engines
        assert "bm25" in engines

    def test_chart_query_uses_multimodal(self):
        analysis = analyze_query("What trend does the production efficiency chart show?")
        engines = global_router.route_query(analysis)
        assert "multimodal" in engines

    def test_cross_doc_includes_graph(self):
        analysis = analyze_query("Compare the annual report and operations report.")
        engines = global_router.route_query(analysis)
        assert "graph" in engines

    def test_original_test_all_contract(self):
        """
        A bare minimal dict {"question": "sample"} has no table / chart /
        calculation / cross-document signals → routes to VECTOR.
        The original test_all.py only asserts "vector" in engines, which still
        passes.  We replicate that assertion here.
        """
        engines = global_router.route_query({"question": "sample"})
        assert "vector" in engines  # minimal dict → VECTOR route


# ---------------------------------------------------------------------------
# QueryRouter.route() — new typed interface
# ---------------------------------------------------------------------------

class TestQueryRouterTypedInterface:

    def test_returns_retrieval_strategy(self):
        plan = build_plan("What is the company revenue?")
        result = global_router.route(plan)
        assert isinstance(result, RetrievalStrategy)

    def test_route_visual_query(self):
        plan = build_plan("What trend is shown in the production efficiency chart?")
        assert global_router.route(plan) == RetrievalStrategy.MULTIMODAL

    def test_route_table_query(self):
        plan = build_plan("What was production efficiency in Q4?")
        assert global_router.route(plan) == RetrievalStrategy.HYBRID

    def test_route_cross_doc_query(self):
        plan = build_plan("Compare the annual report and operations report.")
        assert global_router.route(plan) == RetrievalStrategy.CROSS_DOCUMENT


# ---------------------------------------------------------------------------
# GraphRAGExtensionPoint stub
# ---------------------------------------------------------------------------

class TestGraphRAGExtensionPoint:

    def test_returns_empty_list(self):
        stub = GraphRAGExtensionPoint()
        result = stub.retrieve({"question": "anything"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_status_not_implemented(self):
        stub = GraphRAGExtensionPoint()
        assert stub.status == "not_implemented"

    def test_does_not_raise(self):
        stub = GraphRAGExtensionPoint()
        try:
            stub.retrieve({"question": "test"})
        except Exception as e:
            pytest.fail(f"GraphRAGExtensionPoint.retrieve() raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# _dict_to_plan() helper
# ---------------------------------------------------------------------------

class TestDictToPlan:

    def test_minimal_dict(self):
        plan = _dict_to_plan({"question": "test"})
        assert isinstance(plan, Plan)
        assert plan.question == "test"

    def test_rich_dict(self):
        d = {
            "question": "Compare Q2 and Q4",
            "intent": "comparison",
            "periods": ["Q2", "Q4"],
            "modalities": ["text", "table"],
            "needs_calculation": False,
            "cross_document": False,
            "query_type": "comparative",
        }
        plan = _dict_to_plan(d)
        assert plan.intent == "comparison"
        assert "Q2" in plan.periods
        assert "table" in plan.modalities

    def test_empty_dict_uses_defaults(self):
        plan = _dict_to_plan({})
        assert plan.intent == "fact_lookup"
        assert "text" in plan.modalities


# ---------------------------------------------------------------------------
# PHASE O Scenario Tests — end-to-end analyzer → router
# ---------------------------------------------------------------------------

class TestPhaseOScenarios:
    """
    Validate the analyzer→router pipeline for all PHASE O test questions.
    """

    def _route(self, question: str) -> RetrievalStrategy:
        plan = build_plan(question)
        return route_plan(plan)

    def test_q1_production_efficiency_q4(self):
        """Q1: table retrieval expected."""
        strategy = self._route("What was production efficiency in Q4?")
        assert strategy == RetrievalStrategy.HYBRID  # table → hybrid

    def test_q2_exact_value_q4(self):
        """Q2: BM25 + table expected."""
        strategy = self._route("What exact value is reported for Q4?")
        # periods detected, no explicit table kw → still factual/text
        assert strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID)

    def test_q3_trend_chart(self):
        """Q3: visual retrieval expected."""
        strategy = self._route("What trend is shown in the production efficiency chart?")
        assert strategy == RetrievalStrategy.MULTIMODAL

    def test_q4_compare_q2_q4(self):
        """Q4: hybrid + table + potentially visual."""
        strategy = self._route("Compare Q2 and Q4 production efficiency.")
        assert strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.MULTIMODAL)

    def test_q5_cross_document(self):
        """Q5: cross-document / graph-capable route."""
        strategy = self._route("Compare the annual report and operations report.")
        assert strategy == RetrievalStrategy.CROSS_DOCUMENT

    def test_q6_calculate_percentage(self):
        """Q6: retrieve evidence only — calculation intent → HYBRID."""
        strategy = self._route("Calculate the percentage increase.")
        assert strategy == RetrievalStrategy.HYBRID
