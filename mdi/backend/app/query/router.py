"""
Adaptive Query Router Module for MDI.

Chooses a retrieval strategy based on the structured Plan produced by the
Query Analyzer.

Two public surfaces:
  1. global_router.route_query(query_analysis: Dict) -> List[str]
        Legacy interface — returns list of engine name strings (e.g.
        ["vector", "bm25"]).  Kept intact for all existing callers in
        api/query.py and test_all.py.

  2. route_plan(plan: Plan) -> RetrievalStrategy
        New interface — consumes a Plan and returns a typed
        RetrievalStrategy enum value.  Used by fetch_evidence() pipeline.

Routing rules (deterministic, no LLM required):
  ┌──────────────────────────────────────────────┬──────────────────────┐
  │ Condition                                    │ Strategy             │
  ├──────────────────────────────────────────────┼──────────────────────┤
  │ cross_document = True                        │ CROSS_DOCUMENT       │
  │ chart or image in modalities                 │ MULTIMODAL           │
  │ (comparison or calculation) AND table        │ HYBRID               │
  │ table in modalities                          │ HYBRID               │
  │ calculation needed (no table/visual)         │ HYBRID               │
  │ text only                                    │ VECTOR               │
  └──────────────────────────────────────────────┴──────────────────────┘

Graph RAG remains an extension point.  The router can select GRAPH as a
strategy; the GraphRetriever (Phase M) handles the implementation.
"""

from typing import Dict, Any, List, Optional
from backend.app.query.schemas import Plan, RetrievalStrategy


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_plan(plan: Plan) -> RetrievalStrategy:
    """
    Determine the optimal retrieval strategy from a structured Plan.

    Parameters
    ----------
    plan : Plan
        Structured query plan produced by build_plan() / analyze_query().

    Returns
    -------
    RetrievalStrategy
        One of VECTOR, HYBRID, GRAPH, MULTIMODAL, CROSS_DOCUMENT.
    """
    modalities = set(plan.modalities or [])
    intent     = plan.intent or "fact_lookup"

    # ── Rule 1: Cross-document queries → dedicated strategy ──────────────
    if plan.cross_document:
        return RetrievalStrategy.CROSS_DOCUMENT

    # ── Rule 2: Visual modalities (chart / image) → MULTIMODAL ───────────
    if modalities & {"chart", "image"}:
        return RetrievalStrategy.MULTIMODAL

    # ── Rule 3: Table-involving queries benefit from both exact keyword
    #            (BM25) and semantic (vector) retrieval → HYBRID ──────────
    if "table" in modalities:
        return RetrievalStrategy.HYBRID

    # ── Rule 4: Calculation queries need precise value retrieval → HYBRID ─
    if plan.needs_calculation:
        return RetrievalStrategy.HYBRID

    # ── Rule 5: Comparison without table/visual — still HYBRID (two things
    #            being compared → more recall needed) ─────────────────────
    if intent == "comparison":
        return RetrievalStrategy.HYBRID

    # ── Rule 6: Exact terminology / numeric values benefit from keyword BM25 → HYBRID
    q_lower = plan.question.lower()
    if any(k in q_lower for k in ("exact", "specifically", "verbatim", "code", "id:")):
        return RetrievalStrategy.HYBRID

    # ── Default: pure semantic text retrieval ────────────────────────────
    return RetrievalStrategy.VECTOR


def strategy_to_engines(strategy: RetrievalStrategy) -> List[str]:
    """
    Map a RetrievalStrategy to the list of engine name strings that the
    legacy retrieval layer understands.

    Used internally so the new typed API can feed the old engine names
    without duplicating dispatch logic.
    """
    _MAP: Dict[RetrievalStrategy, List[str]] = {
        RetrievalStrategy.VECTOR:         ["vector"],
        RetrievalStrategy.HYBRID:         ["vector", "bm25"],
        RetrievalStrategy.MULTIMODAL:     ["vector", "bm25", "multimodal"],
        RetrievalStrategy.CROSS_DOCUMENT: ["vector", "bm25", "graph"],
        RetrievalStrategy.GRAPH:          ["graph"],
    }
    return _MAP.get(strategy, ["vector", "bm25"])


# ---------------------------------------------------------------------------
# Graph RAG extension point (Phase M will provide a real implementation)
# ---------------------------------------------------------------------------

class GraphRAGExtensionPoint:
    """
    Extension point for Graph RAG retrieval delegating to GraphRetriever.
    """

    def __init__(self, retriever: Optional[Any] = None):
        if retriever is not None:
            self._retriever = retriever
        else:
            try:
                from backend.app.query.graph import global_graph_retriever
                self._retriever = global_graph_retriever
            except Exception:
                self._retriever = None

    def retrieve(self, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._retriever:
            q = query_analysis.get("question", "")
            return self._retriever.retrieve_raw(q)
        return []

    @property
    def status(self) -> str:
        if self._retriever and getattr(self._retriever, "enabled", False):
            return self._retriever.status()
        return "not_implemented"


# ---------------------------------------------------------------------------
# QueryRouter — backward-compatible class used by api/query.py & test_all.py
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Adaptive query router.

    route_query() keeps the original dict-in / list[str]-out signature
    so existing callers require zero changes.

    Internally it delegates to route_plan() via a lightweight Plan built
    from the analysis dict, so routing decisions are always consistent.
    """

    def __init__(self):
        self.graph_rag = GraphRAGExtensionPoint()

    # ── Legacy interface ──────────────────────────────────────────────────
    def route_query(self, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Determine which retrieval engines should be executed.

        Parameters
        ----------
        query_analysis : Dict
            Dict returned by analyze_query() (or any dict with overlapping keys).

        Returns
        -------
        List[str]
            Engine name strings, e.g. ["vector", "bm25"].
        """
        plan = _dict_to_plan(query_analysis)
        strategy = route_plan(plan)
        return strategy_to_engines(strategy)

    # ── New typed interface ───────────────────────────────────────────────
    def route(self, plan: Plan) -> RetrievalStrategy:
        """Return a typed RetrievalStrategy from a Plan."""
        return route_plan(plan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_plan(d: Dict[str, Any]) -> Plan:
    """
    Build a lightweight Plan from a raw analyze_query() dict so that
    route_query() can call route_plan() without requiring callers to be
    updated first.
    """
    return Plan(
        question=d.get("question", ""),
        intent=d.get("intent", "fact_lookup"),
        entities=d.get("entities", []),
        periods=d.get("periods", []),
        modalities=d.get("modalities", ["text"]),
        needs_calculation=d.get("needs_calculation", False),
        cross_document=d.get("cross_document", False),
        query_type=d.get("query_type", "factual"),
        filters=d.get("filters", {}),
        sub_questions=d.get("sub_questions", []),
    )


# ---------------------------------------------------------------------------
# Module-level singleton (used by api/query.py)
# ---------------------------------------------------------------------------

global_router = QueryRouter()
