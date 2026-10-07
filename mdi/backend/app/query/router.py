"""
Query Router Module for MDI
Routes analyzed queries to active retrieval engines (Vector RAG, BM25).
Provides explicit extension point for Phase 2 Graph RAG.
"""

from typing import Dict, Any, List

class GraphRAGExtensionPoint:
    """Extension point for Phase 2 Graph RAG."""
    def retrieve(self, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Graph RAG intentionally not implemented in Phase 1 MVP
        return []

class QueryRouter:
    def __init__(self):
        self.graph_rag = GraphRAGExtensionPoint()

    def route_query(self, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Determine which retrieval engines should be executed.
        For MVP Phase 1: Always routes to Vector RAG and BM25.
        """
        active_engines = ["vector", "bm25"]
        
        # Future extension: if query_analysis.get("needs_graph"):
        # active_engines.append("graph")
        
        return active_engines

global_router = QueryRouter()
