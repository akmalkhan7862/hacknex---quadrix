"""
Graph RAG Extension Point Interface Module for MDI.

Defines:
  - GraphRetriever: Extension point interface for future Knowledge Graph RAG retrieval.
  - Does NOT implement a full graph database.
  - Serves as an extension point returning typed EvidenceUnit stubs or an empty list when unconfigured.
  - Ensures end-to-end pipelines run without errors when Graph RAG is enabled.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.query.schemas import EvidenceUnit


class GraphEdgeStub(BaseModel):
    """Represents a lightweight entity relation stub for cross-document testing."""
    source_entity: str
    target_entity: str
    relation: str
    doc_id: str
    evidence_content: str
    page: int = 1
    section_path: str = "Knowledge Graph"


class GraphRetriever:
    """
    Extension point interface for Graph RAG retrieval.

    When unconfigured, retrieve() returns an empty list without raising exceptions.
    When configured with mock entity edges, retrieve() returns linked EvidenceUnits
    representing cross-document entity traversal.
    """

    def __init__(self, enabled: bool = False):
        self.enabled: bool = enabled
        self._mock_edges: List[GraphEdgeStub] = []

    def status(self) -> str:
        """
        Return current operational status of the Graph RAG extension point.
        """
        if not self.enabled:
            return "unconfigured"
        if self._mock_edges:
            return f"configured_mock_graph ({len(self._mock_edges)} edges)"
        return "enabled_empty_graph"

    def add_mock_edge(
        self,
        source_entity: str,
        target_entity: str,
        relation: str,
        doc_id: str,
        evidence_content: str,
        page: int = 1,
        section_path: str = "Knowledge Graph"
    ):
        """Add mock relation edge for cross-document reasoning testing."""
        self.enabled = True
        self._mock_edges.append(
            GraphEdgeStub(
                source_entity=source_entity,
                target_entity=target_entity,
                relation=relation,
                doc_id=doc_id,
                evidence_content=evidence_content,
                page=page,
                section_path=section_path,
            )
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceUnit]:
        """
        Retrieve graph evidence units matching query entities.
        Returns empty list when unconfigured.
        """
        if not self.enabled or not self._mock_edges:
            return []

        q_lower = question.lower()
        matched_units: List[EvidenceUnit] = []

        for idx, edge in enumerate(self._mock_edges):
            # Check if source entity or target entity is mentioned in question
            src_match = edge.source_entity.lower() in q_lower
            tgt_match = edge.target_entity.lower() in q_lower
            rel_match = edge.relation.lower() in q_lower

            if src_match or tgt_match or rel_match:
                if filters and "document_id" in filters:
                    if edge.doc_id != filters["document_id"]:
                        continue

                unit = EvidenceUnit(
                    doc_id=edge.doc_id,
                    document_title=f"Graph Document: {edge.doc_id}",
                    page=edge.page,
                    page_label=f"Page {edge.page}",
                    section_path=edge.section_path,
                    element_type="text",
                    element_id=f"graph-edge-{idx + 1}",
                    content=(
                        f"[Graph Relation: {edge.source_entity} --({edge.relation})--> {edge.target_entity}] "
                        f"{edge.evidence_content}"
                    ),
                    score=0.95 if (src_match and tgt_match) else 0.75,
                    metadata={
                        "graph_source": edge.source_entity,
                        "graph_target": edge.target_entity,
                        "graph_relation": edge.relation,
                    }
                )
                matched_units.append(unit)

        matched_units.sort(key=lambda x: x.score or 0.0, reverse=True)
        return matched_units[:top_k]

    def retrieve_raw(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Legacy dict-based retrieve method."""
        units = self.retrieve(question=question, top_k=top_k, filters=filters)
        return [u.model_dump() for u in units]


# Singleton instance
global_graph_retriever = GraphRetriever()
