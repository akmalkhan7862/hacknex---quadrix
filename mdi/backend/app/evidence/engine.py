"""
Evidence Engine Module for MDI
Orchestrates citation validation, source mapping, and output contract construction.
"""

from typing import Dict, Any, List
from backend.app.evidence.validator import validate_claims_and_evidence
from backend.app.evidence.citations import format_citations

class EvidenceEngine:
    def process(
        self, 
        llm_output: Dict[str, Any], 
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes LLM answer and retrieved evidence into the strict response schema.
        """
        final_answer, valid_claims, referenced_ids = validate_claims_and_evidence(llm_output, retrieved_evidence)
        
        if final_answer == "Not found in the provided documents." or not valid_claims:
            return {
                "answer": "Not found in the provided documents.",
                "claims": [],
                "sources": [],
                "calculations": []
            }
            
        sources = format_citations(retrieved_evidence, referenced_ids)
        
        return {
            "answer": final_answer,
            "claims": valid_claims,
            "sources": sources,
            "calculations": []
        }

global_evidence_engine = EvidenceEngine()
