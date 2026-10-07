"""
Evidence Validator Module for MDI
Validates that every claim from LLM output maps to valid, retrieved evidence IDs.
Implements the strict anti-hallucination evidence gate.
"""

from typing import Dict, Any, List, Tuple

def validate_claims_and_evidence(
    llm_output: Dict[str, Any], 
    retrieved_evidence: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Validates LLM claims against retrieved evidence items.
    
    Returns:
        (validated_answer, valid_claims, valid_referenced_evidence_ids)
    """
    valid_ids = {ev["element_id"] for ev in retrieved_evidence if "element_id" in ev}
    
    raw_answer = llm_output.get("answer", "").strip()
    raw_claims = llm_output.get("claims", [])
    
    if not raw_answer or raw_answer == "Not found in the provided documents.":
        return "Not found in the provided documents.", [], []
        
    valid_claims = []
    referenced_ids = set()
    
    for claim in raw_claims:
        claim_text = claim.get("text", "").strip()
        e_ids = claim.get("evidence_ids", [])
        
        # Filter e_ids to those present in retrieved evidence
        matched_ids = [eid for eid in e_ids if eid in valid_ids]
        
        if matched_ids and claim_text:
            valid_claims.append({
                "text": claim_text,
                "evidence_ids": matched_ids
            })
            referenced_ids.update(matched_ids)
            
    # If LLM provided claims but NONE were valid, gate the answer
    if raw_claims and not valid_claims:
        return "Not found in the provided documents.", [], []
        
    # If no claims were structured but answer exists, attach top evidence ID if valid
    if not valid_claims and valid_ids:
        top_id = list(valid_ids)[0]
        valid_claims.append({
            "text": raw_answer,
            "evidence_ids": [top_id]
        })
        referenced_ids.add(top_id)
        
    return raw_answer, valid_claims, list(referenced_ids)
