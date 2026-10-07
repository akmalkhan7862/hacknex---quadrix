"""
Query Analyzer Module for MDI
Analyzes user questions to produce structured query metadata.
"""

import re
from typing import Dict, Any, List

def analyze_query(question: str) -> Dict[str, Any]:
    """
    Analyze question string and output structured query parameters.
    """
    q_lower = question.lower().strip()
    
    # Keyword detection for table / calculation intent
    table_keywords = ["table", "column", "row", "revenue", "profit", "sales", "percentage", "amount", "cost", "total", "margin", "budget", "financial"]
    calc_keywords = ["calculate", "sum", "total", "difference", "average", "growth", "increase", "ratio", "percent"]
    
    needs_table = any(kw in q_lower for kw in table_keywords)
    needs_calculation = any(kw in q_lower for kw in calc_keywords)
    
    # Extract keywords (words with length >= 3)
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", question)
    stop_words = {"what", "where", "when", "which", "how", "many", "does", "that", "this", "from", "with", "have", "been", "were"}
    keywords = [w for w in words if w.lower() not in stop_words]
    
    # Extract capitalized entities / proper nouns
    entities = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", question)
    
    return {
        "question": question,
        "needs_text": True,
        "needs_table": needs_table,
        "needs_calculation": needs_calculation,
        "keywords": list(dict.fromkeys(keywords)),
        "entities": list(dict.fromkeys(entities))
    }
