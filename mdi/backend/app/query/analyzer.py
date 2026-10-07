"""
Query Analyzer Module for MDI.

Converts a natural-language question into structured query metadata.

Two public surfaces:
  1. analyze_query(question) -> Dict[str, Any]
        Legacy dict format — kept intact for all existing callers
        (api/query.py, tests, etc.).
  2. build_plan(question) -> Plan
        New structured form consumed by the Adaptive Router and
        the fetch_evidence() pipeline.

Both are fully deterministic / rule-based. No LLM call is required.
"""

import re
from typing import Dict, Any, List, Tuple

from backend.app.query.schemas import Plan


# ---------------------------------------------------------------------------
# Vocabulary tables (all lowercase)
# ---------------------------------------------------------------------------

_TABLE_KEYWORDS = {
    "table", "column", "row", "revenue", "profit", "sales", "percentage",
    "amount", "cost", "total", "margin", "budget", "financial", "figure",
    "figures", "metric", "metrics", "number", "numbers", "value", "values",
    "rate", "ratio", "production", "efficiency", "output", "yield",
    "quarter", "quarterly", "annual", "annually",
}

_CHART_KEYWORDS = {
    "chart", "graph", "plot", "trend", "diagram", "visualization",
    "bar chart", "line chart", "pie chart", "histogram", "shows", "depicted",
    "illustrated", "visual", "visually",
}

_IMAGE_KEYWORDS = {
    "image", "picture", "photo", "photograph", "figure", "illustration",
    "diagram", "schematic", "drawing",
}

_CALC_KEYWORDS = {
    "calculate", "computation", "compute", "sum", "difference", "average",
    "growth", "increase", "decrease", "change", "ratio", "percent",
    "percentage", "rate", "derive", "formula", "how much", "by how much",
}

_COMPARE_KEYWORDS = {
    "compare", "comparison", "versus", "vs", "against", "between",
    "difference between", "relative to", "contrast",
}

_CROSS_DOC_KEYWORDS = {
    "across documents", "both documents", "both reports", "annual report",
    "operations report", "compare documents", "two documents", "two reports",
    "multiple documents", "multiple reports",
}

# Time period patterns: Q1-Q4, FY20XX, H1/H2, years
_PERIOD_PATTERNS = [
    r"\bQ[1-4]\b",
    r"\bH[12]\b",
    r"\bFY\s*\d{2,4}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b20\d{2}\b",
    r"\b(first|second|third|fourth)\s+quarter\b",
    r"\b(first|second)\s+half\b",
]

_STOP_WORDS = {
    "what", "where", "when", "which", "how", "many", "does", "that",
    "this", "from", "with", "have", "been", "were", "the", "and", "for",
    "are", "its", "was", "did", "can", "could", "would", "should", "will",
    "between",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contains_any(text_lower: str, word_set) -> bool:
    """
    Word-boundary-aware keyword match.
    Single-word keywords require \\b boundaries (prevents 'sum' hitting 'summarize').
    Multi-word phrases fall back to substring match (safe — they're specific enough).
    """
    for kw in word_set:
        if " " in kw:           # multi-word phrase: substring match is fine
            if kw in text_lower:
                return True
        else:                   # single word: require word boundary
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return True
    return False


def _extract_periods(text: str) -> List[str]:
    periods: List[str] = []
    for pat in _PERIOD_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            val = m.strip().upper() if isinstance(m, str) else m[0].strip().upper()
            if val and val not in periods:
                periods.append(val)
    return periods


def _extract_entities(text: str) -> List[str]:
    """Capitalized words / proper nouns, deduplicated, stop-words excluded."""
    raw = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", text)
    seen = set()
    out = []
    for e in raw:
        if e.lower() not in _STOP_WORDS and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _extract_keywords(text: str) -> List[str]:
    """Words ≥ 3 chars, deduplicated, stop-words excluded."""
    raw = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text)
    seen = set()
    out = []
    for w in raw:
        wl = w.lower()
        if wl not in _STOP_WORDS and wl not in seen:
            seen.add(wl)
            out.append(w)
    return out


def _detect_modalities(q: str) -> List[str]:
    modalities: List[str] = ["text"]          # always include text
    if _contains_any(q, _TABLE_KEYWORDS):
        if "table" not in modalities:
            modalities.append("table")
    if _contains_any(q, _CHART_KEYWORDS):
        modalities.append("chart")
    if _contains_any(q, _IMAGE_KEYWORDS) and "chart" not in modalities:
        modalities.append("image")
    return list(dict.fromkeys(modalities))     # deduplicate, preserve order


def _detect_intent(q: str) -> Tuple[str, str]:
    """
    Returns (intent, query_type).

    intent:
        comparison | aggregation | fact_lookup | visual_analysis |
        cross_doc_synthesis | calculation
    query_type:
        comparative | analytical | visual | factual
    """
    if _contains_any(q, _COMPARE_KEYWORDS):
        return "comparison", "comparative"
    if _contains_any(q, _CALC_KEYWORDS):
        return "calculation", "analytical"
    if _contains_any(q, _CHART_KEYWORDS) or _contains_any(q, _IMAGE_KEYWORDS):
        return "visual_analysis", "visual"
    if _contains_any(q, {"summarize", "summary", "overview", "explain", "describe"}):
        return "aggregation", "analytical"
    if _contains_any(q, _CROSS_DOC_KEYWORDS):
        return "cross_doc_synthesis", "comparative"
    return "fact_lookup", "factual"


# ---------------------------------------------------------------------------
# Public API 1 — backward-compatible dict (used by existing callers)
# ---------------------------------------------------------------------------

def analyze_query(question: str) -> Dict[str, Any]:
    """
    Analyze question string and output structured query parameters.

    Backward-compatible return type (Dict) so all existing callers remain
    unchanged.  Adds richer fields compared to the original implementation.
    """
    q_lower = question.lower().strip()

    needs_table = _contains_any(q_lower, _TABLE_KEYWORDS)
    needs_calculation = _contains_any(q_lower, _CALC_KEYWORDS)
    needs_visual = _contains_any(q_lower, _CHART_KEYWORDS) or _contains_any(q_lower, _IMAGE_KEYWORDS)
    cross_document = _contains_any(q_lower, _CROSS_DOC_KEYWORDS)
    intent, query_type = _detect_intent(q_lower)
    modalities = _detect_modalities(q_lower)
    periods = _extract_periods(question)
    entities = _extract_entities(question)
    keywords = _extract_keywords(question)

    return {
        # --- original fields (unchanged keys) ---
        "question": question,
        "needs_text": True,
        "needs_table": needs_table,
        "needs_calculation": needs_calculation,
        "keywords": keywords,
        "entities": entities,
        # --- new extended fields ---
        "intent": intent,
        "query_type": query_type,
        "periods": periods,
        "modalities": modalities,
        "needs_visual": needs_visual,
        "cross_document": cross_document,
    }


# ---------------------------------------------------------------------------
# Public API 2 — structured Plan (used by Adaptive Router / fetch_evidence)
# ---------------------------------------------------------------------------

def build_plan(question: str) -> Plan:
    """
    Convert a natural-language question into a fully structured Plan object.

    The Plan is the primary contract between the Query Analyzer and the
    Adaptive Router.

    Examples
    --------
    >>> p = build_plan("Compare production efficiency between Q2 and Q4.")
    >>> p.intent
    'comparison'
    >>> p.periods
    ['Q2', 'Q4']
    >>> 'table' in p.modalities
    True
    """
    analysis = analyze_query(question)

    return Plan(
        question=question,
        intent=analysis["intent"],
        entities=analysis["entities"],
        periods=analysis["periods"],
        modalities=analysis["modalities"],
        needs_calculation=analysis["needs_calculation"],
        cross_document=analysis["cross_document"],
        query_type=analysis["query_type"],
        filters={},
        sub_questions=[],
    )
