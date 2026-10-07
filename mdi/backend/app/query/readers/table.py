"""
Table Reader Module for MDI.

Given:
  question + table EvidenceUnit

Returns:
  relevant values + row/column information + units + confidence

IMPORTANT:
  Does NOT perform mathematical reasoning (e.g. differences, percentages, aggregations).
  Calculations are owned by the downstream reasoning teammate.
  This module strictly extracts and structures the relevant evidence values.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.query.schemas import EvidenceUnit


class TableCellMatch(BaseModel):
    """Specific cell match corresponding to question constraints."""
    header: str
    value: str
    row_idx: int
    col_idx: int
    detected_unit: Optional[str] = None


class TableReadResult(BaseModel):
    """Structured evidence extraction from a tabular EvidenceUnit."""
    element_id: str
    relevant_values: List[str] = Field(default_factory=list)
    row_info: List[Dict[str, Any]] = Field(default_factory=list)
    column_info: List[str] = Field(default_factory=list)
    units: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    supporting_markdown: str = ""
    cell_matches: List[TableCellMatch] = Field(default_factory=list)
    doc_id: str = "unknown"
    page: int = 1
    section_path: str = "Unknown Section"


class TableReader:
    """
    Parses table data from table_json or markdown, isolates relevant rows/columns,
    extracts units and values, and returns a TableReadResult.
    """

    _UNIT_PATTERNS = [
        (r"%", "%"),
        (r"\bpercent\b", "%"),
        (r"\$", "$"),
        (r"\bUSD\b", "USD"),
        (r"\bEUR\b", "EUR"),
        (r"\bGBP\b", "GBP"),
        (r"\bunits\b", "units"),
        (r"\bkg\b", "kg"),
        (r"\btons?\b", "tons"),
        (r"\bhours?\b", "hours"),
        (r"\bdays?\b", "days"),
        (r"\bmillion\b", "million"),
        (r"\bbillion\b", "billion"),
    ]

    def _detect_units(self, text: str) -> List[str]:
        """Detect measurement or currency units in text."""
        detected = []
        for pattern, unit_name in self._UNIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if unit_name not in detected:
                    detected.append(unit_name)
        return detected

    def _parse_table(self, unit: EvidenceUnit) -> tuple[List[str], List[List[str]]]:
        """
        Parse table rows from unit.table_json or unit.table_markdown/content.
        Returns (headers, data_rows).
        """
        # 1. Try table_json
        if unit.table_json:
            try:
                data = json.loads(unit.table_json)
                if isinstance(data, list) and len(data) >= 1:
                    headers = [str(c).strip() for c in data[0]]
                    data_rows = [[str(c).strip() for c in r] for r in data[1:]]
                    return headers, data_rows
            except Exception:
                pass

        # 2. Try parsing markdown table from table_markdown or content
        raw = unit.table_markdown or unit.content or ""
        lines = [line.strip() for line in raw.split("\n") if "|" in line]
        clean_rows = []
        for line in lines:
            # Skip separator lines like |---|---|
            if re.match(r"^\|(\s*:?-+:?\s*\|)+$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if any(cells):
                clean_rows.append(cells)

        if not clean_rows:
            return [], []

        headers = clean_rows[0]
        data_rows = clean_rows[1:]
        return headers, data_rows

    def read(self, question: str, unit: EvidenceUnit) -> TableReadResult:
        """
        Extract relevant values and row/column information from a table EvidenceUnit.
        """
        headers, data_rows = self._parse_table(unit)
        if not headers and not data_rows:
            return TableReadResult(
                element_id=unit.element_id,
                relevant_values=[],
                row_info=[],
                column_info=[],
                units=[],
                confidence=0.0,
                supporting_markdown=unit.table_markdown or unit.content,
                doc_id=unit.doc_id,
                page=unit.page,
                section_path=unit.section_path,
            )

        q_lower = question.lower()
        _STOP_WORDS = {
            "what", "where", "when", "which", "how", "many", "does", "that",
            "this", "from", "with", "have", "been", "were", "the", "and", "for",
            "are", "its", "was", "did", "can", "could", "would", "should", "will",
            "calculate", "find", "determine", "show", "tell", "to", "of", "in",
            "on", "at", "by", "between", "into", "through", "is",
        }
        q_tokens = [w for w in re.findall(r"\w+", q_lower) if len(w) >= 2 and w not in _STOP_WORDS]

        matched_cells: List[TableCellMatch] = []
        matched_rows: List[Dict[str, Any]] = []
        relevant_values: List[str] = []
        detected_units: List[str] = []

        # Find which columns best match question keywords (e.g., "efficiency", "revenue", "output")
        target_col_indices = []
        for c_idx, h in enumerate(headers):
            h_words = set(re.findall(r"\w+", h.lower()))
            if any(tok in h_words or tok in h.lower() for tok in q_tokens):
                target_col_indices.append(c_idx)

        # If question asks about percentage or efficiency, target columns containing % or efficiency
        if any(p in q_lower for p in ("percentage", "percent", "%", "efficiency", "rate", "ratio")):
            for c_idx, h in enumerate(headers):
                if c_idx not in target_col_indices:
                    h_lower = h.lower()
                    has_pct = "%" in h_lower or "efficiency" in h_lower or any("%" in str(row[c_idx]) for row in data_rows if c_idx < len(row))
                    if has_pct:
                        target_col_indices.append(c_idx)

        # Scan each data row to see if it matches row qualifiers in the question (e.g., "Q4", "Q2", "Dallas")
        for r_idx, row in enumerate(data_rows):
            row_dict = {}
            for c_idx, h in enumerate(headers):
                val = row[c_idx] if c_idx < len(row) else ""
                row_dict[h] = val

            row_text = " ".join(str(v).lower() for v in row)
            row_matches = any(tok in row_text for tok in q_tokens)

            if row_matches:
                matched_rows.append(row_dict)

                # Extract relevant cell values from target columns or from all cells in matched row
                cols_to_extract = target_col_indices if target_col_indices else list(range(len(headers)))
                for c_idx in cols_to_extract:
                    if c_idx < len(row):
                        cell_val = row[c_idx]
                        header_name = headers[c_idx] if c_idx < len(headers) else f"Col_{c_idx}"

                        # Detect unit in cell value or header
                        cell_units = self._detect_units(cell_val) or self._detect_units(header_name)
                        for u in cell_units:
                            if u not in detected_units:
                                detected_units.append(u)

                        if cell_val and cell_val not in relevant_values:
                            relevant_values.append(cell_val)

                        matched_cells.append(
                            TableCellMatch(
                                header=header_name,
                                value=cell_val,
                                row_idx=r_idx,
                                col_idx=c_idx,
                                detected_unit=cell_units[0] if cell_units else None,
                            )
                        )

        # Fallback if no specific row matched: return all values from target columns
        if not relevant_values and target_col_indices:
            for r_idx, row in enumerate(data_rows):
                for c_idx in target_col_indices:
                    if c_idx < len(row):
                        val = row[c_idx]
                        if val and val not in relevant_values:
                            relevant_values.append(val)
                            header_name = headers[c_idx]
                            cell_units = self._detect_units(val) or self._detect_units(header_name)
                            matched_cells.append(
                                TableCellMatch(
                                    header=header_name,
                                    value=val,
                                    row_idx=r_idx,
                                    col_idx=c_idx,
                                    detected_unit=cell_units[0] if cell_units else None,
                                )
                            )

        # Calculate extraction confidence
        if matched_rows and target_col_indices:
            confidence = 0.95
        elif matched_rows:
            confidence = 0.80
        elif target_col_indices:
            confidence = 0.65
        else:
            confidence = 0.40

        return TableReadResult(
            element_id=unit.element_id,
            relevant_values=relevant_values,
            row_info=matched_rows,
            column_info=headers,
            units=detected_units,
            confidence=confidence,
            supporting_markdown=unit.table_markdown or unit.content,
            cell_matches=matched_cells,
            doc_id=unit.doc_id,
            page=unit.page,
            section_path=unit.section_path,
        )

    def batch_read(self, question: str, units: List[EvidenceUnit]) -> List[TableReadResult]:
        """Read multiple EvidenceUnits, returning only those of element_type == 'table'."""
        table_units = [u for u in units if u.element_type == "table"]
        return [self.read(question, u) for u in table_units]


global_table_reader = TableReader()
