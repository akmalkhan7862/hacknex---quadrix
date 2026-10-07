"""
Phase I Unit Tests — Text & Table Readers

Covers:
  - TextReader:
      * relevant fact extraction
      * supporting text context
      * confidence scoring
      * batch_read()
      * edge cases (empty text, low overlap)
  - TableReader:
      * relevant values extraction
      * row_info & column_info structure
      * unit detection (%, $, etc.)
      * JSON table and Markdown table parsing
      * NO mathematical calculations performed (values preserved for reasoning engine)
      * batch_read()
"""

import pytest
from typing import List

from backend.app.query.readers.text import TextReader, TextReadResult, global_text_reader
from backend.app.query.readers.table import TableReader, TableReadResult, TableCellMatch, global_table_reader
from backend.app.query.schemas import EvidenceUnit


def _text_unit() -> EvidenceUnit:
    return EvidenceUnit(
        doc_id="doc1",
        document_title="Operations Report 2023",
        page=2,
        page_label="Page 2",
        section_path="Executive Summary",
        element_type="text",
        element_id="doc1-p2-t01",
        content=(
            "In Q4, production efficiency increased significantly to 94.2% across automated lines. "
            "Supply chain disruptions in Q3 were successfully mitigated by dual-sourcing. "
            "Overall annual throughput exceeded operational expectations."
        ),
    )


def _table_unit_json() -> EvidenceUnit:
    return EvidenceUnit(
        doc_id="doc1",
        document_title="Operations Report 2023",
        page=3,
        page_label="Page 3",
        section_path="Performance Metrics",
        element_type="table",
        element_id="doc1-p3-tbl01",
        content="| Quarter | Production Efficiency | Total Cost |\n| Q2 | 88.0% | $4.2M |\n| Q4 | 94.2% | $4.0M |",
        table_json='[["Quarter", "Production Efficiency", "Total Cost"], ["Q2", "88.0%", "$4.2M"], ["Q4", "94.2%", "$4.0M"]]',
        table_markdown="| Quarter | Production Efficiency | Total Cost |\n|---|---|---|\n| Q2 | 88.0% | $4.2M |\n| Q4 | 94.2% | $4.0M |",
    )


def _table_unit_md_only() -> EvidenceUnit:
    return EvidenceUnit(
        doc_id="doc2",
        document_title="Factory Review",
        page=5,
        page_label="Page 5",
        section_path="Factory Output",
        element_type="table",
        element_id="doc2-p5-tbl01",
        content="| Metric | Dallas Plant | Austin Plant |\n| Throughput | 15000 units | 12000 units |\n| Efficiency | 92% | 85% |",
        table_json=None,
        table_markdown="| Metric | Dallas Plant | Austin Plant |\n|---|---|---|\n| Throughput | 15000 units | 12000 units |\n| Efficiency | 92% | 85% |",
    )


class TestTextReader:

    def setup_method(self):
        self.reader = TextReader()
        self.unit = _text_unit()

    def test_extract_relevant_fact(self):
        result = self.reader.read("What was production efficiency in Q4?", self.unit)
        assert isinstance(result, TextReadResult)
        assert "94.2%" in result.relevant_fact
        assert "Q4" in result.relevant_fact
        assert result.confidence > 0.5
        assert result.element_id == self.unit.element_id

    def test_supporting_text_has_context(self):
        result = self.reader.read("production efficiency in Q4", self.unit)
        assert len(result.supporting_text) >= len(result.relevant_fact)
        assert "production efficiency" in result.supporting_text.lower()

    def test_batch_read_filters_non_text(self):
        units = [self.unit, _table_unit_json()]
        results = self.reader.batch_read("production efficiency", units)
        assert len(results) == 1
        assert results[0].element_id == self.unit.element_id

    def test_empty_unit_handling(self):
        empty_u = EvidenceUnit(
            doc_id="doc1", document_title="Test", page=1, page_label="P1",
            section_path="Sec", element_type="text", element_id="e1", content=""
        )
        res = self.reader.read("question", empty_u)
        assert res.relevant_fact == ""
        assert res.confidence == 0.0

    def test_global_singleton(self):
        assert global_text_reader is not None
        assert isinstance(global_text_reader, TextReader)


class TestTableReader:

    def setup_method(self):
        self.reader = TableReader()
        self.unit_json = _table_unit_json()
        self.unit_md = _table_unit_md_only()

    def test_extract_values_and_columns_from_json_table(self):
        question = "What was production efficiency in Q4?"
        result = self.reader.read(question, self.unit_json)

        assert isinstance(result, TableReadResult)
        assert "Quarter" in result.column_info
        assert "Production Efficiency" in result.column_info
        assert "94.2%" in result.relevant_values
        assert "%" in result.units
        assert result.confidence >= 0.8

        # Row info should isolate the Q4 row
        assert any(r.get("Quarter") == "Q4" for r in result.row_info)

    def test_extract_values_from_markdown_only_table(self):
        question = "What is the throughput for Dallas Plant?"
        result = self.reader.read(question, self.unit_md)

        assert isinstance(result, TableReadResult)
        assert "Dallas Plant" in result.column_info
        assert any("15000" in v for v in result.relevant_values)
        assert "units" in result.units
        assert result.confidence >= 0.8

    def test_no_arithmetic_performed(self):
        """
        Verify that TableReader extracts relevant source values only,
        without performing arithmetic calculations.
        """
        question = "Calculate the percentage increase in production efficiency from Q2 to Q4."
        result = self.reader.read(question, self.unit_json)

        # Both Q2 and Q4 efficiency values should be present
        assert "88.0%" in result.relevant_values
        assert "94.2%" in result.relevant_values

        # Ensure TableReader does NOT calculate 6.2% or ratio itself
        # The result values must be source values only
        for v in result.relevant_values:
            assert v in ("88.0%", "94.2%", "Q2", "Q4", "$4.2M", "$4.0M")

    def test_unit_detection(self):
        question = "What was the total cost in Q4?"
        result = self.reader.read(question, self.unit_json)
        assert "$" in result.units
        assert any("$4.0M" in v for v in result.relevant_values)

    def test_batch_read_filters_non_table(self):
        units = [_text_unit(), self.unit_json]
        results = self.reader.batch_read("production efficiency", units)
        assert len(results) == 1
        assert results[0].element_id == self.unit_json.element_id

    def test_global_singleton(self):
        assert global_table_reader is not None
        assert isinstance(global_table_reader, TableReader)
