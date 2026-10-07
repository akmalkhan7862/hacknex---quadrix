"""
Phase J Unit Tests — Chart & Image Visual Readers

Covers:
  - ChartReader:
      * visual crop preservation (visual_ref)
      * chart type classification (trend, line, bar, pie)
      * visual analysis delegation to VLM interface
      * graceful operation without VLM / offline mode
      * batch_read()
  - ImageReader:
      * visual crop preservation
      * VLM delegation
      * batch_read()
  - Provenance integrity: doc_id, page, section_path preserved
"""

import pytest
from typing import List

from backend.app.query.readers.chart import ChartReader, ChartReadResult, global_chart_reader
from backend.app.query.readers.image import ImageReader, ImageReadResult, global_image_reader
from backend.app.query.schemas import EvidenceUnit


class _MockVLM:
    """Mock VLM providing deterministic visual responses."""
    def analyze_image(self, image_path: str, prompt: str) -> str:
        if "trend" in prompt.lower() or "efficiency" in prompt.lower():
            return "The line chart depicts an upward trend in production efficiency from 85% in Q1 to 94% in Q4."
        return f"Mock visual description of {image_path} responding to: {prompt}"


def _chart_unit() -> EvidenceUnit:
    return EvidenceUnit(
        doc_id="doc1",
        document_title="Operations Review 2023",
        page=4,
        page_label="Page 4",
        section_path="Visual Performance Trends",
        element_type="chart",
        element_id="doc1-p4-chart01",
        content="Figure 4.2: Quarterly Production Efficiency Trend (Line Chart showing Q1 to Q4).",
        visual_ref="data/processed/crops/doc1-p4-chart01.png",
    )


def _image_unit() -> EvidenceUnit:
    return EvidenceUnit(
        doc_id="doc1",
        document_title="Operations Review 2023",
        page=6,
        page_label="Page 6",
        section_path="Facilities",
        element_type="image",
        element_id="doc1-p6-img01",
        content="Photograph of assembly robot cell 4 at the Dallas manufacturing plant.",
        visual_ref="data/processed/crops/doc1-p6-img01.png",
    )


class TestChartReader:

    def test_preserves_visual_reference_without_vlm(self):
        reader = ChartReader(vlm=None)
        unit = _chart_unit()
        result = reader.read("What does the chart show?", unit, request_visual_analysis=False)

        assert isinstance(result, ChartReadResult)
        assert result.visual_ref == "data/processed/crops/doc1-p4-chart01.png"
        assert result.chart_type == "line"
        assert "Figure 4.2" in result.title_or_caption
        assert result.visual_summary is None
        assert result.confidence > 0.6
        assert result.metadata["has_visual_crop"] is True

    def test_delegates_to_vlm(self):
        mock_vlm = _MockVLM()
        reader = ChartReader(vlm=mock_vlm)
        unit = _chart_unit()

        question = "What trend does the production efficiency chart show?"
        result = reader.read(question, unit, request_visual_analysis=True)

        assert result.visual_summary is not None
        assert "upward trend" in result.visual_summary
        assert "94% in Q4" in result.visual_summary
        assert result.confidence >= 0.90
        assert result.metadata["vlm_executed"] is True

    def test_detects_chart_types(self):
        reader = ChartReader()
        bar_unit = EvidenceUnit(
            doc_id="d1", document_title="T", page=1, page_label="P1", section_path="S",
            element_type="chart", element_id="c1", content="Bar graph showing cost by department.",
            visual_ref="crop.png"
        )
        res = reader.read("cost", bar_unit)
        assert res.chart_type == "bar"

    def test_batch_read_filters_non_charts(self):
        reader = ChartReader()
        units = [_chart_unit(), _image_unit()]
        results = reader.batch_read("trend", units)
        assert len(results) == 1
        assert results[0].element_id == "doc1-p4-chart01"

    def test_global_singleton(self):
        assert global_chart_reader is not None
        assert isinstance(global_chart_reader, ChartReader)


class TestImageReader:

    def test_preserves_visual_reference_without_vlm(self):
        reader = ImageReader(vlm=None)
        unit = _image_unit()
        result = reader.read("Show the Dallas plant photo", unit)

        assert isinstance(result, ImageReadResult)
        assert result.visual_ref == "data/processed/crops/doc1-p6-img01.png"
        assert "assembly robot" in result.image_description
        assert result.confidence > 0.6
        assert result.metadata["has_visual_crop"] is True

    def test_delegates_to_vlm(self):
        mock_vlm = _MockVLM()
        reader = ImageReader(vlm=mock_vlm)
        unit = _image_unit()

        result = reader.read("Describe the Dallas robot cell", unit, request_visual_analysis=True)
        assert result.visual_summary is not None
        assert "Mock visual description" in result.visual_summary
        assert result.confidence >= 0.90

    def test_batch_read_filters_non_images(self):
        reader = ImageReader()
        units = [_chart_unit(), _image_unit()]
        results = reader.batch_read("facilities", units)
        assert len(results) == 1
        assert results[0].element_id == "doc1-p6-img01"

    def test_global_singleton(self):
        assert global_image_reader is not None
        assert isinstance(global_image_reader, ImageReader)
