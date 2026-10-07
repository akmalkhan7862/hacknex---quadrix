"""
Phase K Unit Tests — VLM Abstraction & Providers

Covers:
  - VisionProvider base interface
  - MockVisionProvider deterministic analysis
  - VisionAnalysisResult structured response
  - Caching mechanism (cache hit flag, clear_cache)
  - GeminiVisionProvider & OpenAIVisionProvider graceful offline fallback
  - get_vision_provider() factory & global_vision_provider singleton
  - Integration with ChartReader and ImageReader
"""

import pytest

from backend.app.models.vlm import (
    VisionProvider,
    VisionAnalysisResult,
    MockVisionProvider,
    GeminiVisionProvider,
    OpenAIVisionProvider,
    get_vision_provider,
    global_vision_provider,
)
from backend.app.query.readers.chart import ChartReader
from backend.app.query.readers.image import ImageReader
from backend.app.query.schemas import EvidenceUnit


class TestMockVisionProvider:

    def setup_method(self):
        self.vlm = MockVisionProvider(enable_cache=True)

    def test_analyze_image_returns_text(self):
        text = self.vlm.analyze_image("crops/chart1.png", "What trend is shown?")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "upward trend" in text.lower() or "efficiency" in text.lower()

    def test_analyze_structured_response(self):
        res = self.vlm.analyze_structured("crops/efficiency_chart.png", "Show efficiency trend")
        assert isinstance(res, VisionAnalysisResult)
        assert "line_chart" in res.visual_elements or "upward_trend" in res.visual_elements
        assert res.confidence > 0.8
        assert res.provider_used == "mock_vlm"
        assert res.cached is False

    def test_callable_syntax(self):
        output = self.vlm("crops/chart.png", "What does the chart show?")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_caching_behavior(self):
        path = "crops/trend.png"
        prompt = "Analyze trend"

        # First call: computed
        res1 = self.vlm.analyze_structured(path, prompt)
        assert res1.cached is False

        # Second call: cached
        res2 = self.vlm.analyze_structured(path, prompt)
        assert res2.cached is True
        assert res1.summary == res2.summary

        # Clear cache
        self.vlm.clear_cache()
        res3 = self.vlm.analyze_structured(path, prompt)
        assert res3.cached is False


class TestProviderFallbacks:

    def test_gemini_without_key_falls_back(self):
        gemini = GeminiVisionProvider(api_key=None)
        res = gemini.analyze_structured("crops/chart.png", "What is the trend?")
        assert isinstance(res, VisionAnalysisResult)
        assert len(res.summary) > 0

    def test_openai_without_key_falls_back(self):
        openai_vlm = OpenAIVisionProvider(api_key=None)
        res = openai_vlm.analyze_structured("crops/photo.png", "Describe facility photo")
        assert isinstance(res, VisionAnalysisResult)
        assert len(res.summary) > 0

    def test_get_vision_provider_factory(self):
        provider = get_vision_provider(prefer_mock=True)
        assert isinstance(provider, MockVisionProvider)

    def test_global_singleton_available(self):
        assert global_vision_provider is not None
        assert isinstance(global_vision_provider, VisionProvider)


class TestChartAndImageReaderIntegration:

    def test_chart_reader_uses_default_vlm(self):
        reader = ChartReader()
        unit = EvidenceUnit(
            doc_id="doc1",
            document_title="Report",
            page=2,
            page_label="Page 2",
            section_path="Metrics",
            element_type="chart",
            element_id="chart-1",
            content="Figure 2: Production efficiency trend over time.",
            visual_ref="crops/fig2_trend.png",
        )

        res = reader.read("What trend does the production efficiency chart show?", unit)
        assert res.visual_summary is not None
        assert "trend" in res.visual_summary.lower()
        assert res.confidence >= 0.85

    def test_image_reader_uses_default_vlm(self):
        reader = ImageReader()
        unit = EvidenceUnit(
            doc_id="doc1",
            document_title="Report",
            page=5,
            page_label="Page 5",
            section_path="Facilities",
            element_type="image",
            element_id="img-1",
            content="Photo of robot manufacturing equipment.",
            visual_ref="crops/robot.png",
        )

        res = reader.read("Describe the equipment photo", unit)
        assert res.visual_summary is not None
        assert res.confidence >= 0.85
