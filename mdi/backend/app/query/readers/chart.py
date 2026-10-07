"""
Chart Reader Module for MDI.

Given:
  question + chart EvidenceUnit

Identifies the chart unit, preserves the visual crop reference (visual_ref),
and optionally delegates visual understanding to the VLM layer.

Never treats OCR text as a replacement for visual understanding.
Preserves the visual reference and crop path throughout the retrieval pipeline.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.query.schemas import EvidenceUnit


class ChartReadResult(BaseModel):
    """Structured extraction and visual analysis request for a chart EvidenceUnit."""
    element_id: str
    visual_ref: Optional[str] = None
    chart_type: str = "unknown"  # bar, line, pie, trend, histogram, scatter, etc.
    title_or_caption: str = ""
    visual_summary: Optional[str] = None
    confidence: float = 0.0
    doc_id: str = "unknown"
    page: int = 1
    section_path: str = "Unknown Section"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChartReader:
    """
    Chart Reader isolating chart EvidenceUnits and dispatching visual inspection
    to the VLM layer while maintaining immutable provenance and visual crop paths.
    """

    _CHART_TYPE_PATTERNS = [
        (r"\bbar\s+chart\b|\bbar\s+graph\b", "bar"),
        (r"\bline\s+chart\b|\bline\s+graph\b|\btrend\b", "line"),
        (r"\bpie\s+chart\b", "pie"),
        (r"\bscatter\s*plot\b", "scatter"),
        (r"\bhistogram\b", "histogram"),
        (r"\barea\s+chart\b", "area"),
    ]

    def __init__(self, vlm: Optional[Any] = None):
        """
        Parameters
        ----------
        vlm : VisionProvider / VLM interface | None
            Optional vision model used to analyze the visual crop.
        """
        self.vlm = vlm

    def _detect_chart_type(self, text: str) -> str:
        text_lower = text.lower()
        for pattern, chart_type in self._CHART_TYPE_PATTERNS:
            if re.search(pattern, text_lower):
                return chart_type
        return "chart"

    def read(
        self,
        question: str,
        unit: EvidenceUnit,
        request_visual_analysis: bool = True
    ) -> ChartReadResult:
        """
        Read chart EvidenceUnit, extract metadata, and request visual analysis
        from the VLM layer if a visual reference exists.
        """
        visual_ref = unit.visual_ref or unit.metadata.get("visual_ref") or unit.metadata.get("crop_path")
        chart_type = self._detect_chart_type(unit.content + " " + question)

        # Extract caption / title candidate from first sentence of content
        first_line = unit.content.split("\n")[0].strip() if unit.content else ""

        visual_summary: Optional[str] = None
        confidence = 0.70

        # Request VLM analysis if configured and visual reference is present
        if request_visual_analysis and self.vlm is not None and visual_ref:
            try:
                prompt = f"Analyze this chart regarding the question: '{question}'"
                # VLM interface can have analyze_image() or analyze()
                if hasattr(self.vlm, "analyze_image"):
                    visual_summary = self.vlm.analyze_image(visual_ref, prompt)
                elif callable(self.vlm):
                    visual_summary = self.vlm(visual_ref, prompt)
                confidence = 0.95
            except Exception as e:
                visual_summary = f"[VLM analysis unavailable: {e}]"
                confidence = 0.60
        elif visual_ref:
            # Visual crop exists and is ready for downstream VLM
            confidence = 0.85

        return ChartReadResult(
            element_id=unit.element_id,
            visual_ref=visual_ref,
            chart_type=chart_type,
            title_or_caption=first_line,
            visual_summary=visual_summary,
            confidence=confidence,
            doc_id=unit.doc_id,
            page=unit.page,
            section_path=unit.section_path,
            metadata={
                "has_visual_crop": visual_ref is not None,
                "vlm_executed": visual_summary is not None,
            }
        )

    def batch_read(self, question: str, units: List[EvidenceUnit]) -> List[ChartReadResult]:
        """Read multiple EvidenceUnits, returning only those of element_type == 'chart'."""
        chart_units = [u for u in units if u.element_type == "chart"]
        return [self.read(question, u) for u in chart_units]


global_chart_reader = ChartReader()
