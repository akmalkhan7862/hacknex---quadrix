"""
Image Reader Module for MDI.

Given:
  question + image EvidenceUnit

Preserves the image crop/file reference (visual_ref),
and optionally delegates visual understanding to the VLM layer.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.query.schemas import EvidenceUnit


class ImageReadResult(BaseModel):
    """Structured extraction and visual analysis request for an image EvidenceUnit."""
    element_id: str
    visual_ref: Optional[str] = None
    image_description: str = ""
    visual_summary: Optional[str] = None
    confidence: float = 0.0
    doc_id: str = "unknown"
    page: int = 1
    section_path: str = "Unknown Section"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageReader:
    """
    Image Reader isolating image EvidenceUnits and dispatching visual inspection
    to the VLM layer while maintaining visual references and provenance.
    """

    def __init__(self, vlm: Optional[Any] = None):
        if vlm is not None:
            self.vlm = vlm
        else:
            try:
                from backend.app.models.vlm import global_vision_provider
                self.vlm = global_vision_provider
            except Exception:
                self.vlm = None

    def read(
        self,
        question: str,
        unit: EvidenceUnit,
        request_visual_analysis: bool = True
    ) -> ImageReadResult:
        """
        Read image EvidenceUnit, extract metadata, and request visual analysis
        from the VLM layer if a visual reference exists.
        """
        visual_ref = unit.visual_ref or unit.metadata.get("visual_ref") or unit.metadata.get("crop_path")
        description = unit.content.split("\n")[0].strip() if unit.content else ""

        visual_summary: Optional[str] = None
        confidence = 0.70

        if request_visual_analysis and self.vlm is not None and visual_ref:
            try:
                prompt = f"Analyze this image in context of the question: '{question}'"
                if hasattr(self.vlm, "analyze_image"):
                    visual_summary = self.vlm.analyze_image(visual_ref, prompt)
                elif callable(self.vlm):
                    visual_summary = self.vlm(visual_ref, prompt)
                confidence = 0.95
            except Exception as e:
                visual_summary = f"[VLM analysis unavailable: {e}]"
                confidence = 0.60
        elif visual_ref:
            confidence = 0.85

        return ImageReadResult(
            element_id=unit.element_id,
            visual_ref=visual_ref,
            image_description=description,
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

    def batch_read(self, question: str, units: List[EvidenceUnit]) -> List[ImageReadResult]:
        """Read multiple EvidenceUnits, returning only those of element_type == 'image'."""
        image_units = [u for u in units if u.element_type == "image"]
        return [self.read(question, u) for u in image_units]


global_image_reader = ImageReader()
