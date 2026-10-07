"""
VLM (Vision-Language Model) Abstraction Module for MDI.

Defines:
  - VisionProvider (ABC)
  - VisionAnalysisResult (structured response)
  - MockVisionProvider (deterministic, zero-dependency offline fallback)
  - GeminiVisionProvider (Gemini multimodal vision)
  - OpenAIVisionProvider (GPT-4o vision)
  - get_vision_provider() factory with automatic fallback, retries, and in-memory caching.
"""

import os
import time
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field


class VisionAnalysisResult(BaseModel):
    """Structured visual output returned by a VLM provider."""
    summary: str
    visual_elements: List[str] = Field(default_factory=list)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.90
    provider_used: str = "mock"
    cached: bool = False


class VisionProvider(ABC):
    """Abstract interface for Multimodal Vision-Language Models."""

    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple[str, str], VisionAnalysisResult] = {}

    @abstractmethod
    def _execute_analysis(self, image_path: str, prompt: str) -> VisionAnalysisResult:
        """Internal execution method implemented by concrete providers."""
        pass

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image or visual crop and return textual visual summary.
        Applies caching and error handling.
        """
        res = self.analyze_structured(image_path=image_path, prompt=prompt)
        return res.summary

    def analyze_structured(self, image_path: str, prompt: str) -> VisionAnalysisResult:
        """
        Analyze an image or visual crop and return structured visual data.
        """
        cache_key = (image_path, prompt)
        if self.enable_cache and cache_key in self._cache:
            cached_res = self._cache[cache_key].model_copy()
            cached_res.cached = True
            return cached_res

        result = self._execute_analysis(image_path=image_path, prompt=prompt)

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    def __call__(self, image_path: str, prompt: str) -> str:
        """Convenience callable interface."""
        return self.analyze_image(image_path, prompt)

    def clear_cache(self):
        """Clear visual analysis cache."""
        self._cache.clear()


class MockVisionProvider(VisionProvider):
    """
    Deterministic offline Vision Provider requiring no external API keys or network access.
    Produces structured answers based on query keywords and visual filename clues.
    """

    def _execute_analysis(self, image_path: str, prompt: str) -> VisionAnalysisResult:
        p_lower = prompt.lower()
        path_lower = image_path.lower()

        visual_elements = []
        extracted_data = {}

        if "trend" in p_lower or "trend" in path_lower or "efficiency" in p_lower:
            summary = (
                "The line chart demonstrates an upward trend in production efficiency "
                "progressing from approximately 85% in Q1 to 94.2% in Q4."
            )
            visual_elements = ["line_chart", "upward_trend", "x_axis_quarters", "y_axis_percentages"]
            extracted_data = {"q1": "85%", "q4": "94.2%", "trend": "positive"}
        elif "table" in p_lower or "matrix" in p_lower:
            summary = "The visual represents a tabular grid containing performance metrics across operational units."
            visual_elements = ["data_grid", "column_headers", "numeric_values"]
        elif "photo" in p_lower or "equipment" in p_lower or "facility" in p_lower:
            summary = "The photograph captures factory floor equipment, automated robotic stations, and assembly tooling."
            visual_elements = ["manufacturing_floor", "robotic_station", "tooling"]
        else:
            summary = f"Visual analysis of image '{Path(image_path).name}' in response to query: '{prompt}'."
            visual_elements = ["general_visual_evidence"]

        return VisionAnalysisResult(
            summary=summary,
            visual_elements=visual_elements,
            extracted_data=extracted_data,
            confidence=0.92,
            provider_used="mock_vlm",
            cached=False
        )


class GeminiVisionProvider(VisionProvider):
    """
    Multimodal Vision Provider using Google GenAI (Gemini 2.5 / 1.5).
    Includes automatic retries and fallback to MockVisionProvider on failure.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", max_retries: int = 2):
        super().__init__()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self._fallback = MockVisionProvider()

    def _execute_analysis(self, image_path: str, prompt: str) -> VisionAnalysisResult:
        if not self.api_key:
            return self._fallback.analyze_structured(image_path, prompt)

        for attempt in range(self.max_retries + 1):
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)

                # If file exists, read bytes; otherwise analyze filename/reference
                file_path = Path(image_path)
                image_bytes = file_path.read_bytes() if file_path.exists() and file_path.is_file() else None

                contents = [prompt]
                if image_bytes:
                    contents.append(image_bytes)
                else:
                    contents.append(f"[Image reference: {image_path}]")

                response = client.models.generate_content(
                    model=self.model,
                    contents=contents
                )
                text = response.text.strip() if response and response.text else "No visual description generated."

                return VisionAnalysisResult(
                    summary=text,
                    visual_elements=["gemini_visual_inspection"],
                    extracted_data={"source_image": image_path},
                    confidence=0.95,
                    provider_used="gemini_vlm",
                    cached=False
                )
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"[vlm.py] Gemini VLM call failed ({e}). Falling back to MockVisionProvider.")
                return self._fallback.analyze_structured(image_path, prompt)

        return self._fallback.analyze_structured(image_path, prompt)


class OpenAIVisionProvider(VisionProvider):
    """
    Multimodal Vision Provider using OpenAI GPT-4o vision.
    Includes automatic retries and fallback to MockVisionProvider on failure.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", max_retries: int = 2):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self._fallback = MockVisionProvider()

    def _execute_analysis(self, image_path: str, prompt: str) -> VisionAnalysisResult:
        if not self.api_key:
            return self._fallback.analyze_structured(image_path, prompt)

        for attempt in range(self.max_retries + 1):
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)

                file_path = Path(image_path)
                content_payload = [{"type": "text", "text": prompt}]

                if file_path.exists() and file_path.is_file():
                    img_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                else:
                    content_payload.append({
                        "type": "text",
                        "text": f"[Referenced image: {image_path}]"
                    })

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content_payload}],
                )
                text = response.choices[0].message.content.strip()

                return VisionAnalysisResult(
                    summary=text,
                    visual_elements=["openai_visual_inspection"],
                    extracted_data={"source_image": image_path},
                    confidence=0.95,
                    provider_used="openai_vlm",
                    cached=False
                )
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"[vlm.py] OpenAI VLM call failed ({e}). Falling back to MockVisionProvider.")
                return self._fallback.analyze_structured(image_path, prompt)

        return self._fallback.analyze_structured(image_path, prompt)


def get_vision_provider(prefer_mock: bool = False) -> VisionProvider:
    """
    Factory creating configured VisionProvider based on available environment variables.
    Defaults to MockVisionProvider if no API keys are present.
    """
    if prefer_mock:
        return MockVisionProvider()

    if os.getenv("GEMINI_API_KEY"):
        return GeminiVisionProvider()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIVisionProvider()

    return MockVisionProvider()


# Global singleton instance
global_vision_provider = get_vision_provider()
