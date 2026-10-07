"""
Models package for MDI multimodal foundation models and provider abstractions.
"""

from backend.app.models.vlm import (
    VisionProvider,
    VisionAnalysisResult,
    MockVisionProvider,
    GeminiVisionProvider,
    OpenAIVisionProvider,
    get_vision_provider,
    global_vision_provider,
)
from backend.app.models.embedder import (
    EmbeddingProvider,
    SentenceTransformerEmbedder,
    DeterministicFallbackEmbedder,
    MockEmbedder,
    get_embedder,
    global_embedder,
)

__all__ = [
    "VisionProvider",
    "VisionAnalysisResult",
    "MockVisionProvider",
    "GeminiVisionProvider",
    "OpenAIVisionProvider",
    "get_vision_provider",
    "global_vision_provider",
    "EmbeddingProvider",
    "SentenceTransformerEmbedder",
    "DeterministicFallbackEmbedder",
    "MockEmbedder",
    "get_embedder",
    "global_embedder",
]
