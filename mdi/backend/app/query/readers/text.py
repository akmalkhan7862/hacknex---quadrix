"""
Text Reader Module for MDI.

Given:
  question + textual EvidenceUnit

Returns:
  relevant fact + supporting text + confidence

Does not perform final reasoning or hallucinations.
Extracts grounded evidence directly from unit content.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.query.schemas import EvidenceUnit


class TextReadResult(BaseModel):
    """Structured extraction from a textual EvidenceUnit."""
    element_id: str
    relevant_fact: str
    supporting_text: str
    confidence: float
    doc_id: str = "unknown"
    page: int = 1
    section_path: str = "Unknown Section"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextReader:
    """
    Extracts facts, supporting context, and confidence from textual EvidenceUnits.
    """

    _STOP_WORDS = {
        "what", "where", "when", "which", "how", "many", "does", "that",
        "this", "from", "with", "have", "been", "were", "the", "and", "for",
        "are", "its", "was", "did", "can", "could", "would", "should", "will",
    }

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return [w for w in words if w not in self._STOP_WORDS]

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return sentences if sentences else [text.strip()]

    def read(self, question: str, unit: EvidenceUnit) -> TextReadResult:
        """
        Extract the most relevant fact and supporting text from an EvidenceUnit.
        """
        content = unit.content.strip()
        if not content:
            return TextReadResult(
                element_id=unit.element_id,
                relevant_fact="",
                supporting_text="",
                confidence=0.0,
                doc_id=unit.doc_id,
                page=unit.page,
                section_path=unit.section_path,
            )

        q_tokens = set(self._tokenize(question))
        sentences = self._split_sentences(content)

        scored_sentences = []
        for idx, s in enumerate(sentences):
            s_tokens = set(self._tokenize(s))
            overlap = len(q_tokens & s_tokens)
            scored_sentences.append((overlap, idx, s))

        # Sort descending by token overlap
        scored_sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        best_overlap, best_idx, best_sentence = scored_sentences[0]

        # Calculate confidence based on overlap ratio
        if q_tokens:
            overlap_ratio = min(best_overlap / max(len(q_tokens), 1), 1.0)
            confidence = round(0.3 + 0.7 * overlap_ratio, 4) if best_overlap > 0 else 0.2
        else:
            confidence = 0.5

        # Supporting text includes the best sentence and adjacent sentences for context
        start_idx = max(0, best_idx - 1)
        end_idx = min(len(sentences), best_idx + 2)
        supporting_text = " ".join(sentences[start_idx:end_idx])

        return TextReadResult(
            element_id=unit.element_id,
            relevant_fact=best_sentence,
            supporting_text=supporting_text,
            confidence=confidence,
            doc_id=unit.doc_id,
            page=unit.page,
            section_path=unit.section_path,
            metadata={
                "overlap_tokens": best_overlap,
                "sentence_index": best_idx,
                "total_sentences": len(sentences),
            }
        )

    def batch_read(self, question: str, units: List[EvidenceUnit]) -> List[TextReadResult]:
        """Read multiple EvidenceUnits, returning only those of element_type == 'text'."""
        text_units = [u for u in units if u.element_type == "text"]
        return [self.read(question, u) for u in text_units]


global_text_reader = TextReader()
