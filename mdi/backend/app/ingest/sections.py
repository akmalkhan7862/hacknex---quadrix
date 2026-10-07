"""
Section Detection & Hierarchy Tracker
Detects section headings from page blocks and maintains section path hierarchy.
"""

import re
from typing import List, Optional

# Regex patterns for common section headers
SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(\d+(?:\.\d+)*)\s+([A-Z0-9].*)|"  # 1. Introduction, 1.1 Overview
    r"(SECTION|CHAPTER|PART|ARTICLE)\s+(\d+|[A-Z]+)\b:?\s*(.*)|"  # SECTION 1: Details
    r"([A-Z0-9\s]{3,60})$"  # ALL CAPS HEADER
    r")",
    re.IGNORECASE
)

class SectionTracker:
    def __init__(self):
        self.current_hierarchy: List[str] = []

    def process_text_line(self, line: str) -> str:
        """
        Check if line is a heading. Update state if heading, and return current section path.
        """
        text = line.strip()
        if not text:
            return self.get_section_path()

        # Simple heuristic for headings:
        # 1. Matches numeric section format (e.g., "1. Executive Summary", "3.2 Operations")
        # 2. Short line (< 80 chars) ending with no trailing period, with title case or all caps
        
        match = SECTION_PATTERN.match(text)
        is_numeric = bool(re.match(r"^\d+(\.\d+)*\s+[A-Z]", text))
        is_title = len(text) < 75 and not text.endswith(".") and (text.isupper() or text.istitle())

        if is_numeric or is_title:
            # Clean header name
            header_name = text.rstrip(":")
            # If numeric, determine level by count of dots
            num_match = re.match(r"^(\d+(?:\.\d+)*)", header_name)
            if num_match:
                level = len(num_match.group(1).split("."))
                # Adjust stack to level - 1
                self.current_hierarchy = self.current_hierarchy[:level - 1]
                self.current_hierarchy.append(header_name)
            else:
                # Top level or sub level header
                if len(self.current_hierarchy) > 0:
                    self.current_hierarchy = [header_name]
                else:
                    self.current_hierarchy.append(header_name)

        return self.get_section_path()

    def get_section_path(self) -> str:
        if not self.current_hierarchy:
            return "Unknown Section"
        return " > ".join(self.current_hierarchy)

def extract_section_path(text: str, tracker: SectionTracker) -> str:
    """Helper to update tracker and return section path for a text chunk."""
    lines = text.strip().split("\n")
    first_line = lines[0] if lines else ""
    return tracker.process_text_line(first_line)
