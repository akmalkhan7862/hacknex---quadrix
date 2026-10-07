"""
Readers package for MDI multimodal query evidence extraction.
"""

from backend.app.query.readers.text import TextReader, TextReadResult, global_text_reader
from backend.app.query.readers.table import TableReader, TableReadResult, TableCellMatch, global_table_reader

__all__ = [
    "TextReader",
    "TextReadResult",
    "global_text_reader",
    "TableReader",
    "TableReadResult",
    "TableCellMatch",
    "global_table_reader",
]
