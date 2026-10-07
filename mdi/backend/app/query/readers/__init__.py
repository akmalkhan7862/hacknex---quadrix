"""
Readers package for MDI multimodal query evidence extraction.
"""

from backend.app.query.readers.text import TextReader, TextReadResult, global_text_reader
from backend.app.query.readers.table import TableReader, TableReadResult, TableCellMatch, global_table_reader
from backend.app.query.readers.chart import ChartReader, ChartReadResult, global_chart_reader
from backend.app.query.readers.image import ImageReader, ImageReadResult, global_image_reader

__all__ = [
    "TextReader",
    "TextReadResult",
    "global_text_reader",
    "TableReader",
    "TableReadResult",
    "TableCellMatch",
    "global_table_reader",
    "ChartReader",
    "ChartReadResult",
    "global_chart_reader",
    "ImageReader",
    "ImageReadResult",
    "global_image_reader",
]
