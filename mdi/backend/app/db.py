import sqlite3
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure data directories exist
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"

for d in [DATA_DIR, PROCESSED_DIR, RAW_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "mdi.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                document_title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                page_count INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'UPLOADED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
        """)
        
        # Pages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                page_type TEXT NOT NULL,
                quality_score REAL NOT NULL,
                text_content TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """)
        
        # Sections Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                section_path TEXT NOT NULL,
                parent_path TEXT,
                level INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """)
        
        # Evidence Units Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence_units (
                element_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                document_title TEXT NOT NULL,
                page INTEGER NOT NULL,
                page_label TEXT NOT NULL,
                section_path TEXT NOT NULL,
                element_type TEXT NOT NULL,
                content TEXT NOT NULL,
                table_json TEXT,
                table_markdown TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()

init_db()
