"""
Main FastAPI Application Entrypoint for Multimodal Document Intelligence (MDI)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

from backend.app.db import init_db, RAW_DIR
from backend.app.api import health, documents, query

# Initialize SQLite tables on startup
init_db()

app = FastAPI(
    title="Multimodal Document Intelligence (MDI) API",
    description="HNX26PSI01 Specification Source-of-Truth Document Intelligence Backend",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(query.router, tags=["Query & Evidence"])

# Serve raw PDF files for frontend viewer
app.mount("/files", StaticFiles(directory=str(RAW_DIR)), name="files")

@app.get("/")
def root():
    return {
        "message": "Welcome to Multimodal Document Intelligence API",
        "docs_url": "/docs",
        "health_url": "/health"
    }
