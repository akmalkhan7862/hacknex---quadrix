"""
Health Check API Endpoint
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Multimodal Document Intelligence",
        "version": "1.0.0"
    }
