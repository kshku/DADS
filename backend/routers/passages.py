"""Passages API — list and serve pre-existing passage PDFs (read-only)."""

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/passages", tags=["passages"])

PASSAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "App", "Passages")


@router.get("")
async def list_passages():
    """List all PDF files in the Passages directory."""
    if not os.path.exists(PASSAGES_DIR):
        return {"passages": []}

    pdfs = [f for f in os.listdir(PASSAGES_DIR) if f.lower().endswith(".pdf")]
    return {"passages": [{"name": f, "url": f"/api/passages/{f}"} for f in sorted(pdfs)]}


@router.get("/{name}")
async def get_passage(name: str):
    """Serve a passage PDF file."""
    file_path = os.path.join(PASSAGES_DIR, name)
    if not os.path.exists(file_path):
        return {"error": "Passage not found"}, 404
    return FileResponse(file_path, media_type="application/pdf")
