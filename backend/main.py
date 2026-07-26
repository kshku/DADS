"""DADS Web Backend — FastAPI application entry point."""

import os
import sys

# Ensure backend/ is on path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import analysis, passages
from starlette.requests import Request
from starlette.responses import HTMLResponse

app = FastAPI(title="DADS - Stutter Detection", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Include routers
app.include_router(passages.router)
app.include_router(analysis.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page with passages and PDF viewer."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    """Serve the audio analysis page."""
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/health")
async def health():
    """Healthcheck endpoint for Render deployment."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
