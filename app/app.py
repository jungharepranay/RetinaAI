"""
app.py
------
FastAPI application factory.
Creates and configures the app with routes, static file serving,
and dashboard routers.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Retinal Disease Detection API",
        description="Multi-label retinal disease detection from fundus images using EfficientNet-B3 (PyTorch).",
        version="2.0.0",
    )

    # Mount static files (CSS, JS, etc.)
    static_dir = os.path.join(APP_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Mount reports directory so Grad-CAM heatmaps are accessible
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

    # Mount data/uploads directory for stored scan images
    uploads_dir = os.path.join(DATA_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    app.mount("/data/uploads", StaticFiles(directory=uploads_dir), name="uploads")

    return app


# Template engine
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
