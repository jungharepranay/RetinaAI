"""
app.py
------
FastAPI application factory.
Creates and configures the app with routes and static file serving.
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Retinal Disease Detection API",
        description="Multi-label retinal disease detection from fundus images using DenseNet121.",
        version="1.0.0",
    )

    # Mount static files
    static_dir = os.path.join(APP_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


# Template engine
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
