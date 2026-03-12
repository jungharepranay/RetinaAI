"""
main.py
-------
FastAPI server entry-point for the Retinal Disease Detection system.

Endpoints
---------
GET  /         Serve web UI
POST /predict  Accept image upload, return detected diseases

Usage::

    uvicorn app.main:app --reload
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import tensorflow as tf
from fastapi import File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import create_app, templates
from src.predict import predict_image

# ----- Create app ----- #
app = create_app()

# ----- Load model once at startup ----- #
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "retinal_model.keras")
_model = None


def _get_model():
    """Lazy-load the model."""
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run `python src/train.py` first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[main] Model loaded from {MODEL_PATH}")
    return _model


# -------------------- Routes -------------------- #

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept a retinal fundus image and return detected diseases.

    Returns JSON::

        {
            "filename": "...",
            "detected_diseases": [...],
            "probabilities": { ... }
        }
    """
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        model = _get_model()
        detected, probs = predict_image(tmp_path, model)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "detected_diseases": detected,
        "probabilities": probs,
    }
