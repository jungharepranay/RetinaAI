"""
main.py
-------
FastAPI server entry-point for the Retinal Disease Detection system.

Backend: 3-Model Ensemble (EfficientNet-B3 + DenseNet-121 + ConvNeXt-Tiny)
         with per-class AUC-weighted soft-voting.
         Falls back to single EfficientNet-B3 if ensemble unavailable.

Endpoints
---------
GET  /                   Serve web UI
GET  /debug              Health-check / API status
POST /predict            Simple prediction (image → diseases)
POST /predict/initial    Full clinical pipeline with optional patient context
POST /predict/assess     Context-aware assessment with patient info
POST /explain            LLM-based explanation (optional)
POST /predict/refine     Legacy compatibility endpoint
POST /predict/full       Legacy compatibility endpoint

Usage::

    uvicorn app.main:app --reload
"""

import os
import sys
import io
import json
import tempfile
import shutil

import numpy as np
import torch
from fastapi import File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import create_app, templates
from src.predict import (
    predict_image, predict_initial, predict_with_context,
    _get_ensemble, USE_ENSEMBLE,
)
from src.efficientnet_model import load_efficientnet_model, DEVICE

# ----- Create app ----- #
app = create_app()

# ----- Dashboard routers ----- #
from app.routes_auth import router as auth_router
from app.routes_patient import router as patient_router
from app.routes_clinician import router as clinician_router

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(clinician_router)


@app.on_event("startup")
async def startup_event():
    """Initialize the database on server startup."""
    from app.database import init_db
    await init_db()

# ----- Model paths (single-model fallback for Grad-CAM) ----- #
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_odir_final_checkpoint.pth")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_odir_final.pth")

if os.path.isfile(CHECKPOINT_PATH):
    MODEL_PATH = CHECKPOINT_PATH
else:
    MODEL_PATH = WEIGHTS_PATH

_model = None
_thresholds = None


def _get_model():
    """Lazy-load the EfficientNet-B3 model (used for Grad-CAM fallback)."""
    global _model, _thresholds
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run the EfficientNet training notebook first."
            )
        _model, _thresholds = load_efficientnet_model(MODEL_PATH, DEVICE)
        print(f"[main] EfficientNet-B3 loaded from {MODEL_PATH}")
    return _model


# -------------------- Utilities -------------------- #

def _clean_response(obj):
    """Recursively convert numpy/torch types to Python native types."""
    if isinstance(obj, dict):
        return {k: _clean_response(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_response(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def _convert_gradcam_paths(result: dict) -> dict:
    """Convert absolute Grad-CAM paths to relative /reports/ URLs."""
    reports_root = os.path.join(PROJECT_ROOT, "reports")

    if "gradcam_paths" in result and isinstance(result["gradcam_paths"], dict):
        converted = {}
        for disease, abs_path in result["gradcam_paths"].items():
            if abs_path and os.path.isabs(abs_path):
                try:
                    rel = os.path.relpath(abs_path, reports_root)
                    converted[disease] = "/reports/" + rel.replace("\\", "/")
                except ValueError:
                    converted[disease] = abs_path
            else:
                converted[disease] = abs_path
        result["gradcam_paths"] = converted

    if result.get("gradcam_paths"):
        first_url = next(iter(result["gradcam_paths"].values()), None)
        result["gradcam_url"] = first_url
    else:
        result["gradcam_url"] = None

    if result.get("gradcam") and os.path.isabs(str(result["gradcam"])):
        try:
            rel = os.path.relpath(str(result["gradcam"]), reports_root)
            result["gradcam"] = "/reports/" + rel.replace("\\", "/")
        except ValueError:
            pass

    return result


def _parse_patient_info(
    age: str = None,
    diabetic: str = None,
    hypertension: str = None,
    vision_issues: str = None,
) -> dict:
    """Parse patient info from form fields into a clean dict."""
    info = {}
    if age:
        try:
            info["age"] = int(age)
        except ValueError:
            pass
    if diabetic:
        info["diabetic"] = diabetic.strip().lower()
    if hypertension:
        info["hypertension"] = hypertension.strip().lower()
    if vision_issues:
        info["vision_issues"] = vision_issues.strip()
    return info


# -------------------- Routes -------------------- #

@app.get("/debug")
async def debug_endpoint():
    """Health-check endpoint."""
    ensemble_info = "unavailable"
    try:
        ensemble = _get_ensemble()
        if ensemble is not None:
            ensemble_info = {
                "status": "loaded",
                "models": ensemble.model_names,
                "method": ensemble.ensemble_method,
                "num_models": len(ensemble.models),
            }
        else:
            ensemble_info = "not loaded (config missing or load failed)"
    except Exception as e:
        ensemble_info = f"error: {e}"

    return {
        "status": "API working",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "ensemble_available": USE_ENSEMBLE,
        "ensemble": ensemble_info,
        "backend": "3-Model Ensemble (EfficientNet-B3 + DenseNet-121 + ConvNeXt-Tiny)",
        "device": str(DEVICE),
        "pipeline": "Clinical Reasoning v2.0 (no score mutation)",
    }


@app.get("/")
async def root(request: Request):
    """Redirect root to dashboard (or login if not authenticated)."""
    from app.auth import get_current_user
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the main screening dashboard (auth-gated legacy UI)."""
    from app.auth import get_current_user
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
    })


@app.get("/legacy")
async def legacy_index(request: Request):
    """Backwards-compatible alias for the dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


# -------------------- Image Validation -------------------- #

@app.post("/validate-image")
async def validate_image_endpoint(file: UploadFile = File(...)):
    """Validate whether an uploaded image is a retinal fundus photograph."""
    from src.retina_validator import validate_retinal_image

    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = validate_retinal_image(image_path=tmp_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=_clean_response(result))


# -------------------- PDF Report Generation -------------------- #

@app.post("/generate-pdf")
async def generate_pdf_endpoint(
    request: Request,
    assessment: str = Form(...),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    questionnaire: str = Form("{}"),
    gradcam_path: str = Form(""),
):
    """Generate a PDF screening report from assessment data."""
    from app.pdf_report import generate_pdf_report
    from fastapi.responses import StreamingResponse

    try:
        assessment_dict = json.loads(assessment)
    except (json.JSONDecodeError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid assessment JSON."},
        )

    try:
        questionnaire_dict = json.loads(questionnaire) if questionnaire else None
    except (json.JSONDecodeError, TypeError):
        questionnaire_dict = None

    # Resolve gradcam path
    gc_abs = None
    if gradcam_path and gradcam_path.startswith("/reports/"):
        gc_abs = os.path.join(PROJECT_ROOT, gradcam_path.lstrip("/"))
        if not os.path.isfile(gc_abs):
            gc_abs = None

    try:
        pdf_bytes = generate_pdf_report(
            assessment_data=assessment_dict,
            patient_name=patient_name,
            patient_age=patient_age,
            questionnaire_data=questionnaire_dict,
            gradcam_path=gc_abs,
        )
    except ImportError as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"PDF generation failed: {e}"},
        )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=RetinAI_Report.pdf"
        },
    )


# -------------------- Save Scan to DB -------------------- #

@app.post("/save-scan")
async def save_scan_endpoint(
    request: Request,
    assessment: str = Form(...),
    patient_name: str = Form(""),
    clinical_context: str = Form("{}"),
):
    """Save a completed scan result to the database (authenticated)."""
    from app.auth import get_current_user
    from app.database import create_scan

    user = get_current_user(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required."},
        )

    try:
        assessment_dict = json.loads(assessment)
    except (json.JSONDecodeError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid assessment JSON."},
        )

    try:
        context_dict = json.loads(clinical_context) if clinical_context else {}
    except (json.JSONDecodeError, TypeError):
        context_dict = {}

    # Determine risk priority
    clinical_assessment = assessment_dict.get("clinical_assessment", {})
    risk_priority = clinical_assessment.get("urgency", "routine")

    scan_id = await create_scan(
        user_id=user["user_id"],
        role=user["role"],
        image_path="",
        prediction_output=assessment_dict,
        clinical_context=context_dict,
        risk_priority=risk_priority,
        patient_name=patient_name or user["name"],
    )

    return JSONResponse(content={"scan_id": scan_id, "status": "saved"})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Simple prediction — accept image, return detected diseases.

    Uses the 3-model ensemble when available, falls back to single model.
    """
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detected, probs = predict_image(tmp_path)
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


# -------------------- Full Clinical Pipeline -------------------- #

@app.post("/predict/initial")
async def predict_initial_endpoint(
    file: UploadFile = File(...),
    age: str = Form(None),
    diabetic: str = Form(None),
    hypertension: str = Form(None),
    vision_issues: str = Form(None),
):
    """Full clinical pipeline with optional patient context."""
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    patient_info = _parse_patient_info(age, diabetic, hypertension,
                                        vision_issues)

    try:
        result = predict_initial(tmp_path, patient_info=patient_info)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    result["filename"] = file.filename
    result = _convert_gradcam_paths(result)
    result = _clean_response(result)

    return JSONResponse(content=result)


# -------------------- Context-Aware Assessment -------------------- #

@app.post("/predict/assess")
async def predict_assess_endpoint(
    file: UploadFile = File(...),
    age: str = Form(None),
    diabetic: str = Form(None),
    hypertension: str = Form(None),
    vision_issues: str = Form(None),
):
    """Context-aware clinical assessment with patient info."""
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    patient_info = _parse_patient_info(age, diabetic, hypertension,
                                        vision_issues)

    try:
        result = predict_with_context(tmp_path, patient_info)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    result["filename"] = file.filename
    result = _convert_gradcam_paths(result)
    result = _clean_response(result)

    return JSONResponse(content=result)


# -------------------- LLM Explanation -------------------- #

@app.post("/explain")
async def explain_endpoint(
    assessment: str = Form(...),
    user_question: str = Form(None),
):
    """Generate LLM-based explanation or Q&A from clinical assessment."""
    from src.llm_explainer import (
        generate_llm_explanation,
        generate_qa_explanation,
    )

    print(f"[/explain] >> Received request -- question: {user_question}")

    try:
        assessment_dict = json.loads(assessment)
    except (json.JSONDecodeError, TypeError):
        print("[/explain] !! Invalid assessment JSON")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid assessment JSON."},
        )

    try:
        if user_question and user_question.strip():
            # Q&A mode -- answer a specific question
            print(f"[/explain] >> Q&A mode -- question: {user_question.strip()[:80]}")
            result = generate_qa_explanation(
                assessment_dict, user_question.strip()
            )
        else:
            # Summary mode -- general explanation
            print("[/explain] >> Summary mode -- generating explanation")
            result = generate_llm_explanation(assessment_dict)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[/explain] !! Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    print(f"[/explain] OK Returning response -- provider: {result.get('provider', 'unknown')}, "
          f"is_llm: {result.get('is_llm_generated', 'unknown')}")
    return JSONResponse(content=result)


# -------------------- Legacy Compatibility -------------------- #

@app.post("/predict/refine")
async def predict_refine_endpoint(
    file: UploadFile = File(...),
    answers: str = Form("{}"),
    age: str = Form(None),
    diabetic: str = Form(None),
    hypertension: str = Form(None),
    vision_issues: str = Form(None),
):
    """Legacy endpoint: redirects to context-aware assessment."""
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    patient_info = _parse_patient_info(age, diabetic, hypertension,
                                        vision_issues)

    try:
        result = predict_with_context(tmp_path, patient_info)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    result["filename"] = file.filename
    result = _convert_gradcam_paths(result)
    result = _clean_response(result)

    return JSONResponse(content=result)


@app.post("/predict/full")
async def predict_full_endpoint(file: UploadFile = File(...)):
    """Full hybrid clinical pipeline (backwards compatible)."""
    suffix = os.path.splitext(file.filename or "img.jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_initial(tmp_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)

    result["filename"] = file.filename
    result = _convert_gradcam_paths(result)
    result = _clean_response(result)

    return JSONResponse(content=result)
