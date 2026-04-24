"""
routes_patient.py
-----------------
Patient-facing dashboard routes.

All routes require authenticated patient role.
Patients can only see their own scans.
"""

import os
import sys
import json
import shutil
import tempfile
import uuid

from fastapi import APIRouter, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

from app.auth import get_current_user, login_redirect
from app.database import (
    get_scans_for_user, get_scan_by_id, create_scan,
    get_dashboard_stats, UPLOADS_DIR,
)
from app.app import templates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter(prefix="/patient", tags=["patient"])


def _require_patient(request: Request):
    """Check that user is authenticated and has patient role."""
    user = get_current_user(request)
    if not user or user["role"] != "patient":
        return None
    return user


# ------------------------------------------------------------------ #
#  DASHBOARD                                                           #
# ------------------------------------------------------------------ #

@router.get("/dashboard", response_class=HTMLResponse)
async def patient_dashboard(request: Request):
    """Patient dashboard with welcome, stats, and recent scans."""
    user = _require_patient(request)
    if not user:
        return login_redirect()

    stats = await get_dashboard_stats(user_id=user["user_id"])
    recent_scans = await get_scans_for_user(user["user_id"], limit=5)

    return templates.TemplateResponse("patient_dashboard.html", {
        "request": request,
        "user": user,
        "stats": stats,
        "recent_scans": recent_scans,
    })


# ------------------------------------------------------------------ #
#  UPLOAD                                                              #
# ------------------------------------------------------------------ #

@router.get("/upload", response_class=HTMLResponse)
async def patient_upload_page(request: Request):
    """Render the upload form for patients."""
    user = _require_patient(request)
    if not user:
        return login_redirect()

    return templates.TemplateResponse("patient_dashboard.html", {
        "request": request,
        "user": user,
        "stats": await get_dashboard_stats(user_id=user["user_id"]),
        "recent_scans": await get_scans_for_user(user["user_id"], limit=5),
        "show_upload": True,
    })


@router.post("/upload")
async def patient_upload_submit(
    request: Request,
    file: UploadFile = File(...),
    age: str = Form(None),
    diabetic: str = Form(None),
    hypertension: str = Form(None),
    vision_issues: str = Form(None),
):
    """Process patient upload: run prediction pipeline, save results."""
    user = _require_patient(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Save uploaded image permanently
    ext = os.path.splitext(file.filename or "img.jpg")[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    permanent_path = os.path.join(UPLOADS_DIR, unique_name)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    with open(permanent_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run prediction pipeline (reuse existing logic)
    try:
        from app.main import _get_model, _clean_response, _convert_gradcam_paths
        from src.predict import predict_initial

        patient_info = {}
        if age:
            try:
                patient_info["age"] = int(age)
            except ValueError:
                pass
        if diabetic:
            patient_info["diabetic"] = diabetic.strip().lower()
        if hypertension:
            patient_info["hypertension"] = hypertension.strip().lower()
        if vision_issues:
            patient_info["vision_issues"] = vision_issues.strip()

        model = _get_model()
        result = predict_initial(permanent_path, model, patient_info=patient_info)
        result["filename"] = file.filename
        result = _convert_gradcam_paths(result)
        result = _clean_response(result)

        # Determine risk priority
        assessment = result.get("clinical_assessment", {})
        risk_priority = assessment.get("urgency", "routine")

        # Save scan to database
        scan_id = await create_scan(
            user_id=user["user_id"],
            role="patient",
            image_path=f"/data/uploads/{unique_name}",
            prediction_output=result,
            clinical_context=patient_info,
            risk_priority=risk_priority,
            patient_name=user["name"],
        )

        return RedirectResponse(
            url=f"/patient/report/{scan_id}", status_code=303
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # On error, still redirect to dashboard with error param
        return RedirectResponse(
            url="/patient/dashboard?error=Analysis+failed", status_code=303
        )


# ------------------------------------------------------------------ #
#  REPORT VIEW                                                         #
# ------------------------------------------------------------------ #

@router.get("/report/{scan_id}", response_class=HTMLResponse)
async def patient_report(request: Request, scan_id: int):
    """View a patient-friendly scan report."""
    user = _require_patient(request)
    if not user:
        return login_redirect()

    scan = await get_scan_by_id(scan_id)
    if not scan or scan["user_id"] != user["user_id"]:
        return RedirectResponse(url="/patient/dashboard", status_code=303)

    return templates.TemplateResponse("patient_report.html", {
        "request": request,
        "user": user,
        "scan": scan,
    })
