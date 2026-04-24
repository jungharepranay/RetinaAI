"""
routes_clinician.py
-------------------
Clinician-facing dashboard routes.

All routes require authenticated clinician role.
Clinicians can see all patient cases.
"""

import os
import sys

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import get_current_user, login_redirect
from app.database import (
    get_all_scans, get_scan_by_id, get_dashboard_stats,
    add_clinician_note, get_notes_for_scan,
)
from app.app import templates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter(prefix="/clinician", tags=["clinician"])


def _require_clinician(request: Request):
    """Check that user is authenticated and has clinician role."""
    user = get_current_user(request)
    if not user or user["role"] != "clinician":
        return None
    return user


# ------------------------------------------------------------------ #
#  DASHBOARD                                                           #
# ------------------------------------------------------------------ #

@router.get("/dashboard", response_class=HTMLResponse)
async def clinician_dashboard(request: Request):
    """Clinician dashboard with overview stats."""
    user = _require_clinician(request)
    if not user:
        return login_redirect()

    stats = await get_dashboard_stats()
    recent_cases = await get_all_scans(sort_by="latest", limit=5)

    return templates.TemplateResponse("clinician_dashboard.html", {
        "request": request,
        "user": user,
        "stats": stats,
        "recent_cases": recent_cases,
    })


# ------------------------------------------------------------------ #
#  CASE LIST                                                           #
# ------------------------------------------------------------------ #

@router.get("/cases", response_class=HTMLResponse)
async def clinician_cases(request: Request, sort: str = "latest"):
    """List all cases with sorting options."""
    user = _require_clinician(request)
    if not user:
        return login_redirect()

    if sort not in ("latest", "risk"):
        sort = "latest"

    cases = await get_all_scans(sort_by=sort, limit=100)

    return templates.TemplateResponse("clinician_cases.html", {
        "request": request,
        "user": user,
        "cases": cases,
        "current_sort": sort,
    })


# ------------------------------------------------------------------ #
#  CASE DETAIL                                                         #
# ------------------------------------------------------------------ #

@router.get("/case/{scan_id}", response_class=HTMLResponse)
async def clinician_case_detail(request: Request, scan_id: int):
    """Full case detail view with all predictions, features, and heatmaps."""
    user = _require_clinician(request)
    if not user:
        return login_redirect()

    scan = await get_scan_by_id(scan_id)
    if not scan:
        return RedirectResponse(url="/clinician/cases", status_code=303)

    notes = await get_notes_for_scan(scan_id)

    return templates.TemplateResponse("clinician_case.html", {
        "request": request,
        "user": user,
        "scan": scan,
        "notes": notes,
    })


# ------------------------------------------------------------------ #
#  ADD NOTE                                                            #
# ------------------------------------------------------------------ #

@router.post("/case/{scan_id}/note")
async def clinician_add_note(
    request: Request,
    scan_id: int,
    note_type: str = Form("general"),
    note_text: str = Form(...),
):
    """Add a clinician note to a case."""
    user = _require_clinician(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    scan = await get_scan_by_id(scan_id)
    if not scan:
        return RedirectResponse(url="/clinician/cases", status_code=303)

    if note_text.strip():
        await add_clinician_note(
            scan_id=scan_id,
            clinician_id=user["user_id"],
            note_type=note_type,
            note_text=note_text.strip(),
        )

    return RedirectResponse(
        url=f"/clinician/case/{scan_id}#notes-section", status_code=303
    )
