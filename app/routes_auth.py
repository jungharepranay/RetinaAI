"""
routes_auth.py
--------------
Authentication routes: login, register, logout.
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import (
    hash_password, verify_password,
    create_session, clear_session, get_current_user,
)
from app.database import create_user, get_user_by_email
from app.app import templates

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "mode": "login",
        "error": None,
    })


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    """Process login form."""
    user = await get_user_by_email(email.strip().lower())
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "mode": "login",
            "error": "Invalid email or password.",
        })

    response = RedirectResponse(url="/dashboard", status_code=303)
    create_session(response, user["id"], user["role"], user["name"])
    return response


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render registration page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "mode": "register",
        "error": None,
    })


@router.post("/register")
async def register_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
):
    """Process registration form."""
    email = email.strip().lower()
    name = name.strip()

    # Validate
    if role not in ("patient", "clinician"):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "mode": "register",
            "error": "Invalid role selected.",
        })

    if len(password) < 6:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "mode": "register",
            "error": "Password must be at least 6 characters.",
        })

    # Check duplicate
    existing = await get_user_by_email(email)
    if existing:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "mode": "register",
            "error": "An account with this email already exists.",
        })

    # Create user
    pw_hash = hash_password(password)
    user = await create_user(name, email, pw_hash, role)

    response = RedirectResponse(url="/dashboard", status_code=303)
    create_session(response, user["id"], user["role"], user["name"])
    return response


@router.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    response = RedirectResponse(url="/login", status_code=303)
    clear_session(response)
    return response
