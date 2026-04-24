"""
auth.py
-------
Authentication helpers for RetinAI dashboard.

Uses bcrypt for password hashing and itsdangerous (bundled with Starlette)
for signed session cookies. No external session store needed.
"""

import os
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request
from fastapi.responses import RedirectResponse

# Secret key for signing cookies — generated once, persists in .env
SECRET_KEY = os.environ.get("RETINAI_SECRET_KEY", "retinai-dev-secret-key-change-in-prod")
SESSION_COOKIE = "retinai_session"
SESSION_MAX_AGE = 86400 * 7  # 7 days

_serializer = URLSafeTimedSerializer(SECRET_KEY)


# ================================================================== #
#  PASSWORD UTILS                                                      #
# ================================================================== #

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


# ================================================================== #
#  SESSION MANAGEMENT                                                  #
# ================================================================== #

def create_session(response: RedirectResponse, user_id: int, role: str, name: str) -> None:
    """Set a signed session cookie on the response."""
    session_data = {"user_id": user_id, "role": role, "name": name}
    token = _serializer.dumps(session_data)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


def get_current_user(request: Request) -> dict | None:
    """
    Read and validate the session cookie.

    Returns dict with user_id, role, name or None if invalid/missing.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None

    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
        if "user_id" in data and "role" in data:
            return data
    except (BadSignature, SignatureExpired):
        pass

    return None


def clear_session(response: RedirectResponse) -> None:
    """Clear the session cookie."""
    response.delete_cookie(key=SESSION_COOKIE)


def require_login(request: Request) -> dict | None:
    """Get current user or None. Route handlers use this to check auth."""
    return get_current_user(request)


def login_redirect() -> RedirectResponse:
    """Create a redirect response to the login page."""
    return RedirectResponse(url="/login", status_code=303)
